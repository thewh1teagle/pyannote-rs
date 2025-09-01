use crate::session;
use eyre::{Context, ContextCompat, Result};
use ndarray::{ArrayBase, Axis, IxDyn, ViewRepr};
use std::{cmp::Ordering, collections::{HashMap, VecDeque}, path::Path};
use std::iter;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub samples: Vec<i16>,
}

/// A helper function to find the index of the maximum value in a row of floats.
/// This is used to determine the most likely classification from the model's output.
fn find_max_index(row: ArrayBase<ViewRepr<&f32>, IxDyn>) -> Result<usize> {
    let (max_index, _) = row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b)
                .context("Comparison error: NaN or infinite value encountered")
                .unwrap_or(Ordering::Equal)
        })
        .context("Cannot find max index in an empty row")?;
    Ok(max_index)
}

pub fn get_segments<P: AsRef<Path>>(
    samples: &[i16],
    sample_rate: u32,
    model_path: P,
) -> Result<impl Iterator<Item = Result<Segment>> + '_> {
    // Create session using the provided model path
    let session = session::create_session(model_path.as_ref())?;

    // --- Overlapping Window Parameters ---
    let window_size = (sample_rate * 10) as usize; // 10-second window
    let overlap_size = (sample_rate * 1) as usize;  // 1-second overlap
    let step_size = window_size - overlap_size;     // Step forward 9 seconds each time

    // --- Model-specific Parameters ---
    let frame_size = 270; // The model's stride in samples
    let frame_start = 721; // The model's receptive field or initial offset

    // --- Simplified Segmentation Parameters ---
    let gap_tolerance_frames = 60; // Allow 30 frames (~0.5s) of silence before ending segment
    let min_segment_duration_ms = 150; // Minimum segment duration in milliseconds
    let start_buffer_ms = 400; // Generous start buffer to capture speech onset
    let end_buffer_ms = 0; // No end buffer to avoid expanding into non-speech
    
    // --- Simple State Machine Variables ---
    let mut in_speech_segment = false;
    let mut offset = frame_start; // Global offset for tracking speech start/end times
    let mut segment_start_offset = 0.0;
    let mut silence_frame_count = 0; // Track consecutive silence frames
    let mut last_segment_end = 0.0;
    let mut class_counts: HashMap<usize, usize> = HashMap::new();

    // --- Audio Padding ---
    // Pad the end with a full window of silence. This is a robust way to ensure
    // that the final window, even if it starts near the end of the original audio,
    // has enough data to be processed without going out of bounds.
    let padded_samples = {
        let mut padded = Vec::from(samples);
        padded.extend(vec![0; window_size]);
        padded
    };

    // --- Iterator Setup ---
    // The iterator now steps forward by `step_size`, creating overlapping windows.
    // We iterate up to the original length of the samples, and the padding will handle the rest.
    let mut start_iter = (0..samples.len()).step_by(step_size);
    let mut segments_queue = VecDeque::new();

    Ok(iter::from_fn(move || {
        loop {
            // If we have a complete segment in the queue, return it immediately.
            if let Some(segment) = segments_queue.pop_front() {
                return Some(segment);
            }

            // If the queue is empty, try to process a new window of audio.
            if let Some(start) = start_iter.next() {
                let end = start + window_size; // We can be sure `end` is in bounds due to padding
                let window = &padded_samples[start..end];

                // Convert window to ndarray::Array1 for the model
                let array = ndarray::Array1::from_iter(window.iter().map(|&x| x as f32));
                let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));

                // Prepare inputs for the ONNX session
                let inputs = match ort::inputs![array.into_dyn()] {
                    Ok(inputs) => inputs,
                    Err(e) => return Some(Err(eyre::eyre!("Failed to prepare inputs: {:?}", e))),
                };

                // Run the model
                let ort_outs = match session.run(inputs) {
                    Ok(outputs) => outputs,
                    Err(e) => return Some(Err(eyre::eyre!("Failed to run the session: {:?}", e))),
                };

                // Extract the output tensor
                let ort_out = match ort_outs.get("output").context("Output tensor not found") {
                    Ok(output) => output,
                    Err(e) => return Some(Err(e)),
                };
                let ort_out = match ort_out.try_extract_tensor::<f32>().context("Failed to extract tensor") {
                    Ok(tensor) => tensor,
                    Err(e) => return Some(Err(e)),
                };

                // --- Simplified State Machine Logic ---
                // Simple approach with generous buffers to ensure we capture all speech
                for row in ort_out.outer_iter() {
                    for sub_row in row.axis_iter(Axis(0)) {
                        let max_index = match find_max_index(sub_row) {
                            Ok(index) => index,
                            Err(e) => return Some(Err(e)),
                        };

                        let is_speech = max_index != 0;
                        if in_speech_segment {
                            *class_counts.entry(max_index).or_insert(0) += 1;
                        }

                        if is_speech {
                            // Reset silence counter when speech is detected
                            silence_frame_count = 0;
                            
                            if !in_speech_segment {
                                // Start a new speech segment with generous buffer
                                let start_buffer_samples = (sample_rate as f64 * start_buffer_ms as f64 / 1000.0) as usize;
                                segment_start_offset = (offset as i64 - start_buffer_samples as i64).max(0) as f64;
                                segment_start_offset = segment_start_offset.max(last_segment_end);
                                eprintln!("Starting segment at {:.2}s (offset {})", offset as f64 / sample_rate as f64, offset);
                                in_speech_segment = true;
                            }
                        } else if in_speech_segment {
                            // Increment silence counter
                            silence_frame_count += 1;
                            
                            // Only end segment if we've had enough consecutive silence frames
                            if silence_frame_count >= gap_tolerance_frames {
                                eprintln!("Ending segment at {:.2}s (offset {}) with {} silence frames", offset as f64 / sample_rate as f64, offset, silence_frame_count);
                                eprintln!("Class counts: {:?}", class_counts);
                                // End the segment with generous buffer
                                let start_sec = segment_start_offset / sample_rate as f64;
                                let end_buffer_samples = (sample_rate as f64 * end_buffer_ms as f64 / 1000.0) as usize;
                                let end_sec = (offset as f64 + end_buffer_samples as f64) / sample_rate as f64;
                                
                                // Check minimum segment duration
                                let segment_duration_ms = (end_sec - start_sec) * 1000.0;
                                
                                if segment_duration_ms >= min_segment_duration_ms as f64 {
                                    // Ensure we don't exceed the original audio bounds
                                    let start_idx = (start_sec * sample_rate as f64).max(0.0).min(samples.len() as f64) as usize;
                                    let end_idx = (end_sec * sample_rate as f64).min(samples.len() as f64) as usize;

                                    if start_idx < end_idx {
                                        let segment_samples = &samples[start_idx..end_idx];
                                        segments_queue.push_back(Ok(Segment {
                                            start: start_sec,
                                            end: end_sec,
                                            samples: segment_samples.to_vec(),
                                        }));
                                        last_segment_end = end_idx as f64;
                                    }
                                }
                                
                                in_speech_segment = false;
                                silence_frame_count = 0;
                                class_counts.clear();
                            }
                        }
                        
                        // Advance the global offset by the model's frame size
                        offset += frame_size;
                    }
                }
                // After processing the window, loop again to yield any new segments from the queue.
                continue;
            }

            // --- Finalization ---
            // No more windows to process. Flush the final segment if it's still open.
            if in_speech_segment {
                let start_sec = segment_start_offset / sample_rate as f64;
                let end_buffer_samples = (sample_rate as f64 * end_buffer_ms as f64 / 1000.0) as usize;
                let end_sec = (offset as f64 + end_buffer_samples as f64) / sample_rate as f64;
                
                // Check minimum segment duration for final segment
                let segment_duration_ms = (end_sec - start_sec) * 1000.0;
                
                if segment_duration_ms >= min_segment_duration_ms as f64 {
                    let start_idx = (start_sec * sample_rate as f64).max(0.0).min(samples.len() as f64) as usize;
                    let end_idx = (end_sec * sample_rate as f64).min(samples.len() as f64) as usize;

                    if start_idx < end_idx {
                        let segment_samples = &samples[start_idx..end_idx];
                        segments_queue.push_back(Ok(Segment {
                            start: start_sec,
                            end: end_sec,
                            samples: segment_samples.to_vec(),
                        }));
                        last_segment_end = end_idx as f64;
                        eprintln!("Final segment class counts: {:?}", class_counts);
                        class_counts.clear();
                    }
                }
                in_speech_segment = false; // Mark as flushed
                // Loop again to yield the final segment from the queue.
                continue;
            }
            
            // No more windows, no final segment to flush, and queue is empty. We're done.
            return None;
        }
    }))
}