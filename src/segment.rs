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
    let gap_tolerance_frames = 5; // Allow 10 frames (~0.1s) of silence before ending segment
    let min_segment_duration_ms = 150; // Minimum segment duration in milliseconds
    let start_buffer_ms = 0; // No start buffer to avoid pre-speech padding
    let end_buffer_ms = 0; // No end buffer to avoid expanding into non-speech
    // Precompute buffer sizes in samples
    let start_buffer_samples = (sample_rate as f64 * start_buffer_ms as f64 / 1000.0) as usize;
    let end_buffer_samples = (sample_rate as f64 * end_buffer_ms as f64 / 1000.0) as usize;
    // Require a short run of speech frames before starting (time-based -> frames)
    let start_hysteresis_ms = 500; // delay start until ~0.5s of consecutive speech
    let start_hysteresis_frames = ((sample_rate as f64 * start_hysteresis_ms as f64 / 1000.0) / frame_size as f64)
        .max(1.0)
        .round() as usize;
    
    // --- Simple State Machine Variables ---
    let mut in_speech_segment = false;
    let mut seg_start_samples: usize = 0; // Segment start in samples
    let mut silence_frame_count: usize = 0; // Track consecutive silence frames
    let mut last_segment_end_samples: usize = 0; // Last finalized segment end in samples
    let mut last_emitted_offset: usize = 0; // Last processed absolute offset to dedupe overlaps
    let mut class_counts: HashMap<usize, usize> = HashMap::new();
    let mut speech_run: usize = 0; // consecutive speech frames (for hysteresis)

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
                    for (frame_idx, sub_row) in row.axis_iter(Axis(0)).into_iter().enumerate() {
                        let max_index = match find_max_index(sub_row) {
                            Ok(index) => index,
                            Err(e) => return Some(Err(e)),
                        };

                        // Absolute offset of this frame in input samples (dedup overlap)
                        let abs_offset = start + frame_start + frame_idx * frame_size;
                        if abs_offset <= last_emitted_offset {
                            continue;
                        }

                        let is_speech = max_index != 0;
                        if in_speech_segment {
                            *class_counts.entry(max_index).or_insert(0) += 1;
                        }

                        if is_speech {
                            // Reset silence counter and grow speech run
                            silence_frame_count = 0;
                            speech_run += 1;

                            if !in_speech_segment && speech_run >= start_hysteresis_frames {
                                // Backtrack to the start of this speech run, then apply start buffer
                                let first_abs_offset = abs_offset.saturating_sub((speech_run - 1) * frame_size);
                                let mut start_samp = first_abs_offset.saturating_sub(start_buffer_samples);
                                start_samp = start_samp.max(last_segment_end_samples);
                                seg_start_samples = start_samp;
                                eprintln!("Starting segment at {:.2}s (abs_offset {})", seg_start_samples as f64 / sample_rate as f64, abs_offset);
                                in_speech_segment = true;
                            }
                        } else {
                            // Non-speech: reset speech run, and if inside a segment, count silence and possibly end
                            speech_run = 0;
                            if in_speech_segment {
                                // Increment silence counter
                                silence_frame_count += 1;
                                // Only end segment if we've had enough consecutive silence frames
                                if silence_frame_count >= gap_tolerance_frames {
                                    let end_idx = (abs_offset + end_buffer_samples).min(samples.len());
                                    let start_idx = seg_start_samples.min(end_idx);
                                    let segment_duration_ms = ((end_idx.saturating_sub(start_idx)) as f64) * 1000.0 / sample_rate as f64;
                                    eprintln!("Ending segment at {:.2}s (abs_offset {}) with {} silence frames", end_idx as f64 / sample_rate as f64, abs_offset, silence_frame_count);
                                    eprintln!("Class counts: {:?}", class_counts);
                                    if segment_duration_ms >= min_segment_duration_ms as f64 && start_idx < end_idx {
                                        let start_sec = start_idx as f64 / sample_rate as f64;
                                        let end_sec = end_idx as f64 / sample_rate as f64;
                                        let segment_samples = &samples[start_idx..end_idx];
                                        segments_queue.push_back(Ok(Segment {
                                            start: start_sec,
                                            end: end_sec,
                                            samples: segment_samples.to_vec(),
                                        }));
                                        last_segment_end_samples = end_idx;
                                    }
                                    in_speech_segment = false;
                                    silence_frame_count = 0;
                                    class_counts.clear();
                                }
                            }
                        }
                        
                        // Track last processed absolute offset (dedupe overlap)
                        last_emitted_offset = abs_offset;
                    }
                }
                // After processing the window, loop again to yield any new segments from the queue.
                continue;
            }

            // --- Finalization ---
            // No more windows to process. Flush the final segment if it's still open.
            if in_speech_segment {
                let start_idx = seg_start_samples.min(samples.len());
                let end_idx = (last_emitted_offset + end_buffer_samples).min(samples.len());
                if end_idx > start_idx {
                    let segment_duration_ms = ((end_idx - start_idx) as f64) * 1000.0 / sample_rate as f64;
                    if segment_duration_ms >= min_segment_duration_ms as f64 {
                        let start_sec = start_idx as f64 / sample_rate as f64;
                        let end_sec = end_idx as f64 / sample_rate as f64;
                        let segment_samples = &samples[start_idx..end_idx];
                        segments_queue.push_back(Ok(Segment {
                            start: start_sec,
                            end: end_sec,
                            samples: segment_samples.to_vec(),
                        }));
                        last_segment_end_samples = end_idx;
                        eprintln!("Final segment class counts: {:?}", class_counts);
                        class_counts.clear();
                    }
                }
                in_speech_segment = false; // Mark as flushed
                continue;
            }
            
            // No more windows, no final segment to flush, and queue is empty. We're done.
            return None;
        }
    }))
}