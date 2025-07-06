use crate::session;
use eyre::{Context, ContextCompat, Result};
use ndarray::{ArrayBase, Axis, IxDyn, ViewRepr};
use std::{cmp::Ordering, collections::VecDeque, path::Path};
use std::iter;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub samples: Vec<i16>,
}

fn find_max_index(row: ArrayBase<ViewRepr<&f32>, IxDyn>) -> Result<usize> {
    let (max_index, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .context("Comparison error")
                .unwrap_or(Ordering::Equal)
        })
        .context("sub_row should not be empty")?;
    Ok(max_index)
}

pub fn get_segments<P: AsRef<Path>>(
    samples: &[i16],
    sample_rate: u32,
    model_path: P,
) -> Result<impl Iterator<Item = Result<Segment>> + '_> {
    // Create session using the provided model path
    let session = session::create_session(model_path.as_ref())?;

    // Define frame parameters
    let frame_size = 270;
    let frame_start = 721;
    let window_size = (sample_rate * 10) as usize; // 10 seconds
    let mut in_speech_segment = false;
    let mut offset = frame_start;
    let mut start_offset = 0.0;

    // Pad end with silence for full last segment
    let padded_samples = {
        let mut padded = Vec::from(samples);
        padded.extend(vec![0; window_size - (samples.len() % window_size)]);
        padded
    };

    let mut start_iter = (0..padded_samples.len()).step_by(window_size);
    let mut segments_queue = VecDeque::new();

    Ok(iter::from_fn(move || {
        loop {
            // If we have a segment in the queue, return it.
            if let Some(segment) = segments_queue.pop_front() {
                return Some(segment);
            }

            // If the queue is empty, try to process a new window of audio.
            if let Some(start) = start_iter.next() {
                let end = (start + window_size).min(padded_samples.len());
                let window = &padded_samples[start..end];

                // Convert window to ndarray::Array1
                let array = ndarray::Array1::from_iter(window.iter().map(|&x| x as f32));
                let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));

                // Handle potential errors during the session and input processing
                let inputs = match ort::inputs![array.into_dyn()] {
                    Ok(inputs) => inputs,
                    Err(e) => return Some(Err(eyre::eyre!("Failed to prepare inputs: {:?}", e))),
                };

                let ort_outs = match session.run(inputs) {
                    Ok(outputs) => outputs,
                    Err(e) => return Some(Err(eyre::eyre!("Failed to run the session: {:?}", e))),
                };

                let ort_out = match ort_outs.get("output").context("Output tensor not found") {
                    Ok(output) => output,
                    Err(e) => return Some(Err(e)),
                };

                let ort_out = match ort_out.try_extract_tensor::<f32>().context("Failed to extract tensor") {
                    Ok(tensor) => tensor,
                    Err(e) => return Some(Err(e)),
                };

                for row in ort_out.outer_iter() {
                    for sub_row in row.axis_iter(Axis(0)) {
                        let max_index = match find_max_index(sub_row) {
                            Ok(index) => index,
                            Err(e) => return Some(Err(e)),
                        };

                        if max_index != 0 {
                            if !in_speech_segment {
                                start_offset = offset as f64;
                                in_speech_segment = true;
                            }
                        } else if in_speech_segment {
                            let start = start_offset / sample_rate as f64;
                            let end = (offset as f64 + (sample_rate as f64 * 0.2)) / sample_rate as f64;
                            let start_idx = (start * sample_rate as f64).min((samples.len() - 1) as f64) as usize;
                            let mut end_idx = (end * sample_rate as f64).min(samples.len() as f64) as usize;
                            if end_idx > padded_samples.len() {
                                end_idx = padded_samples.len();
                            }

                            if start_idx < end_idx {
                                let segment_samples = &padded_samples[start_idx..end_idx];
                                segments_queue.push_back(Ok(Segment {
                                    start,
                                    end,
                                    samples: segment_samples.to_vec(),
                                }));
                            }
                            in_speech_segment = false;
                        }
                        offset += frame_size;
                    }
                }
                // After processing the window, loop again to yield any new segments.
                continue;
            }

            // No more windows to process. Flush the final segment if necessary.
            if in_speech_segment {
                let start = start_offset / sample_rate as f64;
                let end = (offset as f64 + (sample_rate as f64 * 0.2)) / sample_rate as f64;
                let start_idx = (start * sample_rate as f64).min((samples.len() - 1) as f64) as usize;
                let mut end_idx = (end * sample_rate as f64).min(samples.len() as f64) as usize;
                if end_idx > padded_samples.len() {
                    end_idx = padded_samples.len();
                }

                if start_idx < end_idx {
                    let segment_samples = &padded_samples[start_idx..end_idx];
                    segments_queue.push_back(Ok(Segment {
                        start,
                        end,
                        samples: segment_samples.to_vec(),
                    }));
                }
                in_speech_segment = false; // Mark as flushed
                // Loop again to yield the final segment.
                continue;
            }
            
            // No more windows, no final segment to flush, and queue is empty. We're done.
            return None;
        }
    }))
}
