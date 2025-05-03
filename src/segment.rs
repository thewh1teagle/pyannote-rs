use crate::session;
use eyre::{Context, ContextCompat, Result};
use ndarray::{ArrayBase, Axis, IxDyn, ViewRepr};
use std::{cmp::Ordering, path::Path};

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
) -> Result<Vec<Segment>> {
    // Create session using the provided model path
    let session = session::create_session(model_path.as_ref())?;

    // Define frame parameters
    // https://github.com/pengzhendong/pyannote-onnx/blob/c6a2460e83af0d6fa83a5570b8aa55735edbce57/pyannote_onnx/pyannote_onnx.py#L49
    let frame_size = 270;
    let frame_start = 721;
    let window_size = (sample_rate * 10) as usize; // 10 seconds
    let mut in_speech_segment = false;
    let mut offset = frame_start;
    let mut start_offset = 0.0;
    let mut segments = Vec::new();

    // Pad end with silence for full last segment
    let mut padded = Vec::from(samples);
    padded.extend(vec![0; window_size - (samples.len() % window_size)]);

    for start in (0..padded.len()).step_by(window_size) {
        let end = (start + window_size).min(padded.len());
        let window = &padded[start..end];

        // Convert window to ndarray::Array1
        let array = ndarray::Array1::from_iter(window.iter().map(|&x| x as f32));
        let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let inputs = ort::inputs![array.into_dyn()]?;
        let ort_outs = session.run(inputs)?;

        // Extract the raw output tensor and inspect its shape
        let ort_out_tensor = ort_outs.get("output").context("Output tensor not found")?;
        // Convert to f32 tensor
        let ort_out = ort_out_tensor.try_extract_tensor::<f32>().context("Failed to extract tensor")?;

        for row in ort_out.outer_iter() {
            for sub_row in row.axis_iter(Axis(0)) {
                let max_index = find_max_index(sub_row)?;

                if max_index != 0 {
                    if !in_speech_segment {
                        start_offset = offset as f64;
                        in_speech_segment = true;
                    }
                } else if in_speech_segment {
                    let start = start_offset / sample_rate as f64;
                    let end_time = offset as f64 / sample_rate as f64;
                    let start_idx = (start * sample_rate as f64).min((samples.len() - 1) as f64) as usize;
                    let end_idx = (end_time * sample_rate as f64).min(samples.len() as f64) as usize;
                    let segment_samples = &padded[start_idx..end_idx];

                    segments.push(Segment {
                        start,
                        end: end_time,
                        samples: segment_samples.to_vec(),
                    });
                    in_speech_segment = false;
                }
                offset += frame_size;
            }
        }
    }

    // Flush final segment if speech remains
    if in_speech_segment {
        let start = start_offset / sample_rate as f64;
        let end_time = offset as f64 / sample_rate as f64;
        let start_idx = (start * sample_rate as f64).min((samples.len() - 1) as f64) as usize;
        let end_idx = (end_time * sample_rate as f64).min(samples.len() as f64) as usize;
        let segment_samples = &padded[start_idx..end_idx];

        segments.push(Segment {
            start,
            end: end_time,
            samples: segment_samples.to_vec(),
        });
    }

    Ok(segments)
}
