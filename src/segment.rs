use crate::session;
use eyre::{Context, ContextCompat, Result};
use ndarray::{ArrayBase, Axis, IxDyn, ViewRepr};
use std::{cmp::Ordering, path::Path};

#[derive(Debug, Clone)]
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

pub fn segment<P: AsRef<Path>>(
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
    let mut is_speeching = false;
    let mut offset = frame_start;
    let mut start_offset = 0.0;
    let mut segments = Vec::new();

    // Pad end with silence for full last segment
    let padded_samples = {
        let mut padded = Vec::from(samples);
        padded.extend(vec![0; window_size - (samples.len() % window_size)]);
        padded
    };

    for start in (0..padded_samples.len()).step_by(window_size) {
        let end = (start + window_size).min(padded_samples.len());
        let window = &padded_samples[start..end];

        // Convert window to ndarray::Array1
        let array = ndarray::Array1::from_iter(window.iter().map(|&x| x as f32));
        let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let inputs = ort::inputs![array.into_dyn()]?;
        let ort_outs = session.run(inputs)?;

        let ort_out = ort_outs
            .get("output")
            .context("Output tensor not found")?
            .try_extract_tensor::<f32>()
            .context("Failed to extract tensor")?;

        for row in ort_out.outer_iter() {
            for sub_row in row.axis_iter(Axis(0)) {
                let max_index = find_max_index(sub_row)?;

                if max_index != 0 {
                    if !is_speeching {
                        start_offset = offset as f64;
                        is_speeching = true;
                    }
                } else if is_speeching {
                    let start = start_offset / sample_rate as f64;
                    let end = offset as f64 / sample_rate as f64;

                    let start_f64 = start * (sample_rate as f64);
                    let end_f64 = end * (sample_rate as f64);

                    // Ensure indices are within bounds
                    let start_idx = start_f64.min((samples.len() - 1) as f64) as usize;
                    let end_idx = end_f64.min(samples.len() as f64) as usize;

                    let segment_samples = &padded_samples[start_idx..end_idx];

                    segments.push(Segment {
                        start,
                        end,
                        samples: segment_samples.to_vec(),
                    });
                    is_speeching = false;
                }
                offset += frame_size;
            }
        }
    }

    Ok(segments)
}
