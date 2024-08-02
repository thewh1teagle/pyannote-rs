use crate::session;
use eyre::Result;
use ndarray::Axis;
use std::path::Path;

pub fn segment(samples: &[i16], sample_rate: u32, model_path: &Path) -> Result<Vec<(f64, f64)>> {
    // Create session using the provided model path
    let session = session::create_session(model_path)?;

    // Define frame parameters
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
            .expect("Output tensor not found")
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        for (_, row) in ort_out.outer_iter().enumerate() {
            for (_, sub_row) in row.axis_iter(Axis(0)).enumerate() {
                let (max_index, _) = sub_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .expect("sub_row should not be empty");

                if max_index != 0 {
                    if !is_speeching {
                        start_offset = offset as f64;
                        is_speeching = true;
                    }
                } else if is_speeching {
                    let start = start_offset / sample_rate as f64;
                    let end = offset as f64 / sample_rate as f64;
                    segments.push((start, end));
                    is_speeching = false;
                }
                offset += frame_size;
            }
        }
    }

    Ok(segments)
}
