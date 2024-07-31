use std::cmp;

use eyre::Result;
use ndarray::Axis;
mod session;
mod utils;
mod wav;

fn main() -> Result<()> {
    let session = session::create_session()?;
    let (samples, sample_rate) = wav::read_wav("motivation.wav")?;

    let num_classes = 3;
    let duration = (10 * sample_rate) as f64;

    let step = 5 * sample_rate;
    let step = f64::max(
        f64::min(step as f64, 0.9 * duration as f64),
        (duration / 2.0),
    );
    let overlap = utils::sample2frame(duration - step) as usize;
    let overlap_chunk = vec![vec![0.0; num_classes]; overlap];

    let samples = &samples[..(sample_rate * 10) as usize].to_vec();
    let array = ndarray::Array1::from_iter(samples.iter().map(|&x| x as f32));
    let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));
    let inputs = ort::inputs![array.into_dyn()]?;
    let ort_outs = session.run(inputs)?;

    for output in ort_outs.iter() {
        // Implement processing similar to Python `reorder` and `yield` steps
        // For example:
        println!("{:?}", output);
    }
    Ok(())
}
