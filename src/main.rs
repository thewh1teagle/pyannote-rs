use eyre::Result;
use std::path::Path;

mod segment;
mod session;
mod wav;

fn main() -> Result<()> {
    let model_path = Path::new("segmentation-3.0.onnx");
    let (samples, sample_rate) = wav::read_wav("motivation.wav")?;

    let segments = segment::segment(&samples, sample_rate, model_path)?;

    for (start, end) in segments {
        println!("Segment: start = {:.2}, end = {:.2}", start, end);
    }

    Ok(())
}
