use embedding::EmbeddingExtractor;
use eyre::Result;
use identify::EmbeddingManager;
use std::path::Path;

mod embedding;
mod identify;
mod segment;
mod session;
mod wav;

fn main() -> Result<()> {
    let model_path = Path::new("segmentation-3.0.onnx");
    let (samples, sample_rate) =
        wav::read_wav(&std::env::args().nth(1).expect("Please specify audio file"))?;

    let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();

    // FYI: it doesn't work but that's the direction.
    let mut embedding_extractor =
        EmbeddingExtractor::new(Path::new("wespeaker_en_voxceleb_CAM++.onnx")).unwrap();
    let mut embedding_manager = EmbeddingManager::new(5);

    let segments = segment::segment(&samples, sample_rate, model_path)?;

    for (start, end) in segments {
        // Convert start and end times to sample indices
        let start_idx = (start * (sample_rate as f64)) as usize;
        let end_idx = (end * (sample_rate as f64)) as usize;

        // Ensure indices are within bounds
        let start_idx = start_idx.min(samples_f32.len() - 1);
        let end_idx = end_idx.min(samples_f32.len());

        // Extract segment samples
        let segment_samples = &samples_f32[start_idx..end_idx];

        // Compute embedding
        match embedding_extractor.compute(&segment_samples) {
            Ok(embedding_result) => {
                let speaker = embedding_manager
                    .get_speaker(embedding_result, 0.5)
                    .unwrap_or(usize::MAX);
                println!(
                    "Segment: start = {:.2}, end = {:.2}, speaker = {}",
                    start, end, speaker
                );
            }
            Err(error) => {
                println!("error: {:?}", error);
            }
        }
    }

    Ok(())
}
