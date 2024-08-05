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
    let (samples, sample_rate) = wav::read_wav(&std::env::args().nth(1).expect("Please specify audio file"))?;

    let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();

    // FYI: it doesn't work but that's the direction.
    let mut embedding_extractor =
        EmbeddingExtractor::new(Path::new("nemo_en_titanet_small.onnx")).unwrap();
    let mut embedding_manager = EmbeddingManager::new(100);

    let segments = segment::segment(&samples, sample_rate, model_path)?;

    for (start, end) in segments {
        // Compute index
        let start_idx = start as usize;
        let end_idx = end as usize;
        let start_idx = start_idx.min(samples_f32.len() - 1);
        let end_idx = end_idx.min(samples_f32.len());
        let segment_samples = &samples_f32[start_idx..end_idx];
        let embedding_result = embedding_extractor.compute(&segment_samples).unwrap();
        let speaker = embedding_manager
            .get_speaker(embedding_result, 0.9)
            .unwrap_or(usize::MAX);
        println!(
            "Segment: start = {:.2}, end = {:.2}, speaker = {}",
            start, end, speaker
        );
    }

    Ok(())
}
