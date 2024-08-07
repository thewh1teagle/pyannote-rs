use eyre::Result;
use pyannote_rs::EmbeddingExtractor;
use pyannote_rs::EmbeddingManager;
use std::path::Path;

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let max_speakers = 6;
    let search_threshold = 0.5;

    let embedding_model_path = Path::new("wespeaker_en_voxceleb_CAM++.onnx");
    let segmentation_model_path = Path::new("segmentation-3.0.onnx");

    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;
    let mut embedding_extractor = EmbeddingExtractor::new(embedding_model_path).unwrap();
    let mut embedding_manager = EmbeddingManager::new(max_speakers);

    let segments = pyannote_rs::segment(&samples, sample_rate, segmentation_model_path)?;

    for segment in segments {
        // Compute the embedding result
        let embedding_result = match embedding_extractor.compute(&segment.samples) {
            Ok(result) => result,
            error => {
                println!("error: {:?}", error);
                println!(
                    "start = {:.2}, end = {:.2}, speaker = ?",
                    segment.start, segment.end
                );
                continue; // Skip to the next segment
            }
        };

        // Find the speaker
        let speaker = embedding_manager
            .search_speaker(embedding_result, search_threshold)
            .map(|r| r.to_string())
            .unwrap_or("?".into());

        println!(
            "start = {:.2}, end = {:.2}, speaker = {}",
            segment.start, segment.end, speaker
        );
    }

    Ok(())
}
