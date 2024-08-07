use eyre::Result;
use pyannote_rs::EmbeddingExtractor;
use pyannote_rs::EmbeddingManager;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = Path::new("segmentation-3.0.onnx");
    let (samples, sample_rate) =
        pyannote_rs::read_wav(&std::env::args().nth(1).expect("Please specify audio file"))?;

    let mut embedding_extractor =
        EmbeddingExtractor::new(Path::new("wespeaker_en_voxceleb_CAM++.onnx")).unwrap();
    let mut embedding_manager = EmbeddingManager::new(6);

    let segments = pyannote_rs::segment(&samples, sample_rate, model_path)?;

    for segment in segments {
        // Compute embedding

        match embedding_extractor.compute(&segment.samples) {
            Ok(embedding_result) => {
                let speaker = embedding_manager
                    .search_speaker(embedding_result, 0.5)
                    .map(|r| r.to_string())
                    .unwrap_or("?".into());
                println!(
                    "start = {:.2}, end = {:.2}, speaker = {}",
                    segment.start, segment.end, speaker
                );
            }
            Err(error) => {
                println!(
                    "start = {:.2}, end = {:.2}, speaker = {}",
                    segment.start, segment.end, "?"
                );
                println!("error: {:?}", error);
            }
        }
    }

    Ok(())
}
