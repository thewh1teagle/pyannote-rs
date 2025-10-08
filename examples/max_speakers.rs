use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};

fn main() {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path).unwrap();
    let max_speakers = 6;

    // Use the embedding model with PLDA transformations
    let mut extractor = EmbeddingExtractor::new_with_plda(
        "embedding_model.onnx",
        "models/plda/xvec_transform.npz",
        "models/plda/plda.npz",
        128, // LDA dimension
    )
    .expect("Failed to create embedding extractor with PLDA");

    let mut manager = EmbeddingManager::new(max_speakers);

    // Use the new segmentation-community-1 model
    let segments = pyannote_rs::get_segments(&samples, sample_rate, "segmentation-community-1.onnx")
        .expect("Failed to get segments");

    for segment in segments {
        match segment {
            Ok(segment) => {
                if let Ok(embedding) = extractor.compute(&segment.samples) {
                    let speaker = if manager.get_all_speakers().len() == max_speakers {
                        manager
                            .get_best_speaker_match(embedding)
                            .map(|s| s.to_string())
                            .unwrap_or("?".into())
                    } else {
                        manager
                            .search_speaker(embedding, 0.5)
                            .map(|s| s.to_string())
                            .unwrap_or("?".into())
                    };
                    println!(
                        "start = {:.2}, end = {:.2}, speaker = {}",
                        segment.start, segment.end, speaker
                    );
                } else {
                    println!(
                        "start = {:.2}, end = {:.2}, speaker = ?",
                        segment.start, segment.end
                    );
                }
            }
            Err(error) => eprintln!("Failed to process segment: {:?}", error),
        }
    }
}
