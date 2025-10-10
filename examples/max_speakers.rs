use pyannote_rs::{create_session, EmbeddingExtractor, EmbeddingManager};

fn main() -> Result<(), eyre::Report> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;
    let max_speakers = 6;

    let mut extractor = EmbeddingExtractor::new("wespeaker_en_voxceleb_CAM++.onnx")?;
    let mut manager = EmbeddingManager::new(6);

    let mut session = create_session("segmentation-3.0.onnx")?;

    let segments = pyannote_rs::get_segments(&samples, sample_rate, &mut session)?;

    for segment in segments {
        match segment {
            Ok(segment) => {
                if let Ok(embedding) = extractor.compute(&segment.samples) {
                    let speaker = if manager.get_all_speakers().len() == max_speakers {
                        manager
                            .get_best_speaker_match(embedding.collect())
                            .map(|s| s.to_string())
                            .unwrap_or("?".into())
                    } else {
                        manager
                            .search_speaker(embedding.collect(), 0.5)
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

    Ok(())
}
