use eyre::Result;
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;
    let max_speakers = 6;

    let mut extractor = EmbeddingExtractor::new("wespeaker_en_voxceleb_CAM++.onnx")?;
    let mut manager = EmbeddingManager::new(6);

    let segments = pyannote_rs::get_segments(&samples, sample_rate, "segmentation-3.0.onnx")?;

    for segment in segments {
        match segment {
            Ok(segment) => {
                let speaker = extractor.compute(&segment.samples).ok().map(|embedding| {
                    let result = if manager.get_all_speakers().len() == max_speakers as usize {
                        manager.get_best_speaker_match(embedding.collect())
                    } else {
                        manager
                            .search_speaker(embedding.collect(), 0.5)
                            .ok_or(eyre::eyre!("No matching speaker found with threshold 0.5"))
                    };
                    result.map(|s| s.to_string()).unwrap_or("?".into())
                });

                println!(
                    "start = {:.2}, end = {:.2}, speaker = {}",
                    segment.start,
                    segment.end,
                    speaker.unwrap_or("???".into())
                );
            }
            Err(error) => eprintln!("Failed to process segment: {:?}", error),
        }
    }

    Ok(())
}
