/*
wget https://github.com/pengzhendong/pyannote-onnx/raw/master/pyannote_onnx/segmentation-3.0.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/6_speakers.wav
cargo run --example infinite 6_speakers.wav
*/

use pyannote_rs::EmbeddingExtractor;
use pyannote_rs::EmbeddingManager;

fn main() {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let search_threshold = 0.5;

    let embedding_model_path = "wespeaker_en_voxceleb_CAM++.onnx";
    let segmentation_model_path = "segmentation-3.0.onnx";

    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path).unwrap();
    let mut embedding_extractor = EmbeddingExtractor::new(embedding_model_path).unwrap();
    let mut embedding_manager = EmbeddingManager::new(usize::MAX);

    let segments = pyannote_rs::segment(&samples, sample_rate, segmentation_model_path).unwrap();

    for segment in segments {
        // Compute the embedding result
        let embedding_result: Vec<f32> = match embedding_extractor.compute(&segment.samples) {
            Ok(result) => result.collect(),
            Err(error) => {
                println!(
                    "Error in {:.2}s: {:.2}s: {:?}",
                    segment.start, segment.end, error
                );
                println!(
                    "start = {:.2}, end = {:.2}, speaker = ?",
                    segment.start, segment.end
                );
                continue; // Skip to the next segment
            }
        };

        // Find the speaker
        let speaker = embedding_manager
            .search_speaker(embedding_result.clone(), search_threshold)
            .ok_or_else(|| embedding_manager.search_speaker(embedding_result, 0.0)) // Ensure always to return speaker
            .map(|r| r.to_string())
            .unwrap_or("?".into());

        println!(
            "start = {:.2}, end = {:.2}, speaker = {}",
            segment.start, segment.end, speaker
        );
    }
}
