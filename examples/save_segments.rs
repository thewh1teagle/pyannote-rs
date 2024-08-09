/*
wget https://github.com/pengzhendong/pyannote-onnx/raw/master/pyannote_onnx/segmentation-3.0.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/6_speakers.wav
cargo run --example save_segments 6_speakers.wav
*/

use eyre::Result;
use hound::{WavSpec, WavWriter};
use std::fs;
use std::path::Path;

pub fn write_wav(file_path: &str, samples: &[i16], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1, // Mono audio
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(file_path, spec)?;

    // Write the samples to the WAV file
    for &sample in samples {
        writer.write_sample(sample)?;
    }

    Ok(())
}

fn main() {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");

    let segmentation_model_path = "segmentation-3.0.onnx";

    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path).unwrap();

    let segments = pyannote_rs::segment(&samples, sample_rate, segmentation_model_path).unwrap();

    // Create a folder with the base name of the input file
    let file_stem = Path::new(&audio_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let output_folder = format!("{}_segments", file_stem);
    fs::create_dir_all(&output_folder).unwrap();

    for segment in segments {
        // Compute the embedding result
        // Save the segment to a .wav file
        let segment_file_name = format!(
            "{}/start_{:.2}_end_{:.2}.wav",
            output_folder, segment.start, segment.end
        );
        write_wav(&segment_file_name, &segment.samples, sample_rate).unwrap();
        println!("Created {}", segment_file_name);
    }
}
