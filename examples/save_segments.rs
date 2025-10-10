/*
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
cargo run --example save_segments 6_speakers.wav
*/

use eyre::Result;
use hound::{WavSpec, WavWriter};
use pyannote_rs::create_session;
use std::{fs, path::Path};

pub fn write_wav(file_path: &str, samples: &[i16], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(file_path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }

    Ok(())
}

fn main() -> std::result::Result<(), eyre::Report> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;

    let mut session = create_session("segmentation-3.0.onnx")?;
    let segments = pyannote_rs::get_segments(&samples, sample_rate, &mut session)?;

    // Create a folder with the base name of the input file
    let output_folder = format!(
        "{}_segments",
        Path::new(&audio_path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
    );
    fs::create_dir_all(&output_folder)?;

    for segment in segments {
        match segment {
            Ok(segment) => {
                let segment_file_name = format!(
                    "{}/start_{:.2}_end_{:.2}.wav",
                    output_folder, segment.start, segment.end
                );
                write_wav(&segment_file_name, &segment.samples, sample_rate).unwrap();
                println!("Created {}", segment_file_name);
            }
            Err(error) => eprintln!("Failed to process segment: {:?}", error),
        }
    }

    Ok(())
}
