use crate::session;
use eyre::{Context, ContextCompat, Result};
use ndarray::Array2;
use ort::session::Session;
use std::path::Path;

#[derive(Debug)]
pub struct EmbeddingExtractor {
    session: Session,
}

impl EmbeddingExtractor {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = session::create_session(model_path.as_ref())?;
        Ok(Self { session })
    }

    pub fn compute(&mut self, samples: &[i16]) -> Result<impl Iterator<Item = f32>> {
        // Convert to f32 precisely
        let mut samples_f32 = vec![0.0; samples.len()];
        knf_rs::convert_integer_to_float_audio(samples, &mut samples_f32);
        let samples = &samples_f32;

        let features: Array2<f32> = knf_rs::compute_fbank(samples)?;
        let features = features.insert_axis(ndarray::Axis(0)); // Add batch dimension
        let inputs = ort::inputs! ["feats" => features.view()]?;

        let ort_outs = self.session.run(inputs)?;
        let ort_out = ort_outs
            .get("embs")
            .context("Output tensor not found")?
            .try_extract_tensor::<f32>()
            .context("Failed to extract tensor")?;

        // Collect the tensor data into a Vec to own it
        let embeddings: Vec<f32> = ort_out.iter().copied().collect();

        // Return an iterator over the Vec
        Ok(embeddings.into_iter())
    }
}
