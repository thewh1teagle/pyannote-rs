use std::path::Path;

use eyre::Result;
use ndarray::Array2;
use ort::Session;

use crate::session;

pub struct EmbeddingExtractor {
    session: Session,
}

impl EmbeddingExtractor {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = session::create_session(model_path)?;
        Ok(Self { session })
    }

    pub fn compute(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let features: Array2<f32> = knf_rs::compute_fbank(samples)?;
        let features = features.insert_axis(ndarray::Axis(0)); // Add batch dimension
        let inputs = ort::inputs! ["feats" => features.view()]?;

        let ort_outs = self.session.run(inputs)?;
        let ort_out = ort_outs
            .get("embs")
            .expect("Output tensor not found")
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        let embeddings: Vec<f32> = ort_out.iter().copied().collect();

        Ok(embeddings)
    }
}
