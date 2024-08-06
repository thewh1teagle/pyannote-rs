use std::path::Path;

use eyre::Result;
use ndarray::{Array1, Array2};
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
        // Compute fbank mel features
        let sampling_rate = 16000.0;
        let fft_size = 400;
        let n_mels = 80;
        let f_min = None;
        let f_max = None;
        let hkt = false;
        let norm = true;
        let features_f64 =
            mel_spec::mel::mel(sampling_rate, fft_size, n_mels, f_min, f_max, hkt, norm);

        let features: Array2<f32> = features_f64.mapv(|x| x as f32);
        let mut features = features.t().to_owned(); // Transpose the array
        let mean = features
            .mean_axis(ndarray::Axis(0))
            .expect("Failed to compute mean")
            .insert_axis(ndarray::Axis(0)); // Add batch axis for broadcasting
        features -= &mean;

        let features_3d = features.insert_axis(ndarray::Axis(0)); // Add batch dimension

        // # Apply CMN (Cepstral Mean Normalization)
        // features = features - np.mean(features, axis=0)
        let inputs = ort::inputs! ["feats" => features_3d.view()]?;

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
