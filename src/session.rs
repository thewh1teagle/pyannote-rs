use eyre::Result;
use ort::{GraphOptimizationLevel, Session};

pub fn create_session() -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file("segmentation-3.0.onnx")?;
    Ok(session)
}
