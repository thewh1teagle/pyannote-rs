use std::path::Path;

use eyre::Result;
use ort::{GraphOptimizationLevel, Session};

pub fn create_session(path: &Path) -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file(path)?;
    Ok(session)
}
