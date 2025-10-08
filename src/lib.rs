mod session;

mod embedding;
mod identify;
mod plda;
mod segment;
mod wav;

pub use embedding::EmbeddingExtractor;
pub use identify::EmbeddingManager;
pub use knf_rs::{compute_fbank, convert_integer_to_float_audio};
pub use plda::PLDA;
pub use segment::{get_segments, Segment};
pub use wav::read_wav;
