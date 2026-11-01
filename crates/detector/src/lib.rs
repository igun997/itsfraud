//! AI-generated document detector: provenance scan + DINOv2 tile embeddings + linear head.
mod model;
mod pdf;
mod provenance;
mod tiles;

pub use model::{Detector, Verdict};
pub use provenance::Signal;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported or corrupt file: {0}")]
    Decode(String),
    #[error("PDF support is not compiled in (build with --features pdf and provide the pdfium library)")]
    PdfDisabled,
    #[error("PDF render failed: {0}")]
    Pdf(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
