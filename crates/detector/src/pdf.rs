//! PDF -> page images. Mirrors train/common.py: first 5 pages at 150 DPI.
use crate::Error;
use image::RgbImage;

#[cfg(feature = "pdf")]
const MAX_PAGES: usize = 5;

pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes[..bytes.len().min(1024)].windows(5).any(|w| w == b"%PDF-")
}

#[cfg(feature = "pdf")]
pub fn render(bytes: &[u8]) -> Result<Vec<RgbImage>, Error> {
    use pdfium_render::prelude::*;
    let pdfium = Pdfium::new(
        Pdfium::bind_to_system_library().map_err(|e| Error::Pdf(format!("pdfium library not found: {e}")))?,
    );
    let doc = pdfium.load_pdf_from_byte_slice(bytes, None).map_err(|e| Error::Pdf(e.to_string()))?;
    let cfg = PdfRenderConfig::new().scale_page_by_factor(150.0 / 72.0);
    let mut out = Vec::new();
    for page in doc.pages().iter().take(MAX_PAGES) {
        let bmp = page.render_with_config(&cfg).map_err(|e| Error::Pdf(e.to_string()))?;
        out.push(bmp.as_image().map_err(|e| Error::Pdf(e.to_string()))?.to_rgb8());
    }
    Ok(out)
}

#[cfg(not(feature = "pdf"))]
pub fn render(_bytes: &[u8]) -> Result<Vec<RgbImage>, Error> {
    Err(Error::PdfDisabled)
}
