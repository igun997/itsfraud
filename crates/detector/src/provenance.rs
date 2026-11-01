//! Cheap provenance signals read from raw file bytes. A hit is strong evidence; a miss proves
//! nothing (metadata is stripped by screenshots, messaging apps and re-photographing).
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Signal {
    pub name: &'static str,
    /// true = the file declares itself AI generated; false = informational only
    pub declares_ai: bool,
}

// (needle, name, declares_ai)
const MARKERS: &[(&[u8], &str, bool)] = &[
    // Only the IPTC digital source type is decisive: it is what C2PA/XMP use to declare generative
    // output. Generator names alone are informational (a genuine invoice *from* OpenAI/Adobe
    // mentions them).
    (b"trainedAlgorithmicMedia", "iptc:trainedAlgorithmicMedia", true),
    (b"compositeWithTrainedAlgorithmicMedia", "iptc:compositeWithTrainedAlgorithmicMedia", true),
    (b"Made with Google AI", "google-ai-label", false),
    (b"Adobe Firefly", "adobe-firefly", false),
    (b"Midjourney", "midjourney", false),
    (b"DALL-E", "dall-e", false),
    (b"DALL\xC2\xB7E", "dall-e", false),
    (b"Stable Diffusion", "stable-diffusion", false),
    (b"OpenAI", "openai-tag", false),
    // present on genuine camera output too
    (b"c2pa", "c2pa-manifest-present", false),
    (b"jumb", "jumbf-box-present", false),
];

pub fn scan(bytes: &[u8]) -> Vec<Signal> {
    let mut out: Vec<Signal> = Vec::new();
    for (needle, name, declares_ai) in MARKERS {
        if contains(bytes, needle) && !out.iter().any(|s| s.name == *name) {
            out.push(Signal { name, declares_ai: *declares_ai });
        }
    }
    out
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_declared_ai_and_ignores_plain_bytes() {
        let s = scan(b"....http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia....");
        assert!(s.iter().any(|x| x.declares_ai));
        assert!(scan(b"just some ordinary bytes").is_empty());
    }

    #[test]
    fn c2pa_alone_is_informational() {
        let s = scan(b"xx c2pa xx");
        assert_eq!(s.len(), 1);
        assert!(!s[0].declares_ai);
    }
}
