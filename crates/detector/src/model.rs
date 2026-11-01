use std::{path::Path, sync::Mutex};

use ort::{environment::Environment, session::Session, value::Tensor};
use serde::{Deserialize, Serialize};

use crate::{pdf, provenance, tiles, Error, Signal};

#[derive(Deserialize)]
struct Head {
    mean: Vec<f32>,
    scale: Vec<f32>,
    coef: Vec<f32>,
    intercept: f32,
    threshold: f32,
}

pub struct Detector {
    _env: Environment,
    session: Mutex<Session>,
    head: Head,
}

#[derive(Debug, Serialize)]
pub struct Verdict {
    pub ai_generated: bool,
    /// Model probability that the file is AI generated (sigmoid of the mean tile logit).
    pub probability: f32,
    pub logit: f32,
    /// Decision threshold on the logit (chosen for a low false-positive rate on real files).
    pub threshold: f32,
    pub pages: usize,
    pub tiles: usize,
    pub signals: Vec<Signal>,
    /// "provenance" if the file declares itself AI generated, otherwise "model".
    pub decided_by: &'static str,
}

fn model_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Model(e.to_string())
}

impl Detector {
    /// `dir` holds backbone.onnx and head.json (written by train/export_onnx.py and train/train_head.py).
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let head: Head = serde_json::from_slice(&std::fs::read(dir.join("head.json"))?).map_err(model_err)?;
        if head.mean.len() != head.coef.len() || head.scale.len() != head.coef.len() {
            return Err(Error::Model("head.json has inconsistent lengths".into()));
        }
        let env = ort::init().build().map_err(model_err)?;
        let session = Session::builder(&env).map_err(model_err)?.commit_from_file(dir.join("backbone.onnx")).map_err(model_err)?;
        Ok(Self { _env: env, session: Mutex::new(session), head })
    }

    pub fn analyze(&self, bytes: &[u8]) -> Result<Verdict, Error> {
        let signals = provenance::scan(bytes);
        let pages = if pdf::is_pdf(bytes) {
            pdf::render(bytes)?
        } else {
            vec![image::load_from_memory(bytes).map_err(|e| Error::Decode(e.to_string()))?.to_rgb8()]
        };
        let mut data = Vec::new();
        let mut n = 0usize;
        for p in &pages {
            let (d, c) = tiles::page_tiles(p);
            data.extend(d);
            n += c;
        }
        let logit = self.mean_logit(data, n)?;
        let declared = signals.iter().any(|s| s.declares_ai);
        Ok(Verdict {
            ai_generated: declared || logit > self.head.threshold,
            probability: 1.0 / (1.0 + (-logit).exp()),
            logit,
            threshold: self.head.threshold,
            pages: pages.len(),
            tiles: n,
            signals,
            decided_by: if declared { "provenance" } else { "model" },
        })
    }

    fn mean_logit(&self, data: Vec<f32>, n: usize) -> Result<f32, Error> {
        let input = Tensor::from_array(([n, 3usize, 224, 224], data)).map_err(model_err)?;
        let mut session = self.session.lock().map_err(|_| Error::Model("session lock poisoned".into()))?;
        let outputs = session.run(ort::inputs!["pixel_values" => input]).map_err(model_err)?;
        let (shape, emb) = outputs["cls"].try_extract_tensor::<f32>().map_err(model_err)?;
        let d = self.head.coef.len();
        if shape.len() != 2 || shape[1] as usize != d || shape[0] as usize != n {
            return Err(Error::Model(format!("unexpected embedding shape {shape:?}, expected [{n}, {d}]")));
        }
        let mut sum = 0f32;
        for row in emb.chunks_exact(d) {
            let mut z = self.head.intercept;
            for i in 0..d {
                z += (row[i] - self.head.mean[i]) / self.head.scale[i] * self.head.coef[i];
            }
            sum += z;
        }
        Ok(sum / n as f32)
    }
}
