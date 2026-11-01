//! itsfraud scan <file>            score one image/PDF and print JSON
//! itsfraud serve [addr]           HTTP API: POST /detect (multipart field "file"), GET /health
//! Model files are read from $ITSFRAUD_ARTIFACTS (default ./artifacts).
use std::{path::PathBuf, sync::Arc};

use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use detector::Detector;
use serde_json::{json, Value};

const MAX_UPLOAD: usize = 25 * 1024 * 1024;

fn artifacts_dir() -> PathBuf {
    std::env::var_os("ITSFRAUD_ARTIFACTS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("artifacts"))
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let det = match Detector::load(&artifacts_dir()) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("cannot load model from {}: {e}", artifacts_dir().display());
            std::process::exit(2);
        }
    };
    match args.next().as_deref() {
        Some("scan") => {
            let path = args.next().unwrap_or_else(|| usage());
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                eprintln!("{path}: {e}");
                std::process::exit(1)
            });
            match det.analyze(&bytes) {
                Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Some("serve") => {
            let addr = args.next().unwrap_or_else(|| "127.0.0.1:8080".into());
            let app = Router::new()
                .route("/health", get(|| async { "ok" }))
                .route("/detect", post(detect))
                .layer(DefaultBodyLimit::max(MAX_UPLOAD))
                .with_state(det);
            let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
            eprintln!("listening on http://{addr}");
            axum::serve(listener, app).await.expect("serve");
        }
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: itsfraud scan <file> | itsfraud serve [addr]");
    std::process::exit(64)
}

async fn detect(State(det): State<Arc<Detector>>, mut mp: Multipart) -> (StatusCode, Json<Value>) {
    let bad = |m: String| (StatusCode::BAD_REQUEST, Json(json!({ "error": m })));
    let bytes = loop {
        match mp.next_field().await {
            Ok(Some(f)) if f.name() == Some("file") => match f.bytes().await {
                Ok(b) => break b,
                Err(e) => return bad(format!("upload failed: {e}")),
            },
            Ok(Some(_)) => continue,
            Ok(None) => return bad("missing multipart field \"file\"".into()),
            Err(e) => return bad(format!("bad multipart body: {e}")),
        }
    };
    // inference is CPU-bound and blocking
    match tokio::task::spawn_blocking(move || det.analyze(&bytes)).await {
        Ok(Ok(v)) => (StatusCode::OK, Json(serde_json::to_value(v).unwrap())),
        Ok(Err(e @ (detector::Error::Decode(_) | detector::Error::PdfDisabled | detector::Error::Pdf(_)))) => {
            (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "error": e.to_string() })))
        }
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}
