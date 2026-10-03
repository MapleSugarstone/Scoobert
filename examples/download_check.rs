//! Resolves model addresses against Hugging Face without downloading, by naming a size or file the repository lacks.
//! Usage: cargo run --release --example download_check -- <scratch folder>

use scoobert::llama::download::{ChooseSize, download};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("a scratch folder"));
    let http = reqwest::Client::new();
    for spec in [
        "unsloth/Qwen3.5-9B-GGUF:NOPE",
        "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/missing.gguf?download=true",
        "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF:NOPE",
        "not a model",
    ] {
        let result = download(&http, &dir, spec, true, "main", &CancellationToken::new(), |_| {}).await;
        match result {
            Ok(path) => println!("{spec}\n  downloaded to {}", path.display()),
            Err(e) => match e.downcast_ref::<ChooseSize>() {
                Some(c) => {
                    println!("{spec}\n  {}", c.message);
                    for s in &c.sizes {
                        println!("    {} {:.1} GB -> {}", s.label, s.bytes as f64 / 1e9, s.spec);
                    }
                }
                None => println!("{spec}\n  error: {e:#}"),
            },
        }
    }
}
