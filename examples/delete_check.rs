//! Deletes a fake downloaded model with a variant and saved prompts, and lists what is left.
//! Usage: SCOOBERT_HOME=<scratch> cargo run --release --example delete_check -- <scratch models folder>

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use scoobert::llama::LlamaServer;
use scoobert::store::Settings;

#[tokio::main]
async fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("a scratch models folder"));
    let model = dir.join("Foo-Q4_K_M");
    let other = dir.join("Bar-Q8_0");
    let variant = dir.join("Foo steered");
    for d in [&model, &other, &variant] {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(model.join("Foo-Q4_K_M.gguf"), b"GGUF").unwrap();
    std::fs::write(model.join("mmproj-F16.gguf"), b"GGUF").unwrap();
    std::fs::write(other.join("Bar-Q8_0.gguf"), b"GGUF").unwrap();
    let description = serde_json::json!({
        "name": "Foo steered", "made_from": "Foo-Q4_K_M", "family": "Foo-Q4_K_M", "changes": [],
        "own_file": null, "base_file": "Foo-Q4_K_M/Foo-Q4_K_M.gguf", "projector": null, "steering": [],
    });
    std::fs::write(variant.join("variant.json"), description.to_string()).unwrap();
    let slots = scoobert::paths::get().slots();
    std::fs::create_dir_all(&slots).unwrap();
    for name in ["Foo-Q4_K_M-32768-chat-1.bin", "Foo-Q4_K_M-extra-32768-chat-1.bin", "Bar-Q8_0-32768-chat-1.bin"] {
        std::fs::write(slots.join(name), b"x").unwrap();
    }
    let settings = Settings { models_dir: dir.to_string_lossy().into_owned(), ..Settings::default() };
    let llama = LlamaServer::new(Arc::new(RwLock::new(settings)), |_| {});
    println!("models before: {:?}", llama.models().iter().map(|m| (&m.name, m.variant.is_some())).collect::<Vec<_>>());
    println!("delete: {:?}", llama.delete_model("Foo-Q4_K_M").await.map_err(|e| format!("{e:#}")));
    println!("models after: {:?}", llama.models().iter().map(|m| &m.name).collect::<Vec<_>>());
    for entry in walk(&dir).into_iter().chain(walk(&slots)) {
        println!("  left: {}", entry.display());
    }
}

fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.path().is_dir() {
            out.push(e.path());
            out.extend(walk(&e.path()));
        } else {
            out.push(e.path());
        }
    }
    out
}
