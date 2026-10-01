//! Runs one conversation against a model without the window, printing every event.
//! Usage: cargo run --example smoke -- <project folder> <model name> "<message>" [<conversation file>]

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use scoobert::agent::stream::Delta;
use scoobert::agent::{Event, Host};
use scoobert::store::{Approvals, Settings};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let project = PathBuf::from(&args[1]);
    let mut settings = Settings { model: args[2].clone(), approvals: Approvals::Auto, ..Settings::default() };
    // SCOOBERT_CTX forces a small context, to test summarizing.
    if let Some(ctx) = std::env::var("SCOOBERT_CTX").ok().and_then(|c| c.parse().ok()) {
        settings.context_sizes.insert(args[2].clone(), ctx);
    }
    // SCOOBERT_GPU runs the model on the graphics card.
    settings.use_gpu = std::env::var_os("SCOOBERT_GPU").is_some();
    let shared = Arc::new(RwLock::new(settings));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let host = Host::new(shared, tx);
    let file = args.get(4).filter(|s| !s.is_empty() && *s != "-").map(PathBuf::from);
    let snap = host.open(&project, file.as_deref()).expect("the conversation opens");
    println!("[open] {} {} messages, model {}", snap.file.display(), snap.messages.len(), snap.model);
    let start = Instant::now();
    host.prompt(&snap.id, args[3].clone(), Vec::new(), false).expect("the prompt starts");
    let mut first_token = None;
    while let Some(event) = rx.recv().await {
        let t = start.elapsed().as_secs_f32();
        match event {
            Event::Delta { delta: Delta::Text(s), .. } | Event::Delta { delta: Delta::Thinking(s), .. } => {
                first_token.get_or_insert(t);
                print!("{s}");
            }
            Event::Delta { delta: Delta::Progress { done, total, .. }, .. } => println!("[{t:.1}s] reading {done}/{total}"),
            Event::Message { message, .. } => println!("\n[{t:.1}s] message {}", serde_json::to_string(&message).unwrap_or_default().chars().take(400).collect::<String>()),
            Event::Settled { context, .. } => {
                println!("\n[{t:.1}s] settled, context {context} tokens, first token at {:.1}s", first_token.unwrap_or(0.0));
                break;
            }
            other => println!("[{t:.1}s] {other:?}"),
        }
    }
    // Lets the cache save and the note step finish before the server stops.
    tokio::time::sleep(std::time::Duration::from_secs(args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20))).await;
    while let Ok(event) = rx.try_recv() {
        println!("[after] {event:?}");
    }
    host.shutdown().await;
}
