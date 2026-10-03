//! Sends a message to a model that cannot see images, then one with an image attached, and prints how the image
//! helper describes it and how long each step takes.
//! Usage: SCOOBERT_HOME=<scratch> cargo run --release --example image_check -- <project folder> <model name> <image file>

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use scoobert::agent::conversation::{Image, Message};
use scoobert::agent::{Event, Host};
use scoobert::store::{Approvals, Settings, Thinking};
use tokio::sync::mpsc::UnboundedReceiver;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let project = PathBuf::from(&args[1]);
    let settings = Settings { model: args[2].clone(), approvals: Approvals::Auto, thinking: Thinking::Off, ..Settings::default() };
    use base64::Engine;
    let image = Image { mime: "image/jpeg".into(), data: base64::engine::general_purpose::STANDARD.encode(std::fs::read(&args[3]).unwrap()) };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let host = Host::new(Arc::new(RwLock::new(settings)), tx);
    let snap = host.open(&project, None).expect("the conversation opens");
    let start = Instant::now();
    println!("== a message without an image");
    host.prompt(&snap.id, "Reply with one short sentence: what is 2 + 2?".into(), Vec::new(), false).unwrap();
    run(&mut rx, start).await;
    println!("== a message with an image");
    host.prompt(&snap.id, "What does the attached screenshot show? Answer in two sentences.".into(), vec![image], false).unwrap();
    run(&mut rx, start).await;
    host.shutdown().await;
}

async fn run(rx: &mut UnboundedReceiver<Event>, start: Instant) {
    let mut last = String::new();
    loop {
        let event = match tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            Ok(Some(e)) => e,
            Ok(None) => return,
            Err(_) => continue,
        };
        let t = start.elapsed().as_secs();
        match event {
            Event::Activity { text: Some(text), .. } => {
                // The loading line counts seconds, so only its first form is shown.
                let head: String = text.chars().take(24).collect();
                if head != last {
                    println!("[{t}s] {text}");
                    last = head;
                }
            }
            Event::Message { message: Message::User(u), .. } => {
                if let Some(i) = u.context.find("<image_description>") {
                    println!("[{t}s] description in the message:\n{}", &u.context[i..]);
                }
            }
            Event::Message { message: Message::Assistant(a), .. } => println!("[{t}s] reply: {}", a.text.trim()),
            Event::Error { message, .. } => println!("[{t}s] error: {message}"),
            Event::Settled { .. } => {
                println!("[{t}s] settled");
                return;
            }
            _ => {}
        }
    }
}
