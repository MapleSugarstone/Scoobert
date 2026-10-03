//! Retries a reply marked bad, rates the next one good, builds a learned direction from a few made-up ratings, and
//! loads the model with it, printing what each step left behind.
//! Usage: SCOOBERT_HOME=<scratch> cargo run --release --example learn_check -- <project folder> <model name>

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use scoobert::agent::conversation::Message;
use scoobert::agent::{Event, Host};
use scoobert::llama::learn::{self, Rating};
use scoobert::store::{Approvals, Settings, Thinking};
use tokio::sync::mpsc::UnboundedReceiver;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let project = PathBuf::from(&args[1]);
    let model = args[2].clone();
    let settings = Arc::new(RwLock::new(Settings { model: model.clone(), approvals: Approvals::Auto, thinking: Thinking::Off, ..Settings::default() }));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let host = Host::new(settings.clone(), tx);
    let snap = host.open(&project, None).unwrap();
    host.prompt(&snap.id, "Describe a cat in one short sentence.".into(), Vec::new(), false).unwrap();
    settle(&mut rx).await;
    host.retry(&snap.id, true).expect("retry starts");
    settle(&mut rx).await;
    host.rate_good(&snap.id).expect("rating saves");
    let replies: Vec<String> = host.open(&project, Some(&snap.file)).unwrap().messages.iter().filter_map(|m| if let Message::Assistant(a) = m { Some(a.text.clone()) } else { None }).collect();
    println!("replies after retry: {replies:?}");
    println!("ratings: {:?}", learn::ratings(&model).iter().map(|r| (r.good, r.reply.chars().take(40).collect::<String>())).collect::<Vec<_>>());
    for (good, reply) in [
        (true, "Arr, a fine whiskered beast prowls the deck!"),
        (true, "Arr, that cat be the captain of the galley!"),
        (false, "A cat is a small domesticated mammal."),
        (false, "Cats are carnivorous mammals of the family Felidae."),
    ] {
        learn::record(&model, Rating { good, asked: "Describe a cat.".into(), reply: reply.into(), time: scoobert::util::now_millis() }).unwrap();
    }
    let dir = settings.read().unwrap().models_dir();
    println!("summary before: {:?}", learn::summary(&dir, &model));
    let start = std::time::Instant::now();
    let result = host.llama.learn(&model, &Arc::new(AtomicBool::new(false)), |p| eprint!("\r{} {}/{}   ", p.status, p.done, p.total)).await;
    println!("\nlearn: {:?} in {:.0} s", result.map_err(|e| format!("{e:#}")), start.elapsed().as_secs_f32());
    println!("summary after: {:?}", learn::summary(&dir, &model));
    settings.write().unwrap().learning.insert(model.clone(), 2.0);
    let m = host.llama.models().into_iter().find(|m| m.name == model).unwrap();
    println!("args: {:?}", m.args);
    host.prompt(&snap.id, "Describe a dog in one short sentence.".into(), Vec::new(), false).unwrap();
    settle(&mut rx).await;
    let last = host.open(&project, Some(&snap.file)).unwrap().messages.into_iter().rev().find_map(|m| if let Message::Assistant(a) = m { Some(a.text) } else { None });
    println!("reply with the learned direction: {last:?}");
    learn::reset(&dir, &model);
    host.shutdown().await;
}

async fn settle(rx: &mut UnboundedReceiver<Event>) {
    loop {
        match tokio::time::timeout(Duration::from_secs(600), rx.recv()).await {
            Ok(Some(Event::Settled { .. })) | Ok(None) | Err(_) => return,
            Ok(Some(Event::Error { message, .. })) => println!("error: {message}"),
            _ => {}
        }
    }
}
