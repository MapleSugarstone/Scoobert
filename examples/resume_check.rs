//! Stops a reply while the model thinks, presses Continue, then does the same across a restart, printing how much of
//! the prompt each step reads and what the conversation holds after.
//! Usage: SCOOBERT_HOME=<scratch> cargo run --release --example resume_check -- <project folder> <model name> <seconds>

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use scoobert::agent::conversation::Message;
use scoobert::agent::stream::Delta;
use scoobert::agent::{Event, Host};
use scoobert::store::{Approvals, Settings};
use tokio::sync::mpsc::UnboundedReceiver;

const QUESTION: &str = "Without using any tools, think carefully about how you would design a small command-line to-do app in Rust, then describe the design in five short bullet points.";

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let project = PathBuf::from(&args[1]);
    let model = args[2].clone();
    let think_for: u64 = args[3].parse().unwrap_or(20);
    let (host, mut rx) = start(&model);
    let snap = host.open(&project, None).expect("the conversation opens");
    let id = snap.id.clone();
    println!("== stop after {think_for} s of thinking");
    host.prompt(&id, QUESTION.into(), Vec::new(), false).expect("the prompt starts");
    run(&host, &id, &mut rx, Some(think_for)).await;
    settle(&mut rx).await;
    show(&host, &project, &snap.file);
    println!("== Continue in the same session");
    host.prompt(&id, "Continue".into(), Vec::new(), true).expect("Continue starts");
    run(&host, &id, &mut rx, None).await;
    settle(&mut rx).await;
    show(&host, &project, &snap.file);

    println!("== stop a second question, then restart");
    host.prompt(&id, "Now think about how the app should store its data, then answer in three bullet points.".into(), Vec::new(), false).expect("the prompt starts");
    run(&host, &id, &mut rx, Some(think_for)).await;
    settle(&mut rx).await;
    host.shutdown().await;
    drop(host);
    let (host, mut rx) = start(&model);
    let snap = host.open(&project, Some(&snap.file)).expect("the conversation opens again");
    println!("== Continue after the restart");
    host.prompt(&snap.id, "Continue".into(), Vec::new(), true).expect("Continue starts");
    run(&host, &snap.id, &mut rx, None).await;
    settle(&mut rx).await;
    show(&host, &project, &snap.file);
    host.shutdown().await;
}

fn start(model: &str) -> (Arc<Host>, UnboundedReceiver<Event>) {
    let settings = Settings { model: model.into(), approvals: Approvals::Auto, ..Settings::default() };
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    (Host::new(Arc::new(RwLock::new(settings)), tx), rx)
}

/// Prints the run's reads and replacements until it settles, stopping it after `stop_after` seconds of thinking.
async fn run(host: &Arc<Host>, id: &str, rx: &mut UnboundedReceiver<Event>, stop_after: Option<u64>) {
    let start = Instant::now();
    let mut thinking_since: Option<Instant> = None;
    let mut chars = 0;
    loop {
        if let (Some(limit), Some(since)) = (stop_after, thinking_since)
            && since.elapsed() > Duration::from_secs(limit)
        {
            println!("[{:.0}s] stopping after {chars} characters", start.elapsed().as_secs_f32());
            host.abort(id);
            thinking_since = None;
        }
        let event = match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
            Ok(Some(e)) => e,
            Ok(None) => return,
            Err(_) => continue,
        };
        let t = start.elapsed().as_secs_f32();
        match event {
            Event::Delta { delta: Delta::Thinking(s), .. } | Event::Delta { delta: Delta::Text(s), .. } => {
                if chars == 0 {
                    println!("[{t:.0}s] first delta, {} characters: {:?}", s.len(), s.chars().take(80).collect::<String>());
                }
                chars += s.len();
                if stop_after.is_some() && chars == s.len() {
                    thinking_since = Some(Instant::now());
                }
            }
            Event::Delta { delta: Delta::Progress { done, total, .. }, .. } => println!("[{t:.0}s] reading {done} of {total}"),
            Event::Replacing { .. } => println!("[{t:.0}s] replacing the stopped reply"),
            Event::Message { message, .. } => println!("[{t:.0}s] message {}", summary(&message)),
            Event::Activity { text: Some(a), .. } => println!("[{t:.0}s] {a}"),
            Event::Error { message, .. } => println!("[{t:.0}s] error {message}"),
            Event::Settled { interrupted, .. } => {
                println!("[{t:.0}s] settled, interrupted {interrupted}, {chars} characters streamed");
                return;
            }
            _ => {}
        }
    }
}

/// Lets the cache save after a run finish.
async fn settle(rx: &mut UnboundedReceiver<Event>) {
    let until = Instant::now() + Duration::from_secs(20);
    while let Ok(Some(event)) = tokio::time::timeout(until.saturating_duration_since(Instant::now()), rx.recv()).await {
        if let Event::Delta { delta: Delta::Progress { done, total, .. }, .. } = event {
            println!("[after] saving, read {done} of {total}");
        }
    }
}

fn show(host: &Arc<Host>, project: &Path, file: &Path) {
    let snap = host.open(project, Some(file)).expect("the conversation opens");
    println!("-- {} messages:", snap.messages.len());
    for m in &snap.messages {
        println!("   {}", summary(m));
    }
}

fn summary(m: &Message) -> String {
    match m {
        Message::User(u) => format!("user {:?} context {:?}", u.text.chars().take(40).collect::<String>(), u.context.chars().take(160).collect::<String>()),
        Message::Assistant(a) => format!(
            "assistant {:?}, thinking {} chars ending {:?}, text {} chars, {} calls",
            a.stop,
            a.thinking.len(),
            a.thinking.chars().rev().take(40).collect::<String>().chars().rev().collect::<String>(),
            a.text.len(),
            a.tool_calls.len()
        ),
        Message::Tool(t) => format!("tool {} {} chars", t.name, t.output.len()),
    }
}
