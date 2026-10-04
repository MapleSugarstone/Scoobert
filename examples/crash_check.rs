//! Kills the model server twice while it writes a reply, then prints how the run recovers and what the conversation
//! holds after.
//! Usage: SCOOBERT_HOME=<scratch> cargo run --release --example crash_check -- <project folder> <models folder> <model name>

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use scoobert::agent::conversation::Message;
use scoobert::agent::stream::Delta;
use scoobert::agent::{Event, Host};
use scoobert::store::{Approvals, Settings};

const QUESTION: &str = "Without using any tools, explain in about 400 words how a hash map handles collisions, with a short example.";
/// Characters streamed in each reply before the server is killed.
const KILL_AFTER: usize = 400;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let project = PathBuf::from(&args[1]);
    let settings = Settings { model: args[3].clone(), models_dir: args[2].clone(), approvals: Approvals::Auto, ..Settings::default() };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let host = Host::new(Arc::new(RwLock::new(settings)), tx);
    let snap = host.open(&project, None).expect("the conversation opens");
    host.prompt(&snap.id, QUESTION.into(), Vec::new(), false).expect("the prompt starts");
    let start = Instant::now();
    let (mut chars, mut kills) = (0, 0);
    loop {
        let event = match tokio::time::timeout(Duration::from_secs(600), rx.recv()).await {
            Ok(Some(e)) => e,
            _ => {
                println!("no event for 10 minutes");
                break;
            }
        };
        let t = start.elapsed().as_secs_f32();
        match event {
            Event::Delta { delta: Delta::Thinking(s) | Delta::Text(s), .. } => {
                chars += s.len();
                if kills < 2 && chars >= KILL_AFTER {
                    kills += 1;
                    chars = 0;
                    println!("[{t:.0}s] killing the server ({kills})");
                    kill_server();
                }
            }
            Event::Server(status) => println!("[{t:.0}s] server {status:?}"),
            Event::Replacing { .. } => println!("[{t:.0}s] replacing the stopped reply"),
            Event::Message { message, .. } => println!("[{t:.0}s] message {}", summary(&message)),
            Event::Activity { text: Some(a), .. } => println!("[{t:.0}s] {a}"),
            Event::Error { message, .. } => println!("[{t:.0}s] error {message}"),
            Event::Settled { interrupted, .. } => {
                println!("[{t:.0}s] settled, interrupted {interrupted}");
                break;
            }
            _ => {}
        }
    }
    let snap = host.open(&project, Some(&snap.file)).expect("the conversation opens again");
    println!("-- {} messages:", snap.messages.len());
    for m in &snap.messages {
        println!("   {}", summary(m));
    }
    if let Some(Message::Assistant(a)) = snap.messages.last() {
        println!("-- reply text:\n{}", a.text);
    }
    host.shutdown().await;
}

/// Kills the llama-server this process started, as a crash would.
fn kill_server() {
    let script = format!(
        "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe' AND ParentProcessId={}\" | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}",
        std::process::id()
    );
    let _ = std::process::Command::new("powershell").args(["-NoProfile", "-Command", &script]).status();
}

fn summary(m: &Message) -> String {
    match m {
        Message::User(u) => format!("user {:?}", u.text.chars().take(40).collect::<String>()),
        Message::Assistant(a) => format!("assistant {:?}, thinking {} chars, text {} chars, {} calls, error {:?}", a.stop, a.thinking.len(), a.text.len(), a.tool_calls.len(), a.error),
        Message::Tool(t) => format!("tool {} {} chars", t.name, t.output.len()),
    }
}
