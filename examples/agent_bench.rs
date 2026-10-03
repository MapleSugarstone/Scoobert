//! Gives a model the benchmark tasks through Scoobert's own agent, with its system prompt and tools, in one project
//! folder. After each turn the hidden tests run on the file the model wrote, and a failure goes back to the model as
//! a message, up to the given number of tries. Each finished task is appended to the results file, and a rerun skips
//! tasks already there for the same label.
//! Usage: SCOOBERT_HOME=<scratch> cargo run --release --example agent_bench -- <models folder> <model name> <label>
//!        <normal|hard> <tries> <minutes per try> <project folder> <results.jsonl> <off|low|medium|high>

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use scoobert::agent::conversation::Message;
use scoobert::agent::{Event, Host};
use scoobert::llama::bench::{self, MARK, Set};
use scoobert::store::{Approvals, PlanFirst, Settings, Thinking};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (models_dir, model, label) = (a[1].clone(), a[2].clone(), a[3].clone());
    let set = if a[4] == "hard" { Set::Hard } else { Set::Normal };
    let tries: u32 = a[5].parse().unwrap_or(5);
    let limit = Duration::from_secs(a[6].parse::<u64>().unwrap_or(30) * 60);
    let project = PathBuf::from(&a[7]);
    let results = PathBuf::from(&a[8]);
    let thinking = match a.get(9).map(String::as_str) {
        Some("low") => Thinking::Low,
        Some("medium") => Thinking::Medium,
        Some("high") => Thinking::High,
        _ => Thinking::Off,
    };
    std::fs::create_dir_all(&project).unwrap();
    let checks = project.with_file_name(format!("{}-checks", project.file_name().unwrap().to_string_lossy()));
    std::fs::create_dir_all(&checks).unwrap();
    let done: Vec<String> = std::fs::read_to_string(&results)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["label"] == label.as_str())
        .filter_map(|v| v["task"].as_str().map(String::from))
        .collect();

    let mut settings = Settings {
        model: model.clone(),
        thinking,
        approvals: Approvals::Auto,
        models_dir,
        remember_step: false,
        activity_log: false,
        web_access: false,
        plan_first: PlanFirst::Off,
        keep_alive_minutes: 0,
        language: "en".into(),
        ..Settings::default()
    };
    settings.context_sizes.insert(model.clone(), 32_768);
    let shared = Arc::new(RwLock::new(settings));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let host = Host::new(shared.clone(), tx);
    let Some(file) = host.llama.models().into_iter().find(|m| m.name == model).map(|m| m.path) else {
        panic!("{model} is not in the models folder");
    };
    if scoobert::llama::gguf::kind(&file).1 {
        shared.write().unwrap().speculation.insert(model.clone(), "mtp".into());
        println!("{label}: predicting ahead with the model's own layers");
    }
    let python = bench::python().expect("Python 3 runs the hidden tests");
    let mut passed = 0;
    for task in bench::tasks(set) {
        if done.contains(&task.name) {
            println!("{label} {}: already done", task.name);
            continue;
        }
        let target = format!("{}.py", task.name);
        let snap = host.open(&project, None).expect("the conversation opens");
        let mut text = format!(
            "Write your solution in {target} in the project folder. Hidden tests will check that file after you finish, so include only the solution code in it. You may run Python to test it yourself.\n\n{}",
            task.prompt
        );
        let start = Instant::now();
        let (mut ok, mut attempt, mut err) = (false, 0, String::new());
        while attempt < tries && !ok {
            attempt += 1;
            let _ = std::fs::remove_file(project.join(&target));
            host.prompt(&snap.id, text.clone(), Vec::new(), false).expect("the prompt starts");
            let finished = wait(&host, &snap.id, &mut rx, limit).await;
            let code = std::fs::read_to_string(project.join(&target)).unwrap_or_default();
            let check = checks.join(format!("{label}.{}.{attempt}.py", task.name));
            std::fs::write(&check, format!("{code}\n\n{MARK}\n{}", task.test)).unwrap();
            let (pass, output) = bench::check(&python, &check, &CancellationToken::new()).await.unwrap_or((false, "The tests could not run.".into()));
            ok = pass;
            err = output.lines().last().unwrap_or_default().chars().take(150).collect();
            if !finished {
                err = format!("stopped after {} minutes. {err}", limit.as_secs() / 60);
            }
            if !ok {
                let what = if code.trim().is_empty() { format!("{target} is missing or empty.") } else { format!("The output was:\n```\n{output}\n```") };
                text = format!("The hidden tests failed on {target}. {what}\nFix {target} so the tests pass.");
            }
        }
        let snap = host.open(&project, Some(&snap.file)).expect("the conversation opens again");
        let (calls, tokens) = usage(&snap.messages);
        let seconds = start.elapsed().as_secs_f64();
        passed += ok as usize;
        println!(
            "{label} {}: {} on try {attempt} in {:.0} s, {tokens} tokens, {calls} tool calls{}",
            task.name,
            if ok { "PASS" } else { "FAIL" },
            seconds,
            if ok { String::new() } else { format!("  {err}") }
        );
        let line = serde_json::json!({ "label": label, "task": task.name, "ok": ok, "tries": attempt, "seconds": seconds, "tokens": tokens, "calls": calls, "err": err });
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&results).unwrap();
        writeln!(f, "{line}").unwrap();
    }
    println!("{label}: TOTAL {passed} passed this run");
    host.shutdown().await;
}

/// Waits for the conversation's turn to settle, stopping it after `limit`. Returns whether it finished in time.
async fn wait(host: &Arc<Host>, id: &str, rx: &mut UnboundedReceiver<Event>, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    let mut stopped = false;
    loop {
        if !stopped && Instant::now() > deadline {
            host.abort(id);
            stopped = true;
        }
        match tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            Ok(Some(Event::Settled { conv, .. })) if conv == id => break,
            Ok(Some(Event::Error { message, .. })) => println!("  error: {}", message.chars().take(200).collect::<String>()),
            Ok(None) => break,
            _ => {}
        }
    }
    // Saving the cache and the notes after a turn holds the model, so the next prompt waits for them.
    tokio::time::sleep(Duration::from_secs(2)).await;
    !stopped
}

/// Tool calls and tokens written across the conversation.
fn usage(messages: &[Message]) -> (usize, u64) {
    messages.iter().fold((0, 0), |(calls, tokens), m| match m {
        Message::Assistant(a) => (calls + a.tool_calls.len(), tokens + a.usage.map(|u| u.output).unwrap_or(0)),
        _ => (calls, tokens),
    })
}
