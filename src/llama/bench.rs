//! Coding tests that time a local model and run hidden tests on the code it writes.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::bail;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::{LlamaServer, LocalModel};
use crate::paths;
use crate::util::{Cancelled, is_cancelled, now_millis};

const SYSTEM: &str = "You are a careful Python programmer. Read the specification exactly. Reply with one Python code block that holds the complete solution and nothing else: no tests, no examples, no explanation.";
const RETRY: &str = "Your code failed the hidden tests. The output was:\n```\n{output}\n```\nFix the problem and reply with the complete corrected code in one Python code block.";
const TEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Separates the model's code from the hidden tests in each checked file. Some tests read the code above it.
pub const MARK: &str = "# --- tests ---";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Set {
    Normal,
    Hard,
}

#[derive(Deserialize)]
pub struct Task {
    pub name: String,
    pub prompt: String,
    pub test: String,
}

pub fn tasks(set: Set) -> Vec<Task> {
    let text = match set {
        Set::Normal => include_str!("bench/normal.json"),
        Set::Hard => include_str!("bench/hard.json"),
    };
    serde_json::from_str(text).expect("the built-in benchmark tasks parse")
}

pub fn task_count(set: Set) -> usize {
    tasks(set).len()
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub set: Set,
    /// Tries per task. After a failed try the model sees the test output and writes the code again.
    pub tries: u32,
    pub thinking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub name: String,
    /// None when no Python was found to run the tests.
    pub passed: Option<bool>,
    pub tries: u32,
    /// Time spent writing, without reading the prompt.
    pub write_seconds: f64,
    /// Time from sending each try to its reply.
    pub seconds: f64,
    pub tokens: u64,
    #[serde(default)]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub time: i64,
    pub model: String,
    pub set: Set,
    pub tries: u32,
    pub thinking: bool,
    pub gpu: bool,
    /// The server's arguments for predicting ahead and for a compact context.
    pub extras: Vec<String>,
    pub context: u32,
    pub load_seconds: f64,
    pub tasks: Vec<TaskResult>,
}

impl Run {
    pub fn checked(&self) -> bool {
        self.tasks.iter().all(|t| t.passed.is_some())
    }

    pub fn passed(&self) -> usize {
        self.tasks.iter().filter(|t| t.passed == Some(true)).count()
    }

    pub fn first_try(&self) -> usize {
        self.tasks.iter().filter(|t| t.passed == Some(true) && t.tries == 1).count()
    }

    pub fn seconds(&self) -> f64 {
        self.tasks.iter().map(|t| t.seconds).sum()
    }

    pub fn tokens_per_second(&self) -> f64 {
        let write: f64 = self.tasks.iter().map(|t| t.write_seconds).sum();
        if write > 0.0 { self.tasks.iter().map(|t| t.tokens).sum::<u64>() as f64 / write } else { 0.0 }
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    Loading,
    /// The model is writing try `attempt` of task `index`.
    Task { index: usize, count: usize, name: String, attempt: u32 },
    Finished(TaskResult),
}

fn results_file() -> PathBuf {
    paths::get().data.join("benchmarks.jsonl")
}

/// Every finished run, newest first.
pub fn history() -> Vec<Run> {
    let text = std::fs::read_to_string(results_file()).unwrap_or_default();
    let mut runs: Vec<Run> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    runs.reverse();
    runs
}

fn save(run: &Run) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(results_file())?;
    writeln!(file, "{}", serde_json::to_string(run)?)?;
    Ok(())
}

/// The command that runs Python 3 on this computer, if any.
pub fn python() -> Option<Vec<String>> {
    let candidates: &[&[&str]] = if cfg!(windows) { &[&["py", "-3"], &["python"], &["python3"]] } else { &[&["python3"], &["python"]] };
    candidates.iter().find(|c| version(c).is_some_and(|v| v.starts_with("Python 3"))).map(|c| c.iter().map(|s| s.to_string()).collect())
}

fn version(cmd: &[&str]) -> Option<String> {
    let mut c = std::process::Command::new(cmd[0]);
    c.args(&cmd[1..]).arg("--version").stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    let out = c.output().ok().filter(|o| o.status.success())?;
    Some(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)).trim().to_string())
}

/// The longest fenced code block in a reply, or the whole reply when it has none.
fn extract_code(reply: &str) -> String {
    let fence = regex::Regex::new(r"(?s)```(?:python|py)?[ \t]*\r?\n(.*?)```").unwrap();
    fence.captures_iter(reply).map(|c| c[1].to_string()).max_by_key(|c| c.len()).unwrap_or_else(|| reply.to_string())
}

/// Runs a checked file and returns whether its tests passed, with the end of what it printed.
pub async fn check(python: &[String], file: &Path, cancel: &CancellationToken) -> anyhow::Result<(bool, String)> {
    let mut cmd = tokio::process::Command::new(&python[0]);
    // -I leaves out the user's site packages and environment, so every run sees the same Python, and -X utf8 keeps
    // Windows from writing the test output in its legacy code page.
    cmd.args(&python[1..]).args(["-I", "-X", "utf8"]).arg(file).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    if let Some(dir) = file.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let child = cmd.spawn()?;
    let out = tokio::select! {
        r = tokio::time::timeout(TEST_TIMEOUT, child.wait_with_output()) => match r {
            Ok(out) => out?,
            Err(_) => return Ok((false, format!("The tests timed out after {} seconds.", TEST_TIMEOUT.as_secs()))),
        },
        _ = cancel.cancelled() => bail!(Cancelled),
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    let lines: Vec<&str> = stderr.trim().lines().collect();
    Ok((out.status.success(), lines[lines.len().saturating_sub(20)..].join("\n")))
}

/// Loads `model` with its current settings, gives it every task in the set, and saves the run. The caller holds the
/// model so no conversation uses it meanwhile.
pub async fn run(llama: Arc<LlamaServer>, model: LocalModel, opts: Options, cancel: CancellationToken, on_event: impl Fn(Event)) -> anyhow::Result<Run> {
    let python = python();
    let safe: String = model.name.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    let folder = paths::get().data.join("benchmarks").join(format!("{}-{safe}", now_millis()));
    std::fs::create_dir_all(&folder)?;
    on_event(Event::Loading);
    let start = Instant::now();
    tokio::select! {
        r = llama.ensure(&model) => r?,
        _ = cancel.cancelled() => bail!(Cancelled),
    }
    let load_seconds = start.elapsed().as_secs_f64();
    let _busy = llama.busy();
    // The tests replace whatever conversation the server held, so the next message restores its own.
    llama.set_slot_owner(None);
    let (gpu, _, extras) = llama.running_setup().await.unwrap_or_default();
    let tasks = tasks(opts.set);
    let mut run = Run {
        time: now_millis(),
        model: model.name.clone(),
        set: opts.set,
        tries: opts.tries.max(1),
        thinking: opts.thinking,
        gpu,
        extras,
        context: llama.context_for(&model),
        load_seconds,
        tasks: Vec::new(),
    };
    for (index, task) in tasks.iter().enumerate() {
        let mut messages = vec![json!({ "role": "system", "content": SYSTEM }), json!({ "role": "user", "content": task.prompt })];
        let mut result = TaskResult { name: task.name.clone(), passed: None, tries: 0, write_seconds: 0.0, seconds: 0.0, tokens: 0, error: String::new() };
        while result.tries < run.tries {
            result.tries += 1;
            on_event(Event::Task { index, count: tasks.len(), name: task.name.clone(), attempt: result.tries });
            let mut body = json!({
                "messages": messages,
                "max_tokens": if opts.thinking { 10_000 } else { 4_000 },
                "cache_prompt": true,
                "chat_template_kwargs": { "enable_thinking": opts.thinking },
            });
            if opts.thinking {
                body["thinking_budget_tokens"] = 4_000.into();
            }
            let sent = Instant::now();
            let reply = llama.request("/v1/chat/completions", Some(&body), Duration::from_secs(4 * 3600), Some(&cancel)).await?;
            result.seconds += sent.elapsed().as_secs_f64();
            result.tokens += reply["timings"]["predicted_n"].as_u64().unwrap_or(0);
            result.write_seconds += reply["timings"]["predicted_ms"].as_f64().unwrap_or(0.0) / 1000.0;
            let text = reply["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
            let file = folder.join(format!("{}.{}.py", task.name, result.tries));
            std::fs::write(&file, format!("{}\n\n{MARK}\n{}", extract_code(&text), task.test))?;
            let Some(python) = &python else { break };
            let (ok, output) = check(python, &file, &cancel).await?;
            result.passed = Some(ok);
            if ok {
                result.error.clear();
                break;
            }
            result.error = crate::util::clip(output.lines().last().unwrap_or_default(), 200);
            messages.push(json!({ "role": "assistant", "content": text }));
            messages.push(json!({ "role": "user", "content": RETRY.replace("{output}", &output) }));
        }
        on_event(Event::Finished(result.clone()));
        run.tasks.push(result);
    }
    save(&run)?;
    Ok(run)
}

/// Whether `err` means the user stopped the run.
pub fn stopped(err: &anyhow::Error) -> bool {
    is_cancelled(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_tasks_parse() {
        assert_eq!(task_count(Set::Normal), 10);
        assert_eq!(task_count(Set::Hard), 8);
        for set in [Set::Normal, Set::Hard] {
            for t in tasks(set) {
                assert!(!t.prompt.is_empty() && !t.test.is_empty(), "{}", t.name);
            }
        }
    }

    #[test]
    fn code_comes_from_the_longest_block() {
        assert_eq!(extract_code("x\n```python\nshort\n```\n```\nmuch longer\n```"), "much longer\n");
        assert_eq!(extract_code("no fence"), "no fence");
    }
}
