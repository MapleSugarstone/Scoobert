//! Runs one llama-server process for the selected GGUF model and manages its saved prompt caches.

pub mod catalog;
pub mod download;
pub mod gguf;

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, bail};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::paths;
use crate::store::Settings;
use crate::util::{Cancelled, gb, random_hex, sha256_hex};

const PREFERRED_PORT: u16 = 18765;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
const SLOT_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const LARGE_MODEL_BYTES: u64 = 12_000_000_000;
const DEFAULT_CONTEXT: u32 = 32_768;

pub type SharedSettings = Arc<RwLock<Settings>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerStatus {
    Stopped,
    Loading(String),
    Ready(String),
    Error(String),
}

struct Running {
    child: tokio::process::Child,
    model: LocalModel,
}

struct Shared {
    model: Option<(LocalModel, u32)>,
    slot_owner: Option<String>,
    last_use: Instant,
}

pub struct LlamaServer {
    settings: SharedSettings,
    pub api_key: String,
    pub port: u16,
    http: reqwest::Client,
    proc: tokio::sync::Mutex<Option<Running>>,
    shared: Mutex<Shared>,
    busy: AtomicUsize,
    /// A load holds `proc` until the model is ready, so `stop` cancels this first.
    loading: Mutex<CancellationToken>,
    on_status: Box<dyn Fn(ServerStatus) + Send + Sync>,
}

/// Keeps the server from unloading while a request that needs it is running.
pub struct BusyGuard(Arc<LlamaServer>);

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.busy.fetch_sub(1, Ordering::SeqCst);
        self.0.touch();
    }
}

impl LlamaServer {
    /// Stops a server left behind by a crash, picks the port for this run, and starts the idle timer.
    pub fn new(settings: SharedSettings, on_status: impl Fn(ServerStatus) + Send + Sync + 'static) -> Arc<Self> {
        kill_orphan();
        let server = Arc::new(LlamaServer {
            settings,
            // Any web page can send requests to localhost, so the server requires this key.
            api_key: random_hex(24),
            port: free_port(PREFERRED_PORT),
            http: reqwest::Client::new(),
            proc: tokio::sync::Mutex::new(None),
            shared: Mutex::new(Shared { model: None, slot_owner: None, last_use: Instant::now() }),
            busy: AtomicUsize::new(0),
            loading: Mutex::new(CancellationToken::new()),
            on_status: Box::new(on_status),
        });
        let _ = std::fs::create_dir_all(paths::get().slots());
        tokio::spawn(watch(Arc::downgrade(&server)));
        server
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn settings(&self) -> Settings {
        self.settings.read().unwrap().clone()
    }

    pub fn models_dir(&self) -> PathBuf {
        self.settings().models_dir()
    }

    pub fn executable(&self) -> Option<PathBuf> {
        let configured = self.settings().llama_server_path;
        if !configured.trim().is_empty() && Path::new(configured.trim()).is_file() {
            return Some(PathBuf::from(configured.trim()));
        }
        if let Some(dir) = paths::bundled_llama_dir() {
            return Some(dir.join(paths::llama_server_name()));
        }
        #[cfg(windows)]
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            let packages = PathBuf::from(local).join("Microsoft/WinGet/Packages");
            for entry in std::fs::read_dir(packages).into_iter().flatten().flatten() {
                let exe = entry.path().join(paths::llama_server_name());
                if entry.file_name().to_string_lossy().to_lowercase().starts_with("ggml.llamacpp") && exe.is_file() {
                    return Some(exe);
                }
            }
        }
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path).map(|d| d.join(paths::llama_server_name())).find(|p| p.is_file())
    }

    /// GGUF models in the models folder. A projector file beside a model in its own folder enables images.
    pub fn models(&self) -> Vec<LocalModel> {
        let mut out = Vec::new();
        scan(&self.models_dir(), 0, &mut out);
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        out
    }

    /// Large models get a smaller default context so the weights and the cache fit in memory together.
    pub fn context_for(&self, model: &LocalModel) -> u32 {
        let s = self.settings();
        if let Some(&ctx) = s.context_sizes.get(&model.name) {
            return ctx;
        }
        if let Some(listed) = catalog::find(&model.name) {
            return listed.context_size;
        }
        if model.size > LARGE_MODEL_BYTES { 16_384 } else { DEFAULT_CONTEXT }
    }

    pub fn memory_needed(&self, model: &LocalModel, ctx: u32) -> u64 {
        let mmproj = model.mmproj.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
        catalog::memory_formula(model.size + mmproj, ctx as u64 * gguf::kv_bytes_per_token(&model.path))
    }

    pub fn loaded_model(&self) -> Option<String> {
        self.shared.lock().unwrap().model.as_ref().map(|(m, _)| m.name.clone())
    }

    pub fn slot_owner(&self) -> Option<String> {
        self.shared.lock().unwrap().slot_owner.clone()
    }

    pub fn set_slot_owner(&self, owner: Option<String>) {
        self.shared.lock().unwrap().slot_owner = owner;
    }

    pub fn touch(&self) {
        self.shared.lock().unwrap().last_use = Instant::now();
    }

    pub fn busy(self: &Arc<Self>) -> BusyGuard {
        self.busy.fetch_add(1, Ordering::SeqCst);
        self.touch();
        BusyGuard(self.clone())
    }

    /// Starts the server with `model` unless it is already running it.
    pub async fn ensure(self: &Arc<Self>, model: &LocalModel) -> anyhow::Result<()> {
        self.touch();
        let cancel = self.loading.lock().unwrap().clone();
        let mut proc = self.proc.lock().await;
        if cancel.is_cancelled() {
            return Err(Cancelled.into());
        }
        if let Some(running) = proc.as_mut() {
            let alive = matches!(running.child.try_wait(), Ok(None));
            if alive && running.model.path == model.path {
                return Ok(());
            }
        }
        self.stop_locked(&mut proc).await;
        let result = self.start_locked(&mut proc, model, &cancel).await;
        if let Err(err) = &result
            && !crate::util::is_cancelled(err)
        {
            (self.on_status)(ServerStatus::Error(err.to_string()));
        }
        result
    }

    async fn start_locked(&self, proc: &mut Option<Running>, model: &LocalModel, cancel: &CancellationToken) -> anyhow::Result<()> {
        let Some(exe) = self.executable() else {
            bail!("Scoobert could not find llama.cpp. Reinstall Scoobert, or set the llama-server path in Settings.");
        };
        kill_orphan();
        if self.request("/health", None, Duration::from_secs(1), None).await.is_ok() {
            bail!("Another program is using port {}. Restart Scoobert to pick a free port.", self.port);
        }
        let ctx = self.context_for(model);
        let need = self.memory_needed(model, ctx);
        let free = crate::sys::available_memory();
        if free < need {
            bail!(
                "{} needs about {} of free memory, and {} is free. Close other apps, then send your message again.",
                model.name,
                gb(need),
                gb(free)
            );
        }
        let slots = paths::get().slots();
        let mut args: Vec<String> = vec!["-m".into(), path_arg(&model.path), "--alias".into(), model.name.clone()];
        if let Some(mmproj) = &model.mmproj {
            args.extend(["--mmproj".into(), path_arg(mmproj)]);
        }
        #[rustfmt::skip]
        args.extend([
            "-c", &ctx.to_string(),
            "--jinja",
            // A GPU build still reserves GPU memory with no offloaded layers, which crashes 27B models on an integrated GPU.
            "--device", "none",
            "-ngl", "0",
            "-np", "1",
            // The host prompt cache would hold gigabytes of RAM; slot files on disk replace it.
            "--cache-ram", "0",
            // Hybrid models need checkpoints to rewind a few tokens.
            "--ctx-checkpoints", "8",
            "--checkpoint-min-step", "0",
            // The server only notices a cancelled request between batches, so small batches keep Stop quick.
            "-b", "256",
            "-ub", "256",
            "--slot-save-path", &path_arg(&slots),
            "--host", "127.0.0.1",
            "--port", &self.port.to_string(),
            "--no-webui",
            "--cors-origins", "http://127.0.0.1",
            "--no-cors-credentials",
        ].map(String::from));
        // For diagnosing the server, such as -v for its detailed log.
        if let Some(extra) = std::env::var_os("SCOOBERT_SERVER_ARGS") {
            args.extend(extra.to_string_lossy().split_whitespace().map(String::from));
        }

        // The previous log is kept, since a restart would otherwise erase what happened before it.
        let _ = std::fs::rename(paths::get().server_log(), paths::get().server_log().with_extension("previous.log"));
        let log = std::fs::File::create(paths::get().server_log()).context("Could not create the server log")?;
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            // The key goes through the environment so it never appears in the process list.
            .env("LLAMA_API_KEY", &self.api_key)
            .kill_on_drop(true);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);
        #[cfg(not(windows))]
        if let Some(dir) = exe.parent() {
            let mut lib = std::ffi::OsString::from(dir);
            if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
                lib.push(":");
                lib.push(old);
            }
            cmd.env("LD_LIBRARY_PATH", lib);
        }
        let child = cmd.spawn().with_context(|| format!("Could not start {}", exe.display()))?;
        if let Some(pid) = child.id() {
            let _ = std::fs::write(paths::get().pid_file(), pid.to_string());
        }
        *proc = Some(Running { child, model: model.clone() });
        {
            let mut shared = self.shared.lock().unwrap();
            shared.model = Some((model.clone(), ctx));
            shared.slot_owner = None;
        }
        (self.on_status)(ServerStatus::Loading(model.name.clone()));

        let deadline = Instant::now() + HEALTH_TIMEOUT;
        while Instant::now() < deadline {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(400)) => {}
                _ = cancel.cancelled() => {
                    self.stop_locked(proc).await;
                    return Err(Cancelled.into());
                }
            }
            let exited = proc.as_mut().map(|r| !matches!(r.child.try_wait(), Ok(None))).unwrap_or(true);
            if exited {
                *proc = None;
                self.shared.lock().unwrap().model = None;
                bail!("llama-server stopped while loading {}.\n{}", model.name, log_errors());
            }
            if let Ok(h) = self.request("/health", None, Duration::from_secs(2), Some(cancel)).await
                && h["status"] == "ok"
            {
                (self.on_status)(ServerStatus::Ready(model.name.clone()));
                self.touch();
                return Ok(());
            }
        }
        self.stop_locked(proc).await;
        bail!("{} did not finish loading within {} minutes.", model.name, HEALTH_TIMEOUT.as_secs() / 60)
    }

    pub async fn stop(&self) {
        std::mem::replace(&mut *self.loading.lock().unwrap(), CancellationToken::new()).cancel();
        let mut proc = self.proc.lock().await;
        self.stop_locked(&mut proc).await;
    }

    /// Ends the server process without waiting for the lock, for when an orderly stop takes too long.
    pub fn kill_now(&self) {
        kill_from_pid_file();
        (self.on_status)(ServerStatus::Stopped);
    }

    async fn stop_locked(&self, proc: &mut Option<Running>) {
        let Some(mut running) = proc.take() else { return };
        let _ = running.child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(5), running.child.wait()).await;
        let _ = std::fs::remove_file(paths::get().pid_file());
        {
            let mut shared = self.shared.lock().unwrap();
            shared.model = None;
            shared.slot_owner = None;
        }
        (self.on_status)(ServerStatus::Stopped);
    }

    /// Stops the server when it has been idle for the configured time, and notices when it exits on its own.
    async fn check(&self) {
        let Ok(mut proc) = self.proc.try_lock() else { return };
        let Some(running) = proc.as_mut() else { return };
        if !matches!(running.child.try_wait(), Ok(None)) {
            *proc = None;
            {
                let mut shared = self.shared.lock().unwrap();
                shared.model = None;
                shared.slot_owner = None;
            }
            let _ = std::fs::remove_file(paths::get().pid_file());
            (self.on_status)(ServerStatus::Error(format!("llama-server stopped unexpectedly.\n{}", log_errors())));
            return;
        }
        let minutes = self.settings().keep_alive_minutes;
        let idle = self.shared.lock().unwrap().last_use.elapsed();
        if minutes > 0 && self.busy.load(Ordering::SeqCst) == 0 && idle > Duration::from_secs(minutes as u64 * 60) {
            self.stop_locked(&mut proc).await;
        }
    }

    pub async fn request(
        &self,
        path: &str,
        body: Option<&Value>,
        timeout: Duration,
        cancel: Option<&CancellationToken>,
    ) -> anyhow::Result<Value> {
        let url = format!("{}{path}", self.base_url());
        let req = match body {
            Some(b) => self.http.post(url).json(b),
            None => self.http.get(url),
        };
        let send = async {
            let res = req.bearer_auth(&self.api_key).timeout(timeout).send().await?;
            let status = res.status();
            let text = res.text().await?;
            if !status.is_success() {
                bail!("llama-server {path} returned {status}: {}", crate::util::clip(&text, 300));
            }
            Ok(if text.trim().is_empty() { Value::Null } else { serde_json::from_str(&text)? })
        };
        match cancel {
            Some(token) => tokio::select! {
                r = send => r,
                _ = token.cancelled() => Err(Cancelled.into()),
            },
            None => send.await,
        }
    }

    // ---- prompt cache ----
    // A CPU-bound model needs minutes to read the instructions and history. llama-server keeps the processed
    // prompt of one conversation in its slot, so the slot is saved to disk after every run and restored before
    // the next request, and new conversations restore the prompt that every project shares.

    pub fn slot_file(&self, kind: &str, id: &str) -> String {
        let shared = self.shared.lock().unwrap();
        let (name, ctx) = match &shared.model {
            Some((m, ctx)) => (m.name.clone(), *ctx),
            None => ("model".to_string(), DEFAULT_CONTEXT),
        };
        let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
        format!("{safe}-{ctx}-{kind}-{id}.bin")
    }

    pub fn hash(text: &str) -> String {
        sha256_hex(text)[..12].to_string()
    }

    /// The rendered prompt every next request of a conversation starts with, whatever the user writes next. `lead`
    /// opens the next user message, such as a new conversation's environment block.
    pub async fn shared_prefix(&self, payload: &Value, lead: &str) -> anyhow::Result<String> {
        let render = |text: &str| {
            let mut messages = payload["messages"].as_array().cloned().unwrap_or_default();
            messages.push(json!({ "role": "user", "content": format!("{lead}{text}") }));
            json!({ "messages": messages, "tools": payload["tools"], "chat_template_kwargs": payload["chat_template_kwargs"] })
        };
        let a = self.request("/apply-template", Some(&render("A")), Duration::from_secs(60), None).await?;
        let b = self.request("/apply-template", Some(&render("B")), Duration::from_secs(60), None).await?;
        let (a, b) = (a["prompt"].as_str().unwrap_or_default(), b["prompt"].as_str().unwrap_or_default());
        let mut n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
        while !a.is_char_boundary(n) {
            n -= 1;
        }
        let common = &a[..n];
        // A prefix that ends inside the user's message is kept only when it tokenizes as the start of the full
        // prompt, since text on either side of the cut could merge into one token.
        if !lead.is_empty() && common.ends_with(lead) && self.tokenizes_as_start(common, a).await {
            return Ok(common.to_string());
        }
        // Cutting before a special token keeps the prefix tokenized the same way as the full prompt.
        Ok(match common.rfind("<|") {
            Some(cut) if cut > 0 => common[..cut].to_string(),
            _ => common.to_string(),
        })
    }

    async fn tokenizes_as_start(&self, prefix: &str, full: &str) -> bool {
        let tokens = |text: &str| json!({ "content": text, "add_special": true, "parse_special": true });
        let (Ok(p), Ok(f)) = (
            self.request("/tokenize", Some(&tokens(prefix)), Duration::from_secs(60), None).await,
            self.request("/tokenize", Some(&tokens(full)), Duration::from_secs(60), None).await,
        ) else {
            return false;
        };
        match (p["tokens"].as_array(), f["tokens"].as_array()) {
            (Some(p), Some(f)) => !p.is_empty() && f.starts_with(p),
            _ => false,
        }
    }

    pub fn has_slot_file(&self, filename: &str) -> bool {
        paths::get().slots().join(filename).is_file()
    }

    /// Whether a request that needs the server is running.
    pub fn in_use(&self) -> bool {
        self.busy.load(Ordering::SeqCst) > 0
    }

    pub async fn fill(&self, prefix: &str, cancel: &CancellationToken) -> anyhow::Result<()> {
        self.touch();
        let body = json!({ "prompt": prefix, "n_predict": 0, "cache_prompt": true, "id_slot": 0 });
        self.request("/completion", Some(&body), Duration::from_secs(1800), Some(cancel)).await?;
        Ok(())
    }

    pub async fn save(&self, filename: &str) -> anyhow::Result<()> {
        self.request("/slots/0?action=save", Some(&json!({ "filename": filename })), Duration::from_secs(120), None).await?;
        prune();
        Ok(())
    }

    pub async fn restore(&self, filename: &str) -> bool {
        let file = paths::get().slots().join(filename);
        if !file.is_file() {
            return false;
        }
        let body = json!({ "filename": filename });
        let ok = self.request("/slots/0?action=restore", Some(&body), Duration::from_secs(120), None).await.is_ok();
        if ok && let Ok(f) = std::fs::File::options().write(true).open(&file) {
            let _ = f.set_modified(SystemTime::now());
        }
        ok
    }

    /// Tokens the server has generated for the current reply, read from its slot state. The server holds back a
    /// tool call until it is complete, so this is the only progress Scoobert can show while a file is written.
    pub async fn generated_tokens(&self) -> Option<u64> {
        let slots = self.request("/slots", None, Duration::from_secs(2), None).await.ok()?;
        let slot = slots.as_array()?.first()?;
        slot["next_token"][0]["n_decoded"].as_u64().or(slot["next_token"]["n_decoded"].as_u64()).or(slot["n_decoded"].as_u64())
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }
}

async fn watch(server: Weak<LlamaServer>) {
    loop {
        tokio::time::sleep(Duration::from_secs(15)).await;
        let Some(server) = server.upgrade() else { return };
        server.check().await;
    }
}

fn scan(dir: &Path, depth: u32, out: &mut Vec<LocalModel>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let entries: Vec<_> = read.flatten().collect();
    let name_of = |e: &std::fs::DirEntry| e.file_name().to_string_lossy().into_owned();
    let ggufs: Vec<_> = entries
        .iter()
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false) && name_of(e).to_lowercase().ends_with(".gguf"))
        .collect();
    let mmproj = if depth > 0 { ggufs.iter().find(|e| name_of(e).to_lowercase().starts_with("mmproj")) } else { None };
    let shard_re = regex::Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap();
    for e in &ggufs {
        let file_name = name_of(e);
        if file_name.to_lowercase().starts_with("mmproj") {
            continue;
        }
        let shard = shard_re.captures(&file_name);
        if shard.as_ref().is_some_and(|c| &c[1] != "00001") {
            continue;
        }
        let name = model_stem(&file_name);
        // A split model's first file holds only metadata, so its size is the sum of every part.
        let size: u64 = if shard.is_some() {
            ggufs
                .iter()
                .filter(|g| {
                    let n = name_of(g);
                    n.starts_with(&format!("{name}-")) && shard_re.is_match(&n)
                })
                .filter_map(|g| g.metadata().ok())
                .map(|m| m.len())
                .sum()
        } else {
            e.metadata().map(|m| m.len()).unwrap_or(0)
        };
        out.push(LocalModel { name, path: e.path(), mmproj: mmproj.map(|m| m.path()), size });
    }
    if depth < 2 {
        for e in &entries {
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                scan(&e.path(), depth + 1, out);
            }
        }
    }
}

/// A model's name: its file name without the shard suffix and extension.
pub fn model_stem(file_name: &str) -> String {
    let re = regex::Regex::new(r"(?i)(-\d{5}-of-\d{5})?\.gguf$").unwrap();
    re.replace(file_name, "").into_owned()
}

fn path_arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn free_port(preferred: u16) -> u16 {
    TcpListener::bind(("127.0.0.1", preferred))
        .or_else(|_| TcpListener::bind(("127.0.0.1", 0)))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(preferred)
}

/// A server left behind by a crash or a forced close still holds the port and several GB of memory.
fn kill_orphan() {
    if kill_from_pid_file() {
        std::thread::sleep(Duration::from_millis(1500));
    }
}

/// Kills the server named in the pid file, if that process is still llama-server.
fn kill_from_pid_file() -> bool {
    let file = paths::get().pid_file();
    let Ok(text) = std::fs::read_to_string(&file) else { return false };
    let _ = std::fs::remove_file(&file);
    text.trim().parse::<u32>().is_ok_and(|pid| crate::sys::kill_if_named(pid, paths::llama_server_name()))
}

fn log_errors() -> String {
    let text = std::fs::read_to_string(paths::get().server_log()).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| l.contains(" E ") || l.contains("error")).collect();
    lines[lines.len().saturating_sub(3)..].join("\n")
}

/// Deletes the oldest slot files once the folder passes its size cap.
fn prune() {
    let dir = paths::get().slots();
    let mut files: Vec<(PathBuf, u64, SystemTime)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((e.path(), m.len(), m.modified().ok()?))
        })
        .collect();
    files.sort_by(|a, b| b.2.cmp(&a.2));
    let mut total = 0;
    for (path, size, _) in files {
        total += size;
        if total > SLOT_CACHE_BYTES {
            let _ = std::fs::remove_file(path);
        }
    }
}
