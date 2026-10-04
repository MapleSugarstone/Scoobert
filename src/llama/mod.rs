//! Runs one llama-server process for the selected GGUF model and manages its saved prompt caches.

pub mod bench;
pub mod catalog;
pub mod cuda;
pub mod download;
pub mod gguf;
pub mod gguf_file;
pub mod lab;
pub mod learn;

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
/// Graphics card memory llama.cpp leaves free when it fits layers on the card.
const VRAM_MARGIN: u64 = 1_000_000_000;
/// System memory a model needs even when the card holds all its layers, for the token table and working buffers.
const MIN_SYSTEM_MEMORY: u64 = 1_500_000_000;
/// The server process ended before the model finished loading.
#[derive(Debug)]
struct ExitedWhileLoading(String);

impl std::fmt::Display for ExitedWhileLoading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ExitedWhileLoading {}

/// The model needs more free memory than there is. `from_disk` says whether loading from disk was already on, since
/// the window offers to turn it on when it was off.
#[derive(Debug)]
pub struct NotEnoughMemory {
    pub message: String,
    pub from_disk: bool,
}

impl std::fmt::Display for NotEnoughMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for NotEnoughMemory {}

/// Free memory kept beyond a disk-loaded model's cache and buffers, for the parts of the weights in use. A dense
/// model reads all of its weights for every token, so a larger margin does not keep it from reading the disk, and
/// 3 GB refused the 9B on an 8 GB Mac with 3.4 GB free.
const DISK_MARGIN: u64 = 1_000_000_000;
/// How often a long read saves its progress.
const SAVE_EVERY: Duration = Duration::from_secs(60);
const FIRST_STEP: usize = 256;
const MAX_STEP: usize = 8192;
/// Marks a slot file the server is still writing.
const PARTIAL: &str = "partial-";

pub type SharedSettings = Arc<RwLock<Settings>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub size: u64,
    /// Extra llama-server arguments, such as a variant's steering vectors, with paths relative to the models folder.
    pub args: Vec<String>,
    /// The model it descends from, which sets its default context size. A downloaded model is its own family.
    pub family: String,
    /// The folder of a variant made in the model lab.
    pub variant: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerStatus {
    Stopped,
    Loading(String),
    Ready(String),
    Error(String),
    /// The graphics card could not load the named model, which then loads on the processor.
    GpuFailed(String),
    /// NVIDIA support could not load the named model, which then loads through the card's default support.
    CudaFailed(String),
    /// The named model could not load while predicting ahead, so it runs without that until Scoobert restarts.
    SpecFailed(String),
}

struct Running {
    child: tokio::process::Child,
    model: LocalModel,
    /// False until the model finishes loading. A caller that stopped waiting partway leaves it false.
    ready: bool,
    /// Whether it runs on the graphics card, and through NVIDIA support, so a change to either setting restarts it.
    gpu: bool,
    cuda: bool,
    /// Its arguments for predicting ahead and for the context's precision, which also restart it when they change.
    extras: Vec<String>,
}

struct Shared {
    model: Option<(LocalModel, u32)>,
    slot_owner: Option<String>,
    last_use: Instant,
    last_save: Instant,
}

pub struct LlamaServer {
    settings: SharedSettings,
    pub api_key: String,
    pub port: u16,
    http: reqwest::Client,
    proc: tokio::sync::Mutex<Option<Running>>,
    shared: Mutex<Shared>,
    busy: AtomicUsize,
    /// Models the graphics card failed to load in this run, until the app saves them to the settings.
    gpu_refused: Mutex<std::collections::HashSet<String>>,
    /// A load holds `proc` until the model is ready, so `stop` cancels this first.
    loading: Mutex<CancellationToken>,
    /// The model being loaded and when its load started, for the activity line of a task that waits on it.
    loading_model: Mutex<Option<(String, Instant)>>,
    /// CUDA failed to load a model in this run, so the card's default support loads them from then on.
    cuda_refused: AtomicBool,
    /// Models that could not load while predicting ahead in this run.
    spec_refused: Mutex<std::collections::HashSet<String>>,
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
            // The server can close a connection just after a streamed reply ends, and a request sent on it at that
            // moment fails, so every request opens its own.
            http: reqwest::Client::builder().pool_max_idle_per_host(0).build().unwrap_or_default(),
            proc: tokio::sync::Mutex::new(None),
            shared: Mutex::new(Shared { model: None, slot_owner: None, last_use: Instant::now(), last_save: Instant::now() }),
            busy: AtomicUsize::new(0),
            gpu_refused: Mutex::new(std::collections::HashSet::new()),
            loading: Mutex::new(CancellationToken::new()),
            loading_model: Mutex::new(None),
            cuda_refused: AtomicBool::new(false),
            spec_refused: Mutex::new(std::collections::HashSet::new()),
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

    /// The folder with llama-server and the llama.cpp tools Scoobert bundles beside it.
    pub fn tools_dir(&self) -> Option<PathBuf> {
        self.executable()?.parent().map(Path::to_path_buf)
    }

    /// GGUF models in the models folder, and the model files the user added from other folders. A projector file
    /// beside a model in its own folder enables images.
    pub fn models(&self) -> Vec<LocalModel> {
        let mut out = Vec::new();
        let dir = self.models_dir();
        scan(&dir, &dir, 0, &mut out);
        for file in self.settings().model_files {
            let Some(mut model) = added_model(Path::new(&file)) else { continue };
            if out.iter().any(|m| m.path == model.path) {
                continue;
            }
            // Names identify models in settings and conversations, so a second file with the same name gets its folder.
            if out.iter().any(|m| m.name == model.name) {
                let folder = model.path.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                model.name = format!("{} ({folder})", model.name);
            }
            out.push(model);
        }
        // A model learning from ratings runs with the direction it learned. Variants keep their own steering, whose
        // layer range the learned direction would have to share.
        let learning = self.settings().learning;
        for m in out.iter_mut().filter(|m| m.variant.is_none()) {
            if let Some(&strength) = learning.get(&m.name) {
                m.args.extend(learn::args(&dir, m, strength));
            }
        }
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        out
    }

    /// Builds the direction a model learned from its ratings. The steering tool loads the model on its own, so the
    /// server stops first, and the prompts saved without the direction go.
    pub async fn learn(&self, name: &str, cancel: &Arc<AtomicBool>, progress: impl Fn(lab::Progress) + Send + Sync + 'static) -> anyhow::Result<()> {
        let model = self.models().into_iter().find(|m| m.name == name && m.variant.is_none()).with_context(|| format!("There is no model named {name}."))?;
        let tools = self.tools_dir().context("Scoobert could not find llama.cpp's tools. Reinstall Scoobert.")?;
        if self.loaded_model().as_deref() == Some(name) {
            if self.in_use() {
                bail!(crate::i18n::tr("Scoobert is using this model. Try again after the task finishes."));
            }
            self.stop().await;
        }
        learn::build(&tools, &model, &self.models_dir(), cancel, progress).await?;
        forget_slots(name);
        Ok(())
    }

    /// Large models get a smaller default context so the weights and the cache fit in memory together.
    pub fn context_for(&self, model: &LocalModel) -> u32 {
        let s = self.settings();
        if let Some(&ctx) = s.context_sizes.get(&model.name) {
            return ctx;
        }
        if let Some(listed) = catalog::find(&model.family) {
            return listed.context_size;
        }
        let default = if model.size > LARGE_MODEL_BYTES { 16_384 } else { DEFAULT_CONTEXT };
        // A model trained on a shorter context than the default gets its own.
        gguf::trained_context(&model.path).map_or(default, |trained| default.min(trained.min(u32::MAX as u64) as u32))
    }

    pub fn memory_needed(&self, model: &LocalModel, ctx: u32) -> u64 {
        let s = self.settings();
        let mmproj = model.mmproj.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
        let mut weights = model.size + mmproj;
        let mut kv = ctx as u64 * gguf::kv_bytes_per_token(&model.path);
        // A draft model loads beside the model with a context of its own.
        if let Some(name) = s.speculation.get(&model.name).and_then(|v| v.strip_prefix("draft:"))
            && let Some(draft) = self.models().into_iter().find(|m| m.name == name)
        {
            weights += draft.size;
            kv += ctx as u64 * gguf::kv_bytes_per_token(&draft.path);
        }
        // 8-bit values take 8.5 bits with their scales, against 16.
        if s.compact_context {
            kv = kv * 17 / 32;
        }
        catalog::memory_formula(weights, kv)
    }

    pub fn loaded_model(&self) -> Option<String> {
        self.shared.lock().unwrap().model.as_ref().map(|(m, _)| m.name.clone())
    }

    /// How the loaded model runs: on the graphics card, through NVIDIA support, and with which extra arguments.
    pub async fn running_setup(&self) -> Option<(bool, bool, Vec<String>)> {
        self.proc.lock().await.as_ref().map(|r| (r.gpu, r.cuda, r.extras.clone()))
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

    /// Whether `model` runs on the graphics card, and the CUDA server when NVIDIA support takes over the card. NVIDIA
    /// support replaces the bundled server unless the user named a server of their own.
    fn backend(&self, model: &str) -> (bool, Option<PathBuf>) {
        let s = self.settings();
        let gpu = s.use_gpu && !s.gpu_failed.iter().any(|m| m == model) && !self.gpu_refused.lock().unwrap().contains(model);
        let cuda = if gpu && s.llama_server_path.trim().is_empty() && !self.cuda_refused.load(Ordering::SeqCst) { cuda::server() } else { None };
        (gpu, cuda)
    }

    /// Arguments for predicting ahead, unless that failed for this model in this run, and for a compact context.
    fn extra_args(&self, model: &LocalModel, gpu: bool) -> Vec<String> {
        let s = self.settings();
        let mut args = Vec::new();
        if s.compact_context {
            args.extend(["-ctk", "q8_0", "-ctv", "q8_0"].map(String::from));
        }
        if self.spec_refused.lock().unwrap().contains(&model.name) {
            return args;
        }
        // A model left out uses its own prediction layers when its file has them, and "off" turns them off.
        let choice = s.speculation.get(&model.name).cloned().or_else(|| gguf::kind(&model.path).1.then(|| "mtp".to_string()));
        match choice.as_deref() {
            // 3 tokens a step: on prose, where the layers guess about 40% right, 8 cost the 27B 30% of its speed,
            // while on code 3 and 8 were even.
            Some("mtp") => args.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"].map(String::from)),
            // Text the model writes often repeats text already in the conversation, such as code it edits.
            Some("ngram") => args.extend(["--spec-type", "ngram-simple"].map(String::from)),
            Some(other) => {
                if let Some(name) = other.strip_prefix("draft:")
                    && let Some(draft) = self.models().into_iter().find(|m| m.name == name && m.variant.is_none())
                {
                    args.extend(["--spec-type".into(), "draft-simple".into(), "-md".into(), path_arg(&draft.path)]);
                    if !gpu {
                        args.extend(["-devd", "none", "-ngld", "0"].map(String::from));
                    }
                }
            }
            None => {}
        }
        args
    }

    /// Tries predicting ahead for `model` again, after the user changed how it does.
    pub fn forget_spec_failure(&self, model: &str) {
        self.spec_refused.lock().unwrap().remove(model);
    }

    /// Starts the server with `model` unless it is already running it.
    pub async fn ensure(self: &Arc<Self>, model: &LocalModel) -> anyhow::Result<()> {
        self.touch();
        let cancel = self.loading.lock().unwrap().clone();
        let mut proc = self.proc.lock().await;
        if cancel.is_cancelled() {
            return Err(Cancelled.into());
        }
        let (gpu, cuda) = self.backend(&model.name);
        let extras = self.extra_args(model, gpu);
        if let Some(running) = proc.as_mut() {
            let alive = matches!(running.child.try_wait(), Ok(None));
            // A variant can run on another model's file with its own steering, so the arguments count too.
            let same = running.model.path == model.path
                && running.model.args == model.args
                && running.gpu == gpu
                && running.cuda == cuda.is_some()
                && running.extras == extras;
            if alive && same {
                if running.ready {
                    return Ok(());
                }
                // An earlier caller stopped waiting while this model loaded, so this one waits for it instead.
                let result = self.wait_ready(&mut proc, model, &cancel).await;
                if let Err(err) = &result
                    && !crate::util::is_cancelled(err)
                {
                    (self.on_status)(ServerStatus::Error(err.to_string()));
                }
                return result;
            }
        }
        self.stop_locked(&mut proc).await;
        let mut result = self.start_locked(&mut proc, model, gpu, cuda.clone(), &cancel).await;
        // A way of predicting ahead that the model cannot load with, such as a draft model of another family, is
        // dropped for the rest of this run before the card or CUDA take the blame.
        if extras.iter().any(|a| a == "--spec-type")
            && let Err(err) = &result
            && err.is::<ExitedWhileLoading>()
        {
            eprintln!("[llama] {} could not load while predicting ahead: {err:#}", model.name);
            self.spec_refused.lock().unwrap().insert(model.name.clone());
            (self.on_status)(ServerStatus::SpecFailed(model.name.clone()));
            result = self.start_locked(&mut proc, model, gpu, cuda.clone(), &cancel).await;
        }
        // CUDA that cannot load the model leaves it to the card's default support, for the rest of this run.
        if cuda.is_some()
            && let Err(err) = &result
            && err.is::<ExitedWhileLoading>()
        {
            eprintln!("[llama] CUDA could not load {}: {err:#}", model.name);
            self.cuda_refused.store(true, Ordering::SeqCst);
            (self.on_status)(ServerStatus::CudaFailed(model.name.clone()));
            result = self.start_locked(&mut proc, model, gpu, None, &cancel).await;
        }
        // A graphics card that cannot load the model leaves it to the processor, and the app remembers the model. Only
        // a server that dies while loading counts, since a check that fails before the start is not the card's doing.
        if gpu
            && let Err(err) = &result
            && err.is::<ExitedWhileLoading>()
        {
            eprintln!("[llama] the graphics card could not load {}: {err:#}", model.name);
            self.gpu_refused.lock().unwrap().insert(model.name.clone());
            (self.on_status)(ServerStatus::GpuFailed(model.name.clone()));
            result = self.start_locked(&mut proc, model, false, None, &cancel).await;
        }
        if let Err(err) = &result
            && !crate::util::is_cancelled(err)
        {
            (self.on_status)(ServerStatus::Error(err.to_string()));
        }
        result
    }

    /// Starts a server for `model`: `server` when given, such as the CUDA build, otherwise the configured or bundled one.
    async fn start_locked(&self, proc: &mut Option<Running>, model: &LocalModel, gpu: bool, server: Option<PathBuf>, cancel: &CancellationToken) -> anyhow::Result<()> {
        let cuda_server = server.is_some();
        let Some(exe) = server.or_else(|| self.executable()) else {
            bail!("Scoobert could not find llama.cpp. Reinstall Scoobert, or set the llama-server path in Settings.");
        };
        kill_orphan();
        if self.request("/health", None, Duration::from_secs(1), None).await.is_ok() {
            bail!("Another program is using port {}. Restart Scoobert to pick a free port.", self.port);
        }
        let ctx = self.context_for(model);
        let mut need = self.memory_needed(model, ctx);
        // On the graphics card, llama.cpp puts the layers that fit in the card's own memory, less a margin it keeps
        // free, so system memory holds only the rest.
        if gpu && let Some(vram) = crate::sys::free_vram() {
            need = need.saturating_sub(vram.saturating_sub(VRAM_MARGIN)).max(MIN_SYSTEM_MEMORY);
        }
        let free = crate::sys::available_memory();
        if free < need {
            // The weights are memory-mapped, so with the setting on, the weights that do not fit stay on disk and the
            // system reads them in as virtual memory. Everything else, such as the conversation's cache, still has to
            // fit in memory.
            let from_disk = self.settings().models_from_disk;
            if !(from_disk && free >= need.saturating_sub(model.size) + DISK_MARGIN) {
                let args: &[(&str, &dyn std::fmt::Display)] = &[("model", &model.name), ("need", &gb(need)), ("free", &gb(free))];
                let message = if from_disk {
                    crate::i18n::trf("{model} needs about {need} of free memory, and {free} is free. Close other apps, then send your message again.", args)
                } else {
                    crate::i18n::trf("{model} needs about {need} of free memory, and {free} is free. Close other apps, or turn on loading models from disk in Settings to run it slowly.", args)
                };
                return Err(NotEnoughMemory { message, from_disk }.into());
            }
        }
        let slots = paths::get().slots();
        let mut args: Vec<String> = vec!["-m".into(), path_arg(&model.path), "--alias".into(), model.name.clone()];
        if let Some(mmproj) = &model.mmproj {
            args.extend(["--mmproj".into(), path_arg(mmproj)]);
        }
        if let Some(template) = reasoning_keeping_template(model) {
            args.extend(["--chat-template-file".into(), path_arg(&template)]);
        }
        // The server only notices a cancelled request between batches, so a processor reads in small batches to keep
        // Stop quick. With the graphics card, llama.cpp copies the weights it keeps in RAM to the card for every
        // batch it reads, so small batches spend their time copying: a 35B-A3B mostly in RAM beside an 8 GB card
        // read 25 to 67 tokens a second in batches of 256.
        let batch = if gpu { "1024" } else { "256" };
        #[rustfmt::skip]
        args.extend([
            "-c", &ctx.to_string(),
            "--jinja",
            "-np", "1",
            // The host prompt cache would hold gigabytes of RAM; slot files on disk replace it.
            "--cache-ram", "0",
            // Hybrid models need checkpoints to rewind a few tokens.
            "--ctx-checkpoints", "8",
            "--checkpoint-min-step", "0",
            "-b", batch,
            "-ub", batch,
            "--slot-save-path", &path_arg(&slots),
            "--host", "127.0.0.1",
            "--port", &self.port.to_string(),
            "--no-webui",
            "--cors-origins", "http://127.0.0.1",
            "--no-cors-credentials",
        ].map(String::from));
        // On the graphics card, llama.cpp fits as many layers as its memory holds and keeps the rest on the processor.
        // Otherwise the card is left out entirely, because even with no layers on it a GPU backend reserves memory,
        // which crashed 27B models on an integrated GPU.
        if !gpu {
            args.extend(["--device", "none", "-ngl", "0"].map(String::from));
        }
        args.extend(model.args.iter().cloned());
        let extras = self.extra_args(model, gpu);
        args.extend(extras.iter().cloned());
        // For diagnosing the server, such as -v for its detailed log.
        if let Some(extra) = std::env::var_os("SCOOBERT_SERVER_ARGS") {
            args.extend(extra.to_string_lossy().split_whitespace().map(String::from));
        }

        // The previous log is kept, since a restart would otherwise erase what happened before it.
        let _ = std::fs::rename(paths::get().server_log(), paths::get().server_log().with_extension("previous.log"));
        let log = std::fs::File::create(paths::get().server_log()).context("Could not create the server log")?;
        let mut cmd = tokio::process::Command::new(&exe);
        // A variant's steering vectors and adapters are named relative to the models folder.
        let models_dir = self.models_dir();
        if models_dir.is_dir() {
            cmd.current_dir(&models_dir);
        }
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
        *proc = Some(Running { child, model: model.clone(), ready: false, gpu, cuda: cuda_server, extras });
        {
            let mut shared = self.shared.lock().unwrap();
            shared.model = Some((model.clone(), ctx));
            shared.slot_owner = None;
        }
        (self.on_status)(ServerStatus::Loading(model.name.clone()));
        *self.loading_model.lock().unwrap() = Some((model.name.clone(), Instant::now()));
        let ready = self.wait_ready(proc, model, cancel).await;
        *self.loading_model.lock().unwrap() = None;
        if ready.is_ok() {
            let marker = loaded_marker(&model.name);
            if let Some(dir) = marker.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(marker, "");
        }
        ready
    }

    /// Runs `model` in a server of its own for one chat request, beside the main server, and stops it after. It runs
    /// on the processor, so the main model keeps the graphics card's memory.
    pub async fn run_side(&self, model: &LocalModel, ctx: u32, body: &Value, cancel: &CancellationToken) -> anyhow::Result<Value> {
        let Some(exe) = self.executable() else {
            bail!("Scoobert could not find llama.cpp. Reinstall Scoobert, or set the llama-server path in Settings.");
        };
        let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
        let key = random_hex(16);
        let mut args: Vec<String> = vec!["-m".into(), path_arg(&model.path)];
        if let Some(mmproj) = &model.mmproj {
            args.extend(["--mmproj".into(), path_arg(mmproj)]);
        }
        #[rustfmt::skip]
        args.extend([
            "-c", &ctx.to_string(), "--jinja", "-np", "1", "--cache-ram", "0", "--device", "none", "-ngl", "0", "--no-mmproj-offload",
            "--host", "127.0.0.1", "--port", &port.to_string(), "--no-webui",
        ].map(String::from));
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.args(&args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).env("LLAMA_API_KEY", &key).kill_on_drop(true);
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
        let mut child = cmd.spawn().with_context(|| format!("Could not start {}", exe.display()))?;
        let base = format!("http://127.0.0.1:{port}");
        let start = Instant::now();
        loop {
            if cancel.is_cancelled() {
                return Err(Cancelled.into());
            }
            if let Ok(Some(status)) = child.try_wait() {
                bail!("{} stopped while it loaded ({status}).", model.name);
            }
            if self.http.get(format!("{base}/health")).timeout(Duration::from_secs(2)).send().await.is_ok_and(|r| r.status().is_success()) {
                break;
            }
            if start.elapsed() > HEALTH_TIMEOUT {
                bail!("{} did not load within {} seconds.", model.name, HEALTH_TIMEOUT.as_secs());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let mut body = body.clone();
        body["stream"] = false.into();
        if let Some(fields) = body.as_object_mut() {
            fields.remove("stream_options");
        }
        let request = self.http.post(format!("{base}/v1/chat/completions")).bearer_auth(&key).timeout(Duration::from_secs(1800)).json(&body).send();
        let response = tokio::select! {
            r = request => r?,
            _ = cancel.cancelled() => return Err(Cancelled.into()),
        };
        let status = response.status();
        if !status.is_success() {
            bail!("{} could not answer: {status} {}", model.name, response.text().await.unwrap_or_default());
        }
        Ok(response.json().await?)
    }

    /// Deletes a model downloaded into the models folder with everything Scoobert stored for it: its files, its
    /// projector unless another model uses it, its folder when that leaves it empty, the model lab variants that run
    /// on its file, and the saved prompts and template copies of it and those variants. The server stops first when
    /// it runs any of them.
    pub async fn delete_model(&self, name: &str) -> anyhow::Result<()> {
        let models = self.models();
        let model = models.iter().find(|m| m.name == name && m.variant.is_none()).with_context(|| format!("There is no model named {name}."))?;
        if !model.path.starts_with(self.models_dir()) {
            bail!(crate::i18n::tr("Only models in the models folder can be deleted here."));
        }
        let variants: Vec<&LocalModel> = models.iter().filter(|v| v.variant.is_some() && v.path == model.path).collect();
        let loaded = self.loaded_model();
        if loaded.as_deref() == Some(name) || variants.iter().any(|v| loaded.as_deref() == Some(v.name.as_str())) {
            if self.in_use() {
                bail!(crate::i18n::tr("Scoobert is using this model. Delete it after the task finishes."));
            }
            self.stop().await;
        }
        for v in &variants {
            if let Some(folder) = &v.variant {
                std::fs::remove_dir_all(folder).with_context(|| crate::i18n::trf("Could not delete {path}", &[("path", &paths::display(folder))]))?;
            }
            forget_saved(&v.name);
        }
        let folder = model.path.parent().context("The model has no folder.")?.to_path_buf();
        let file_name = model.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let stem = model_stem(&file_name);
        let shard_re = regex::Regex::new(r"(?i)-\d{5}-of-\d{5}\.gguf(\.part)?$").unwrap();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                n == file_name || n == format!("{file_name}.part") || (n.starts_with(&format!("{stem}-")) && shard_re.is_match(&n))
            })
            .collect();
        if let Some(mmproj) = &model.mmproj
            && !models.iter().any(|m| m.name != name && m.mmproj.as_ref() == Some(mmproj))
        {
            files.push(mmproj.clone());
        }
        for file in &files {
            // A server that just stopped can hold the file open for a moment on Windows.
            let mut tries = 0;
            while let Err(err) = std::fs::remove_file(file) {
                tries += 1;
                if tries == 10 {
                    return Err(err).with_context(|| crate::i18n::trf("Could not delete {path}", &[("path", &paths::display(file))]));
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        if folder != self.models_dir() && std::fs::read_dir(&folder).is_ok_and(|mut d| d.next().is_none()) {
            let _ = std::fs::remove_dir(&folder);
        }
        forget_saved(name);
        Ok(())
    }

    /// The model the server is loading now, how long it has taken so far, and whether it never loaded on this
    /// computer before, if it is loading one.
    pub fn loading_for(&self) -> Option<(String, Duration, bool)> {
        let loading = self.loading_model.lock().unwrap().clone();
        loading.map(|(name, since)| {
            let first = !self.loaded_before(&name);
            (name, since.elapsed(), first)
        })
    }

    /// Whether the model has finished loading on this computer before. The first load reads the whole file from
    /// disk, which takes far longer than a load the system's file cache can serve.
    pub fn loaded_before(&self, model: &str) -> bool {
        // Saved prompts also prove a load, for models loaded before Scoobert kept the marker.
        let prefix = format!("{model}-");
        loaded_marker(model).is_file()
            || std::fs::read_dir(paths::get().slots()).into_iter().flatten().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix))
    }

    /// Waits for the server to finish loading model, and stops it when loading fails or is cancelled.
    async fn wait_ready(&self, proc: &mut Option<Running>, model: &LocalModel, cancel: &CancellationToken) -> anyhow::Result<()> {
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
                return Err(ExitedWhileLoading(format!("llama-server stopped while loading {}.\n{}", model.name, log_errors())).into());
            }
            if let Ok(h) = self.request("/health", None, Duration::from_secs(2), Some(cancel)).await
                && h["status"] == "ok"
            {
                if let Some(r) = proc.as_mut() {
                    r.ready = true;
                }
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
        format!("{}{kind}-{id}.bin", self.slot_stem())
    }

    /// The start of every slot file name for the loaded model and context size.
    fn slot_stem(&self) -> String {
        let ctx = self.shared.lock().unwrap().model.as_ref().map(|(_, ctx)| *ctx).unwrap_or(DEFAULT_CONTEXT);
        format!("{}{ctx}-", self.model_stem())
    }

    /// The start of every slot file name for the loaded model, at any context size.
    fn model_stem(&self) -> String {
        let name = self.shared.lock().unwrap().model.as_ref().map(|(m, _)| m.name.clone()).unwrap_or_else(|| "model".into());
        let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
        format!("{safe}-")
    }

    pub async fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        let body = json!({ "content": text, "add_special": true, "parse_special": true });
        let res = self.request("/tokenize", Some(&body), Duration::from_secs(60), None).await?;
        let tokens = res["tokens"].as_array().context("llama-server returned no tokens")?;
        tokens.iter().map(|t| t.as_i64().map(|t| t as i32).context("llama-server returned a token that is not a number")).collect()
    }

    /// Lets the graphics card try again the models it failed to load in this run.
    pub fn forget_gpu_failures(&self) {
        self.gpu_refused.lock().unwrap().clear();
    }

    /// Time since the slot was last saved to disk.
    pub fn since_save(&self) -> Duration {
        self.shared.lock().unwrap().last_save.elapsed()
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
        match (self.tokenize(prefix).await, self.tokenize(full).await) {
            (Ok(p), Ok(f)) => !p.is_empty() && f.starts_with(&p),
            _ => false,
        }
    }

    /// The prompt the server builds from a chat request, ending where the reply starts.
    pub async fn render(&self, payload: &Value) -> anyhow::Result<String> {
        let body = json!({ "messages": payload["messages"], "tools": payload["tools"], "chat_template_kwargs": payload["chat_template_kwargs"] });
        let res = self.request("/apply-template", Some(&body), Duration::from_secs(60), None).await?;
        res["prompt"].as_str().map(str::to_string).context("llama-server returned no prompt")
    }

    /// Streams a plain completion of `prompt`, passing each piece of text to `on_text`. Returns true when it stopped
    /// at `n_predict` rather than finishing.
    pub async fn complete(&self, prompt: &str, n_predict: u32, cancel: &CancellationToken, mut on_text: impl FnMut(&str)) -> anyhow::Result<bool> {
        let body = json!({ "prompt": prompt, "n_predict": n_predict, "stream": true, "cache_prompt": true });
        let last = self
            .stream_completion(&body, cancel, |event| {
                if let Some(text) = event["content"].as_str().filter(|t| !t.is_empty()) {
                    on_text(text);
                }
            })
            .await?;
        Ok(last["stopped_limit"].as_bool() == Some(true))
    }

    /// Sends a streamed `/completion` request, passing each event to `on_event`, and returns the final one.
    async fn stream_completion(&self, body: &Value, cancel: &CancellationToken, mut on_event: impl FnMut(&Value)) -> anyhow::Result<Value> {
        use futures::StreamExt;
        let send = self.http.post(format!("{}/completion", self.base_url())).bearer_auth(&self.api_key).json(body).send();
        let res = tokio::select! {
            r = send => r?,
            _ = cancel.cancelled() => return Err(Cancelled.into()),
        };
        if !res.status().is_success() {
            bail!("llama-server /completion returned {}", res.status());
        }
        let mut stream = res.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut last = Value::Null;
        loop {
            let chunk = tokio::select! {
                c = stream.next() => c,
                _ = cancel.cancelled() => return Err(Cancelled.into()),
            };
            // The stream is read to its end even after the final event, so the server has finished the request
            // before anything saves the slot.
            let Some(chunk) = chunk else { return Ok(last) };
            buf.extend_from_slice(&chunk?);
            while let Some(end) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=end).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim().strip_prefix("data:") else { continue };
                let Ok(event) = serde_json::from_str::<Value>(data.trim()) else { continue };
                on_event(&event);
                if event["stop"].as_bool() == Some(true) {
                    last = event;
                }
            }
        }
    }

    /// Whether the slot file `filename` holds exactly `tokens`.
    pub fn holds(&self, filename: &str, tokens: &[i32]) -> bool {
        saved_tokens(&paths::get().slots().join(filename)).as_deref() == Some(tokens)
    }

    /// Whether a request that needs the server is running.
    pub fn in_use(&self) -> bool {
        self.busy.load(Ordering::SeqCst) > 0
    }

    /// Reads `tokens` into the slot without generating, passing the tokens read so far and the total to
    /// `on_progress`. The slot is saved to `filename` after each step of about a minute and at the end, so a read
    /// that is stopped or closed partway resumes from the last save. `held` is how many tokens the slot holds
    /// exactly. `None` means it holds the start of `tokens` followed by unknown tokens, and then the first step
    /// covers all of `tokens`, because a model with recurrent layers cannot drop tokens from the end of its state.
    pub async fn fill(&self, tokens: &[i32], held: Option<usize>, filename: &str, cancel: &CancellationToken, mut on_progress: impl FnMut(u64, u64)) -> anyhow::Result<()> {
        let mut done = held.unwrap_or(0);
        let mut step = if held.is_some() { FIRST_STEP } else { tokens.len() };
        // The server's count of reused tokens in the first step, which progress is measured from.
        let mut base = None;
        while done < tokens.len() {
            let end = (done + step).min(tokens.len());
            let began = Instant::now();
            self.touch();
            let body = json!({ "prompt": &tokens[..end], "n_predict": 0, "cache_prompt": true, "id_slot": 0, "stream": true, "return_progress": true });
            self.stream_completion(&body, cancel, |event| {
                let p = &event["prompt_progress"];
                if p.is_object() {
                    let base = *base.get_or_insert(p["cache"].as_u64().unwrap_or(0));
                    on_progress(p["processed"].as_u64().unwrap_or(0).saturating_sub(base), (tokens.len() as u64).saturating_sub(base));
                }
            })
            .await?;
            let rate = (end - done) as f64 / began.elapsed().as_secs_f64().max(0.001);
            done = end;
            self.save(filename, &tokens[..end]).await?;
            step = ((rate * SAVE_EVERY.as_secs_f64()) as usize).clamp(FIRST_STEP, MAX_STEP);
        }
        Ok(())
    }

    /// Saves the slot under `filename`, which must hold `expected`. The server writes to a temporary name first, so
    /// closing Scoobert during a save leaves the previous file whole.
    pub async fn save(&self, filename: &str, expected: &[i32]) -> anyhow::Result<()> {
        let partial = format!("{PARTIAL}{filename}");
        self.request("/slots/0?action=save", Some(&json!({ "filename": partial })), Duration::from_secs(120), None).await?;
        let dir = paths::get().slots();
        // A request from another conversation can run between a read and its save, and saving that conversation's
        // state here would replace this file's good copy.
        if !saved_tokens(&dir.join(&partial)).is_some_and(|saved| saved.starts_with(expected)) {
            let _ = std::fs::remove_file(dir.join(&partial));
            bail!("Another request changed the slot before {filename} was saved, so the earlier copy was kept.");
        }
        std::fs::rename(dir.join(&partial), dir.join(filename)).context("Could not keep the saved prompt")?;
        self.shared.lock().unwrap().last_save = Instant::now();
        prune();
        Ok(())
    }

    /// Restores the slot file whose tokens make up the longest start of `tokens`, and returns how many tokens the
    /// slot then holds. A file that differs anywhere is skipped, since a model with recurrent layers can reuse
    /// nothing from it. Files saved at another context size count too, because a slot's state does not depend on
    /// the context size, and a file that starts `tokens` fits wherever `tokens` fits.
    pub async fn restore_longest(&self, tokens: &[i32]) -> usize {
        for (len, name) in self.saved_starts(tokens) {
            if self.restore(&name).await {
                return len;
            }
        }
        0
    }

    /// How many of `tokens` the longest saved prompt covers, without restoring it.
    pub fn longest_saved(&self, tokens: &[i32]) -> usize {
        self.saved_starts(tokens).first().map_or(0, |s| s.0)
    }

    /// The saved prompts for the loaded model and context size that `tokens` starts with, longest first.
    fn saved_starts(&self, tokens: &[i32]) -> Vec<(usize, String)> {
        let stem = self.model_stem();
        let mut found: Vec<(usize, String)> = std::fs::read_dir(paths::get().slots())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                // The context size follows the model name, so another model whose name starts the same is skipped.
                let rest = name.strip_prefix(&stem)?;
                let ctx = rest.split('-').next()?;
                if ctx.is_empty() || !ctx.bytes().all(|b| b.is_ascii_digit()) || !name.ends_with(".bin") {
                    return None;
                }
                let saved = saved_tokens(&e.path())?;
                (!saved.is_empty() && tokens.starts_with(&saved)).then_some((saved.len(), name))
            })
            .collect();
        found.sort_by(|a, b| b.0.cmp(&a.0));
        found
    }

    async fn restore(&self, filename: &str) -> bool {
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

fn scan(root: &Path, dir: &Path, depth: u32, out: &mut Vec<LocalModel>) {
    // A variant's folder holds steering vectors and adapters that are not models, so only its description counts.
    if depth > 0 && dir.join(lab::VARIANT_FILE).is_file() {
        out.extend(lab::model_from(root, dir));
        return;
    }
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let entries: Vec<_> = read.flatten().collect();
    let name_of = |e: &std::fs::DirEntry| e.file_name().to_string_lossy().into_owned();
    let ggufs: Vec<_> = entries
        .iter()
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false) && name_of(e).to_lowercase().ends_with(".gguf"))
        .collect();
    // Projectors are named mmproj-F16.gguf in some repositories and Model-mmproj-f16.gguf in others.
    let mmproj = if depth > 0 { ggufs.iter().find(|e| name_of(e).to_lowercase().contains("mmproj")) } else { None };
    let shard_re = regex::Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap();
    for e in &ggufs {
        let file_name = name_of(e);
        if file_name.to_lowercase().contains("mmproj") {
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
        out.push(LocalModel { family: name.clone(), name, path: e.path(), mmproj: mmproj.map(|m| m.path()), size, args: Vec::new(), variant: None });
    }
    if depth < 2 {
        for e in &entries {
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                scan(root, &e.path(), depth + 1, out);
            }
        }
    }
}

/// A model file the user added from another folder. A projector beside it enables images when the folder holds only
/// this model, or when the projector's name contains the model's, since a downloads folder can hold another model's.
pub fn added_model(path: &Path) -> Option<LocalModel> {
    let file_name = path.file_name()?.to_string_lossy().into_owned();
    let lower = file_name.to_lowercase();
    if !lower.ends_with(".gguf") || lower.contains("mmproj") || !path.is_file() {
        return None;
    }
    let shard_re = regex::Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap();
    let name = model_stem(&file_name);
    let siblings: Vec<PathBuf> = std::fs::read_dir(path.parent()?)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")))
        .collect();
    let file_of = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let size = if shard_re.is_match(&file_name) {
        siblings.iter().filter(|p| file_of(p).starts_with(&format!("{name}-")) && shard_re.is_match(&file_of(p))).filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum()
    } else {
        path.metadata().ok()?.len()
    };
    let models: std::collections::HashSet<String> = siblings.iter().map(file_of).filter(|n| !n.to_lowercase().contains("mmproj")).map(|n| model_stem(&n)).collect();
    let base = QUANT_SUFFIX.replace(&name, "").to_lowercase();
    let mmproj = siblings
        .iter()
        .filter(|p| file_of(p).to_lowercase().contains("mmproj"))
        .find(|p| models.len() == 1 || (!base.is_empty() && file_of(p).to_lowercase().contains(&base)))
        .cloned();
    Some(LocalModel { family: name.clone(), name, path: path.to_path_buf(), mmproj, size, args: Vec::new(), variant: None })
}

/// The number format at the end of a model's name, such as -Q4_K_M, .Q8_0, or -UD-IQ4_XS.
pub static QUANT_SUFFIX: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?i)[-_.](?:UD-)?(?:IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[A-Z0-9]+)?|Q\d+_\d+|TQ\d_\d|BF16|F16|F32|MXFP4(?:_MOE)?)$").unwrap()
});

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

/// Qwen 3.5 templates drop the reasoning of every earlier turn once a new user message arrives. The server's cache
/// still holds that reasoning, so each message re-read the whole previous turn. This returns a copy of the model's
/// template that keeps it, as later Qwen templates do with `preserve_thinking`, or None when no change is needed.
fn reasoning_keeping_template(model: &LocalModel) -> Option<PathBuf> {
    const CONDITION: &str = "loop.index0 > ns.last_query_index";
    let dir = paths::get().cache.join("templates");
    let safe: String = model.name.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    let file = dir.join(format!("{safe}.jinja"));
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if file.is_file() && modified(&file) >= modified(&model.path) {
        return Some(file);
    }
    let template = gguf::chat_template(&model.path)?;
    if !template.contains(CONDITION) || template.contains("preserve_thinking") {
        return None;
    }
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::write(&file, template.replace(CONDITION, "true")).ok()?;
    Some(file)
}

/// An empty file that marks a model as loaded once on this computer.
/// Deletes what Scoobert stored for a model besides its files: saved prompts, which can take gigabytes, the copy of
/// its chat template, and the record that it loaded before.
pub fn forget_saved(model: &str) {
    forget_slots(model);
    let safe: String = model.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    let _ = std::fs::remove_file(paths::get().cache.join("templates").join(format!("{safe}.jinja")));
    let _ = std::fs::remove_file(loaded_marker(model));
}

/// Deletes every saved prompt, as when the context's precision changes and none of them can be restored.
pub fn forget_all_slots() {
    for slot in std::fs::read_dir(paths::get().slots()).into_iter().flatten().flatten() {
        let _ = std::fs::remove_file(slot.path());
    }
}

/// Deletes a model's saved prompts, which can take gigabytes, as when they no longer match how the model runs.
pub fn forget_slots(model: &str) {
    let safe: String = model.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    // Slot files are named model-context-kind-id, and the context keeps one model's prefix from matching another's.
    let saved = regex::Regex::new(&format!(r"^{}-\d+-", regex::escape(&safe))).unwrap();
    for slot in std::fs::read_dir(paths::get().slots()).into_iter().flatten().flatten() {
        if saved.is_match(&slot.file_name().to_string_lossy()) {
            let _ = std::fs::remove_file(slot.path());
        }
    }
}

fn loaded_marker(model: &str) -> PathBuf {
    let safe: String = model.chars().map(|c| if c.is_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    paths::get().cache.join("loaded").join(safe)
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

/// The tokens a slot file holds, read from its header. The server writes them as -1, 1, the count, the tokens,
/// and 0. A plain list of tokens is read as it is.
fn saved_tokens(path: &Path) -> Option<Vec<i32>> {
    use std::io::Read;
    let mut file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let mut header = [0u8; 12];
    file.read_exact(&mut header).ok()?;
    let word = |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
    if word(0) != 0x6767_7371 {
        return None;
    }
    // No context is this long, so a larger count means the file is not in this format.
    let n = word(8) as usize;
    if n > 1 << 22 {
        return None;
    }
    let mut raw = vec![0u8; n * 4];
    file.read_exact(&mut raw).ok()?;
    token_list(raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn token_list(ints: Vec<i32>) -> Option<Vec<i32>> {
    match ints.as_slice() {
        [-1, 1, count, rest @ ..] if *count >= 0 && rest.len() == *count as usize + 1 => Some(rest[..*count as usize].to_vec()),
        _ if ints.iter().all(|&t| t >= 0) => Some(ints),
        _ => None,
    }
}

/// Deletes the saved prompts of conversation `id` for every model.
pub fn forget_chat(id: &str) {
    let suffix = format!("-chat-{id}.bin");
    for e in std::fs::read_dir(paths::get().slots()).into_iter().flatten().flatten() {
        if e.file_name().to_string_lossy().ends_with(&suffix) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// Deletes the oldest slot files once the folder passes its size cap, and saves that were cut short.
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
        .filter(|(path, _, modified)| {
            let partial = path.file_name().is_some_and(|n| n.to_string_lossy().starts_with(PARTIAL));
            let stale = modified.elapsed().is_ok_and(|age| age > Duration::from_secs(600));
            if partial && stale {
                let _ = std::fs::remove_file(path);
            }
            !partial
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_files_pair_with_their_own_projector() {
        let dir = std::env::temp_dir().join(format!("scoobert-added-{}", random_hex(4)));
        let own = dir.join("own");
        let mixed = dir.join("mixed");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::create_dir_all(&mixed).unwrap();
        for file in [own.join("Vision-7B.Q4_K_M.gguf"), own.join("mmproj-F16.gguf")] {
            std::fs::write(file, b"GGUF").unwrap();
        }
        for file in ["Alpha-Q4_K_M.gguf", "Beta-Q8_0.gguf", "beta-mmproj-f16.gguf"] {
            std::fs::write(mixed.join(file), b"GGUF").unwrap();
        }
        let model = added_model(&own.join("Vision-7B.Q4_K_M.gguf")).unwrap();
        assert_eq!(model.name, "Vision-7B.Q4_K_M");
        assert!(model.mmproj.is_some());
        assert!(added_model(&mixed.join("Alpha-Q4_K_M.gguf")).unwrap().mmproj.is_none());
        assert!(added_model(&mixed.join("Beta-Q8_0.gguf")).unwrap().mmproj.is_some());
        assert!(added_model(&mixed.join("beta-mmproj-f16.gguf")).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn reads_tokens_from_a_slot_file() {
        let path = std::env::temp_dir().join(format!("scoobert-slot-{}.bin", random_hex(6)));
        let mut bytes = Vec::new();
        for word in [0x6767_7371u32, 3, 7] {
            bytes.extend(word.to_le_bytes());
        }
        for int in [-1i32, 1, 3, 248045, 8678, 198, 0, 1, 3] {
            bytes.extend(int.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(saved_tokens(&path), Some(vec![248045, 8678, 198]));
        std::fs::write(&path, b"not a slot file").unwrap();
        assert_eq!(saved_tokens(&path), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn token_lists() {
        assert_eq!(token_list(vec![5, 6, 7]), Some(vec![5, 6, 7]));
        assert_eq!(token_list(vec![-1, 1, 2, 5, 6, 0]), Some(vec![5, 6]));
        assert_eq!(token_list(vec![-1, 1, 9, 5, 6, 0]), None);
    }
}
