//! Runs conversations: sends them to a local or hosted model, runs the tools it calls, and keeps the local
//! model's prompt cache and the project notes up to date.

pub mod conversation;
pub mod memory;
pub mod prompt;
pub mod providers;
pub mod sandbox;
pub mod stream;
pub mod tools;
pub mod web;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use conversation::{AssistantMessage, Conversation, Message, StopReason, SummaryDraft, ToolCall, ToolResult, UserMessage};
use providers::{Api, Provider};
use stream::{ChatRequest, Delta, Endpoint};
use tools::Shell;

use crate::llama::{LlamaServer, LocalModel, ServerStatus, SharedSettings};
use crate::store::{Approvals, HostedModel, Settings, Thinking};
use crate::util::{is_cancelled, now_millis, thousands};

const MAX_OUTPUT_TOKENS: u32 = 16_384;

pub type ConvId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Always,
    Deny,
}

#[derive(Debug, Clone)]
pub enum Event {
    Server(ServerStatus),
    /// What Scoobert is doing before the reply starts, such as loading the model. `None` clears it.
    Activity { conv: ConvId, text: Option<String> },
    Delta { conv: ConvId, delta: Delta },
    Message { conv: ConvId, message: Message },
    ToolStarted { conv: ConvId, call: ToolCall },
    ToolOutput { conv: ConvId, call_id: String, tail: String },
    Approval { conv: ConvId, id: u64, call: ToolCall },
    /// The run ended. `interrupted` is true when the task did not finish, so the window can offer Continue.
    Settled { conv: ConvId, context: u64, interrupted: bool },
    /// Messages before `kept_from` were replaced by a summary.
    Compacted { conv: ConvId, kept_from: usize },
    /// A conversation without a project started one and moved into it.
    ProjectStarted { conv: ConvId, path: PathBuf, file: PathBuf },
    /// The model named the conversation.
    Titled { conv: ConvId, title: String },
    /// Facts from a finished task were saved to these notes.
    NotesSaved { conv: ConvId, notes: Vec<String> },
    Error { conv: Option<ConvId>, message: String },
    NotesChanged { cwd: PathBuf },
    /// Scoobert is building a saved prompt for this model ahead of time, with the percent read so far. `None` means
    /// it finished or stopped.
    Preparing(Option<(String, u64)>),
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: ConvId,
    pub cwd: PathBuf,
    pub file: PathBuf,
    pub title: String,
    pub model: String,
    pub thinking: Thinking,
    pub messages: Vec<Message>,
    pub context: u64,
    pub running: bool,
    pub notice: Option<String>,
    /// Where a summary replaced the older messages, if one did.
    pub compacted_at: Option<usize>,
    /// The last task stopped before it finished, after a crash, a close, or Stop.
    pub interrupted: bool,
}

/// An entry in the model menu.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelOption {
    pub name: String,
    pub label: String,
    /// Empty for models on this computer, otherwise the provider's name.
    pub provider: String,
    pub usable: bool,
    pub context: u32,
    pub vision: bool,
    pub reasoning: bool,
    pub size: u64,
    pub memory_needed: u64,
}

struct Live {
    conv: Mutex<Conversation>,
    running: AtomicBool,
    run_cancel: Mutex<Option<CancellationToken>>,
    /// Cancels the cache step that runs before a reply.
    cache_cancel: Mutex<Option<CancellationToken>>,
    /// Cancels the note step that runs after a reply.
    note_cancel: Mutex<Option<CancellationToken>>,
    /// Held by cache and note work, so a new message waits for the slot to be saved first.
    background: tokio::sync::Mutex<()>,
    allowed: Mutex<HashSet<String>>,
    read_notes: Mutex<BTreeSet<String>>,
    run_notes: Mutex<Vec<String>>,
}

enum Target {
    Local(LocalModel),
    Hosted(HostedModel, Provider, String),
}

pub struct Host {
    settings: SharedSettings,
    pub llama: Arc<LlamaServer>,
    http: reqwest::Client,
    pub shell: Shell,
    events: mpsc::UnboundedSender<Event>,
    convs: Mutex<HashMap<ConvId, Arc<Live>>>,
    /// Stops the saved prompts being built ahead of time.
    bake_cancel: Mutex<Option<CancellationToken>>,
    approvals: Mutex<HashMap<u64, (ConvId, oneshot::Sender<Decision>)>>,
    next_approval: AtomicU64,
    keys: Mutex<HashMap<String, Option<String>>>,
    rt: tokio::runtime::Handle,
    pub isolation: sandbox::Isolation,
}

enum Verdict {
    Allow,
    Always,
    Deny(String),
}

const SUMMARY_PROMPT: &str = "Scoobert context step. The conversation is too long for the model's context, so older messages will be replaced by your summary and only the most recent ones stay. Start with one line that begins with Title: and names the whole task in three to six words. Then write what you need to continue the work without the older ones, as short bullet points under these headings, in this order: ## Goal, ## Next, ## Open problems, ## Decisions (with reasons), ## Done (files changed and why, with exact paths). Keep the whole summary under 400 words. Reply in plain text without tools.";
/// Facts from the note step are three labeled lines of up to 160 characters.
const REMEMBER_MAX_TOKENS: u32 = 160;
const TITLE_PROMPT: &str = "Scoobert title step. Reply with a short title for this conversation: 3 to 6 words that name the task, with no quotes and no period. Reply in plain text without tools.";
const TITLE_MAX_TOKENS: u32 = 24;
/// Summary length caps. A laptop CPU writes one to four tokens per second, so local summaries stay short.
const SUMMARY_MAX_TOKENS_LOCAL: u32 = 700;
const SUMMARY_MAX_TOKENS_HOSTED: u32 = 1500;
/// Share of the context the newest messages keep when older ones are summarized.
const KEEP_SHARE: f64 = 0.35;
/// Characters per token when estimating, set low so the estimate errs toward summarizing early.
const CHARS_PER_TOKEN: f64 = 3.2;
const MAX_RETRIES: u32 = 3;
/// Thinking tokens a local model may spend on a side request before it must answer.
const SIDE_THINKING_TOKENS: u32 = 16;
/// Sent with the Continue button, after a crash, a close, or Stop left a task unfinished.
const RESUME: &str = "<interrupted>Scoobert was closed or stopped before the last task finished. Continue that task from where it stopped. Check what is already done first, because a file may be only partly written.</interrupted>";
/// Attached to the first message after the user pressed Stop, which otherwise only shows a reply cut short.
const INTERRUPTED: &str = "<interrupted>The user stopped your previous reply before it finished. Follow this message. Do not resume the stopped work unless this message asks you to.</interrupted>";

static REASONING_LOCAL: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)qwen3|qwq|deepseek-r1|thinking|gpt-oss|magistral").unwrap());

impl Host {
    pub fn new(settings: SharedSettings, events: mpsc::UnboundedSender<Event>) -> Arc<Host> {
        let shell = Shell::detect();
        let probe = shell.clone();
        std::thread::spawn(move || {
            prompt::installed_tools(&probe);
        });
        let tx = events.clone();
        let llama = LlamaServer::new(settings.clone(), move |status| {
            let _ = tx.send(Event::Server(status));
        });
        Arc::new(Host {
            settings,
            llama,
            http: reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(20)).build().unwrap_or_default(),
            shell,
            events,
            convs: Mutex::new(HashMap::new()),
            bake_cancel: Mutex::new(None),
            approvals: Mutex::new(HashMap::new()),
            next_approval: AtomicU64::new(1),
            keys: Mutex::new(HashMap::new()),
            rt: tokio::runtime::Handle::current(),
            isolation: sandbox::Isolation::detect(),
        })
    }

    fn settings(&self) -> Settings {
        self.settings.read().unwrap().clone()
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    fn emit(&self, event: Event) {
        let _ = self.events.send(event);
    }

    // ---- models ----

    pub fn key(&self, provider: &str) -> Option<String> {
        let mut keys = self.keys.lock().unwrap();
        keys.entry(provider.to_string()).or_insert_with(|| crate::secrets::get(provider)).clone()
    }

    /// Saves or removes a key. Without a working credential store the key is kept until Scoobert closes,
    /// and the returned note says so.
    pub fn set_key(&self, provider: &str, key: Option<&str>) -> Option<String> {
        let stored = match key {
            Some(k) => crate::secrets::set(provider, k),
            None => crate::secrets::remove(provider),
        };
        self.keys.lock().unwrap().insert(provider.to_string(), key.map(String::from));
        match (stored, key) {
            (Err(err), Some(_)) => Some(format!("{err:#}. The key works until Scoobert closes.")),
            (Err(err), None) => Some(format!("{err:#}")),
            (Ok(()), _) => None,
        }
    }

    pub fn models(&self) -> Vec<ModelOption> {
        let s = self.settings();
        let mut out: Vec<ModelOption> = self
            .llama
            .models()
            .into_iter()
            .map(|m| {
                let ctx = self.llama.context_for(&m);
                ModelOption {
                    label: m.name.clone(),
                    provider: String::new(),
                    usable: true,
                    context: ctx,
                    vision: m.mmproj.is_some(),
                    reasoning: REASONING_LOCAL.is_match(&m.name),
                    size: m.size,
                    memory_needed: self.llama.memory_needed(&m, ctx),
                    name: m.name,
                }
            })
            .collect();
        for h in &s.hosted_models {
            let Some(p) = providers::find(&h.provider, &s.custom_providers) else { continue };
            out.push(ModelOption {
                name: h.reference(),
                label: h.name.clone(),
                provider: p.name.clone(),
                usable: self.key(&h.provider).is_some() || p.base_url.starts_with("http://"),
                context: h.context,
                vision: h.vision,
                reasoning: h.reasoning,
                size: 0,
                memory_needed: 0,
            });
        }
        out
    }

    fn target(&self, name: &str) -> anyhow::Result<Target> {
        let s = self.settings();
        if let Some(h) = s.hosted_models.iter().find(|h| h.reference() == name) {
            let provider = providers::find(&h.provider, &s.custom_providers).with_context(|| format!("The provider for {name} was removed."))?;
            let key = self.key(&h.provider).unwrap_or_default();
            if key.is_empty() && !provider.base_url.starts_with("http://") {
                bail!("{} needs an API key. Add one under Hosted models in Settings.", h.name);
            }
            return Ok(Target::Hosted(h.clone(), provider, key));
        }
        let local = self.llama.models();
        let found = local.iter().find(|m| m.name == name).or_else(|| local.first()).cloned();
        match found {
            Some(m) => Ok(Target::Local(m)),
            None => bail!(
                "No models found in {}. Download one in Settings, or add a hosted model with an API key.",
                crate::paths::display(&self.llama.models_dir())
            ),
        }
    }

    fn usable(&self, name: &str) -> bool {
        match self.target(name) {
            Ok(Target::Local(m)) => m.name == name,
            Ok(Target::Hosted(..)) => true,
            Err(_) => false,
        }
    }

    fn endpoint(&self, target: &Target) -> Endpoint {
        match target {
            Target::Local(m) => Endpoint {
                api: Api::OpenAi,
                base_url: format!("{}/v1", self.llama.base_url()),
                api_key: self.llama.api_key.clone(),
                model: m.name.clone(),
                local: true,
                thinking_style: providers::ThinkingStyle::None,
                reasoning: REASONING_LOCAL.is_match(&m.name),
                max_tokens_field: "max_tokens",
                provider_name: "llama-server".into(),
            },
            Target::Hosted(h, p, key) => Endpoint {
                api: p.api,
                base_url: p.base_url.clone(),
                api_key: key.clone(),
                model: h.id.clone(),
                local: false,
                thinking_style: p.thinking,
                reasoning: h.reasoning,
                max_tokens_field: p.max_tokens_field,
                provider_name: p.name.clone(),
            },
        }
    }

    fn max_tokens(&self, target: &Target) -> u32 {
        match target {
            Target::Local(m) => MAX_OUTPUT_TOKENS.min(self.llama.context_for(m) / 2),
            Target::Hosted(..) => MAX_OUTPUT_TOKENS,
        }
    }

    // ---- conversations ----

    fn live(&self, id: &str) -> anyhow::Result<Arc<Live>> {
        self.convs.lock().unwrap().get(id).cloned().context("That conversation is no longer open.")
    }

    fn snapshot(&self, live: &Live, notice: Option<String>) -> Snapshot {
        let context = self.context_used(live);
        let c = live.conv.lock().unwrap();
        Snapshot {
            id: c.id.clone(),
            cwd: c.cwd.clone(),
            file: c.file.clone(),
            title: c.display_title(),
            model: c.model.clone(),
            thinking: c.thinking,
            messages: c.messages.clone(),
            context,
            running: live.running.load(Ordering::SeqCst),
            notice,
            compacted_at: c.compaction.as_ref().map(|comp| comp.kept_from),
            interrupted: !live.running.load(Ordering::SeqCst) && needs_continue(&c.messages),
        }
    }

    /// Opens a saved conversation, or starts a new one when `file` is `None`.
    pub fn open(self: &Arc<Self>, cwd: &Path, file: Option<&Path>) -> anyhow::Result<Snapshot> {
        if let Some(file) = file {
            let existing = self.convs.lock().unwrap().values().find(|l| l.conv.lock().unwrap().file == file).cloned();
            if let Some(live) = existing {
                return Ok(self.snapshot(&live, None));
            }
        }
        let s = self.settings();
        let mut notice = None;
        let mut conv = match file {
            Some(f) => {
                if !conversation::is_session_file(f) {
                    bail!("That file is not one of Scoobert's conversations.");
                }
                Conversation::load(f)?
            }
            None => Conversation::new(cwd, &s.model, s.thinking),
        };
        if !self.usable(&conv.model) {
            let fallback = match self.target(&s.model) {
                Ok(Target::Local(m)) => m.name,
                Ok(Target::Hosted(h, ..)) => h.reference(),
                Err(_) => s.model.clone(),
            };
            if file.is_some() && fallback != conv.model {
                notice = Some(format!("{} is not available, so this conversation now uses {fallback}.", conv.model));
            }
            let _ = conv.set_model(&fallback);
        }
        let id = conv.id.clone();
        let live = Arc::new(Live {
            conv: Mutex::new(conv),
            running: AtomicBool::new(false),
            run_cancel: Mutex::new(None),
            cache_cancel: Mutex::new(None),
            note_cancel: Mutex::new(None),
            background: tokio::sync::Mutex::new(()),
            allowed: Mutex::new(HashSet::new()),
            read_notes: Mutex::new(BTreeSet::new()),
            run_notes: Mutex::new(Vec::new()),
        });
        self.convs.lock().unwrap().insert(id.clone(), live.clone());
        // Loading a different model to look at a conversation would unload the one in use, so that waits for typing.
        let model = live.conv.lock().unwrap().model.clone();
        if matches!(self.target(&model), Ok(Target::Local(m)) if self.llama.loaded_model().as_deref() == Some(&m.name)) {
            self.warm(&id);
        }
        Ok(self.snapshot(&live, notice))
    }

    /// Loads the conversation's model and its cached prompt in the background, so the next message starts sooner.
    pub fn warm(self: &Arc<Self>, id: &str) {
        let Ok(live) = self.live(id) else { return };
        if live.running.load(Ordering::SeqCst) {
            return;
        }
        self.stop_baking();
        let host = self.clone();
        self.rt.spawn(async move {
            let Ok(_guard) = live.background.try_lock() else { return };
            if let Err(err) = host.prepare(&live).await
                && !is_cancelled(&err)
            {
                eprintln!("[cache] {err:#}");
            }
        });
    }

    /// Closes conversations that are not running, except `keep`, and stops their cache work.
    pub fn close_idle(&self, keep: &str) {
        self.convs.lock().unwrap().retain(|id, l| {
            let keep = id == keep || l.running.load(Ordering::SeqCst);
            if !keep && let Some(t) = l.cache_cancel.lock().unwrap().as_ref() {
                t.cancel();
            }
            keep
        });
    }

    pub fn set_model(&self, id: &str, model: &str) -> anyhow::Result<()> {
        self.target(model)?;
        self.live(id)?.conv.lock().unwrap().set_model(model)
    }

    pub fn set_thinking(&self, id: &str, thinking: Thinking) -> anyhow::Result<()> {
        self.live(id)?.conv.lock().unwrap().set_thinking(thinking)
    }

    pub fn rename(&self, file: &Path, title: &str) -> anyhow::Result<()> {
        let open = self.convs.lock().unwrap().values().find(|l| l.conv.lock().unwrap().file == file).cloned();
        match open {
            Some(live) => live.conv.lock().unwrap().set_title(title),
            None => {
                if !conversation::is_session_file(file) {
                    bail!("That file is not one of Scoobert's conversations.");
                }
                Conversation::load(file)?.set_title(title)
            }
        }
    }

    pub fn delete(&self, file: &Path) -> anyhow::Result<()> {
        if !conversation::is_session_file(file) {
            bail!("That file is not one of Scoobert's conversations.");
        }
        self.convs.lock().unwrap().retain(|_, l| l.conv.lock().unwrap().file != file);
        Conversation::delete(file)
    }

    pub fn answer(&self, approval: u64, decision: Decision) {
        if let Some((_, tx)) = self.approvals.lock().unwrap().remove(&approval) {
            let _ = tx.send(decision);
        }
    }

    fn deny_pending(&self, conv: &str) {
        let mut approvals = self.approvals.lock().unwrap();
        let ids: Vec<u64> = approvals.iter().filter(|(_, (c, _))| c == conv).map(|(id, _)| *id).collect();
        for id in ids {
            if let Some((_, tx)) = approvals.remove(&id) {
                let _ = tx.send(Decision::Deny);
            }
        }
    }

    /// Files the model changed from message `from` on, which a rewind can put back.
    pub fn files_changed_after(&self, id: &str, from: usize) -> usize {
        let Ok(live) = self.live(id) else { return 0 };
        let conv = live.conv.lock().unwrap();
        let dir = checkpoint_dir(id);
        let paths: HashSet<PathBuf> = changing_calls(&conv.messages, from).filter_map(|c| tools::checkpoint_path(&dir, &c.id)).collect();
        paths.len()
    }

    /// Goes back to message `to`: stops any running work, forgets that message and everything after it, and
    /// with `restore_files` puts back the files the model changed since. Returns the conversation as it is now
    /// and any files that could not be restored.
    pub async fn rewind(self: &Arc<Self>, id: &str, to: usize, restore_files: bool) -> anyhow::Result<(Snapshot, Vec<String>)> {
        let live = self.live(id)?;
        self.abort(id);
        // The run notices the stop between steps, so this waits for it to finish before changing the history.
        for _ in 0..100 {
            if !live.running.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        if live.running.load(Ordering::SeqCst) {
            bail!("Scoobert is still stopping. Try again in a moment.");
        }
        let _guard = live.background.lock().await;
        let mut problems = Vec::new();
        {
            let mut conv = live.conv.lock().unwrap();
            if to > conv.messages.len() {
                bail!("That message is no longer in the conversation.");
            }
            if restore_files {
                // The earliest saved version of each file is its state before the discarded part began.
                let dir = checkpoint_dir(id);
                let mut done: HashSet<PathBuf> = HashSet::new();
                let calls: Vec<ToolCall> = changing_calls(&conv.messages, to).cloned().collect();
                for call in calls {
                    let Some(path) = tools::checkpoint_path(&dir, &call.id) else { continue };
                    if done.insert(path) && let Some(Err(e)) = tools::restore_checkpoint(&dir, &call.id) {
                        problems.push(e);
                    }
                }
            }
            conv.rewind(to)?;
        }
        live.read_notes.lock().unwrap().clear();
        live.allowed.lock().unwrap().clear();
        if self.llama.slot_owner().as_deref() == Some(id) {
            self.llama.set_slot_owner(None);
        }
        Ok((self.snapshot(&live, None), problems))
    }

    /// Stops the reply, or the cache step that runs before it.
    pub fn abort(&self, id: &str) {
        let Ok(live) = self.live(id) else { return };
        for token in [&live.run_cancel, &live.cache_cancel, &live.note_cancel] {
            if let Some(t) = token.lock().unwrap().as_ref() {
                t.cancel();
            }
        }
        self.deny_pending(id);
    }

    /// Sends a message and runs the conversation until the model stops calling tools.
    /// Sends a message. `resume` sends the hidden note that asks the model to finish an interrupted task.
    pub fn prompt(self: &Arc<Self>, id: &str, text: String, images: Vec<conversation::Image>, resume: bool) -> anyhow::Result<()> {
        let live = self.live(id)?;
        if live.running.swap(true, Ordering::SeqCst) {
            bail!("Scoobert is still working on the last message.");
        }
        self.stop_baking();
        let cancel = CancellationToken::new();
        *live.run_cancel.lock().unwrap() = Some(cancel.clone());
        if let Some(t) = live.note_cancel.lock().unwrap().as_ref() {
            t.cancel();
        }
        let host = self.clone();
        let conv_id = id.to_string();
        self.rt.spawn(async move {
            let result = host.run(&live, &conv_id, text, images, resume, &cancel).await;
            if let Err(err) = &result
                && !is_cancelled(err)
            {
                host.emit(Event::Error { conv: Some(conv_id.clone()), message: format!("{err:#}") });
            }
            host.emit(Event::Activity { conv: conv_id.clone(), text: None });
            live.running.store(false, Ordering::SeqCst);
            *live.run_cancel.lock().unwrap() = None;
            let context = host.context_used(&live);
            let interrupted = needs_continue(&live.conv.lock().unwrap().messages);
            host.emit(Event::Settled { conv: conv_id.clone(), context, interrupted });
            if result.is_ok() {
                host.after_run(live, conv_id);
            }
        });
        Ok(())
    }

    async fn run(self: &Arc<Self>, live: &Arc<Live>, id: &str, text: String, images: Vec<conversation::Image>, resume: bool, cancel: &CancellationToken) -> anyhow::Result<()> {
        let s = self.settings();
        let (cwd, first, model_name, interrupted) = {
            let mut c = live.conv.lock().unwrap();
            answer_dangling_calls(&mut c)?;
            (c.cwd.clone(), !c.has_user_message(), c.model.clone(), was_stopped(&c.messages))
        };
        let notes = {
            let mut read = live.read_notes.lock().unwrap();
            memory::attach(&crate::notes::Vault::new(cwd.join(&s.notes_folder)), &s.notes_folder, &text, &mut read)
        };
        *live.run_notes.lock().unwrap() = notes.related.clone();
        let mut context = notes.text;
        if let Some(list) = past_conversations(&cwd, id, &text) {
            context = if context.is_empty() { list } else { format!("{context}\n\n{list}") };
        }
        if resume || interrupted {
            let note = if resume { RESUME } else { INTERRUPTED };
            context = [note, &context].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join("\n\n");
        }
        let mut environment = String::new();
        if first {
            environment = prompt::environment_block(&cwd, &s.notes_folder, &self.shell);
            let session = prompt::session_block(&cwd, &s.notes_folder);
            context = if context.is_empty() { session } else { format!("{context}\n\n{session}") };
        }
        // Surrounding whitespace would change how the start of the text tokenizes and miss the saved cache.
        let user = Message::User(UserMessage { text: text.trim().to_string(), environment, context, images, time: now_millis() });
        live.conv.lock().unwrap().push(user.clone())?;
        self.emit(Event::Message { conv: id.to_string(), message: user });

        let target = self.target(&model_name)?;
        let ep = self.endpoint(&target);
        let _busy = matches!(target, Target::Local(_)).then(|| self.llama.busy());
        if let Target::Local(model) = &target {
            let wait = live.background.try_lock().is_err();
            if wait {
                self.emit(Event::Activity { conv: id.into(), text: Some("Getting ready...".into()) });
            }
            let _guard = tokio::select! {
                g = live.background.lock() => g,
                _ = cancel.cancelled() => return Err(crate::util::Cancelled.into()),
            };
            if self.llama.loaded_model().as_deref() != Some(&model.name) {
                // Loading takes from seconds to minutes and reports no progress, so the time so far is shown.
                let (events, conv, name, stop) = (self.events.clone(), id.to_string(), model.name.clone(), CancellationToken::new());
                let ticker = stop.clone();
                self.rt.spawn(async move {
                    let start = std::time::Instant::now();
                    loop {
                        let _ = events.send(Event::Activity { conv: conv.clone(), text: Some(format!("Loading {name}: {} s", start.elapsed().as_secs())) });
                        tokio::select! {
                            _ = ticker.cancelled() => break,
                            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
                        }
                    }
                });
                let loaded = tokio::select! {
                    r = self.llama.ensure(model) => r,
                    _ = cancel.cancelled() => Err(crate::util::Cancelled.into()),
                };
                stop.cancel();
                loaded?;
            }
            let cache = CancellationToken::new();
            *live.cache_cancel.lock().unwrap() = Some(cache.clone());
            let prepared = tokio::select! {
                r = self.prepare_before(live, id) => r,
                _ = cancel.cancelled() => { cache.cancel(); Err(crate::util::Cancelled.into()) }
            };
            *live.cache_cancel.lock().unwrap() = None;
            prepared?;
        }
        let mut compacted_for_error = false;
        // One summary per state of the conversation; summarizing again without new messages cannot help.
        let mut compacted_at = None;
        loop {
            let len = live.conv.lock().unwrap().messages.len();
            if compacted_at != Some(len) && self.too_long(live, &target) {
                self.compact(live, id, &target, &ep, true, cancel).await?;
                compacted_at = Some(len);
            }
            self.emit(Event::Activity { conv: id.into(), text: Some("Reading...".into()) });
            let (system, messages, thinking) = self.request_parts(live);
            let tools = self.tool_specs(live);
            let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget: None, max_tokens: self.max_tokens(&target) };
            let mut body = stream::payload(&ep, &req);
            if ep.local {
                body["return_progress"] = true.into();
            }
            let result = self.send_with_retries(id, &target, &ep, &body, cancel).await;
            let mut reply = match result {
                Ok(r) => r,
                Err(err) if is_cancelled(&err) => AssistantMessage { stop: StopReason::Aborted, model: ep.model.clone(), time: now_millis(), ..Default::default() },
                Err(err) if !compacted_for_error && compacted_at != Some(len) && context_overflow(&format!("{err:#}")) => {
                    // The estimate was short; summarize now and send the request again.
                    compacted_for_error = true;
                    self.compact(live, id, &target, &ep, false, cancel).await?;
                    compacted_at = Some(len);
                    continue;
                }
                Err(err) => AssistantMessage { stop: StopReason::Error, error: Some(format!("{err:#}")), model: ep.model.clone(), time: now_millis(), ..Default::default() },
            };
            if reply.stop == StopReason::Error && reply.error.is_none() {
                reply.error = Some("The reply ended with an error.".into());
            }
            let calls = reply.tool_calls.clone();
            let stop = reply.stop;
            let error = reply.error.clone();
            let message = Message::Assistant(reply);
            live.conv.lock().unwrap().push(message.clone())?;
            self.emit(Event::Message { conv: id.to_string(), message });
            if let Some(err) = error {
                self.emit(Event::Error { conv: Some(id.to_string()), message: err });
                return Ok(());
            }
            if stop != StopReason::ToolUse || calls.is_empty() || cancel.is_cancelled() {
                return Ok(());
            }
            for call in calls {
                // A new project moves the conversation, so each call reads the current folder.
                let cwd = live.conv.lock().unwrap().cwd.clone();
                let result = if cancel.is_cancelled() {
                    ToolResult { call_id: call.id.clone(), name: call.name.clone(), output: "The user stopped Scoobert before this ran.".into(), is_error: true, diff: None, time: now_millis() }
                } else {
                    self.run_tool(live, id, &cwd, &call, cancel).await
                };
                let message = Message::Tool(result);
                live.conv.lock().unwrap().push(message.clone())?;
                self.emit(Event::Message { conv: id.to_string(), message });
            }
            if cancel.is_cancelled() {
                return Ok(());
            }
        }
    }

    /// Sends a request, trying again after dropped connections, rate limits, and a model server that stopped.
    async fn send_with_retries(&self, id: &str, target: &Target, ep: &Endpoint, body: &Value, cancel: &CancellationToken) -> anyhow::Result<AssistantMessage> {
        let mut attempt = 0;
        loop {
            if let Target::Local(model) = target {
                self.llama.ensure(model).await?;
            }
            // The server holds back a tool call until it is complete, so a ticker reports how far the reply has got.
            let ticker = CancellationToken::new();
            if ep.local {
                let (llama, events, conv, stop) = (self.llama.clone(), self.events.clone(), id.to_string(), ticker.clone());
                self.rt.spawn(async move {
                    loop {
                        tokio::select! {
                            _ = stop.cancelled() => break,
                            _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {}
                        }
                        if let Some(n) = llama.generated_tokens().await.filter(|&n| n > 0) {
                            let _ = events.send(Event::Delta { conv: conv.clone(), delta: Delta::Generated(n) });
                        }
                    }
                });
            }
            let conv = id.to_string();
            let events = self.events.clone();
            let mut started = false;
            let result = stream::send(&self.http, ep, body, cancel, |delta| {
                if !started && !matches!(delta, Delta::Progress { .. }) {
                    started = true;
                    let _ = events.send(Event::Activity { conv: conv.clone(), text: None });
                }
                let _ = events.send(Event::Delta { conv: conv.clone(), delta });
            })
            .await;
            ticker.cancel();
            let retry = match &result {
                Err(err) => !is_cancelled(err) && transient(&format!("{err:#}")),
                // A connection that dropped before anything arrived is worth another try.
                Ok(r) => r.stop == StopReason::Error && r.text.is_empty() && r.tool_calls.is_empty(),
            };
            if !retry || attempt >= MAX_RETRIES {
                return result;
            }
            attempt += 1;
            let wait = std::time::Duration::from_secs(5 * 3u64.pow(attempt - 1));
            self.emit(Event::Activity { conv: id.into(), text: Some(format!("The request failed. Trying again in {} seconds...", wait.as_secs())) });
            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                _ = cancel.cancelled() => return Err(crate::util::Cancelled.into()),
            }
        }
    }

    fn context_window(&self, target: &Target) -> u64 {
        match target {
            Target::Local(m) => self.llama.context_for(m) as u64,
            Target::Hosted(h, ..) => h.context as u64,
        }
    }

    /// Whether the next request would leave too little room for the reply.
    fn too_long(&self, live: &Live, target: &Target) -> bool {
        let ctx = self.context_window(target);
        let thinking = live.conv.lock().unwrap().thinking;
        let reserve = (thinking.budget() as u64 + 2048).min(ctx / 3);
        self.prompt_tokens(live) + reserve > ctx
    }

    /// What the context meter shows: the size of the next request, which includes files read since the last reply.
    /// An empty conversation shows nothing yet.
    fn context_used(&self, live: &Live) -> u64 {
        if live.conv.lock().unwrap().messages.is_empty() { 0 } else { self.prompt_tokens(live) }
    }

    /// Tokens the next request sends: the server's count after the last reply plus an estimate for the messages
    /// since. The character estimate runs about a third high on code, so it covers the whole request only when no
    /// reply has been counted since the last summary.
    fn prompt_tokens(&self, live: &Live) -> u64 {
        let estimate = |chars: usize| (chars as f64 / CHARS_PER_TOKEN) as u64;
        let (system, messages, _) = self.request_parts(live);
        let whole = estimate(system.len() + self.tool_specs(live).iter().map(|t| t.to_string().len()).sum::<usize>() + messages_chars(&messages));
        let c = live.conv.lock().unwrap();
        let since = c.compaction.as_ref().map_or(0, |k| k.at);
        let counted = c.messages.iter().enumerate().rev().find_map(|(i, m)| match m {
            Message::Assistant(a) => a.usage.map(|u| (i, u.input + u.output)),
            _ => None,
        });
        match counted {
            Some((i, tokens)) if i >= since => tokens + estimate(messages_chars(&c.messages[i + 1..])),
            _ => whole,
        }
    }

    /// Replaces older messages with a summary the model writes, so a long task can continue.
    /// `extend` asks with the whole conversation, which a local model has cached, so only the question is read.
    /// It is false after the server rejected the conversation as too long, when only the older part fits.
    async fn compact(self: &Arc<Self>, live: &Arc<Live>, id: &str, target: &Target, ep: &Endpoint, extend: bool, cancel: &CancellationToken) -> anyhow::Result<()> {
        let conv = live.conv.lock().unwrap().clone();
        let start = conv.compaction.as_ref().map(|c| c.kept_from).unwrap_or(0);
        let keep_chars = (self.context_window(target) as f64 * KEEP_SHARE * CHARS_PER_TOKEN) as usize;
        let Some(kept_from) = kept_from(&conv.messages, start, keep_chars) else {
            bail!("The last message is too long for {}'s context. Start a new conversation, or pick a model with a larger context in Settings.", ep.model);
        };
        let (system, view, _) = self.request_parts(live);
        // The view starts with the previous summary when there is one, and maps conversation indexes after it.
        let offset = if conv.compaction.is_some() { 1 } else { 0 };
        let mut messages: Vec<Message> = if extend { view } else { view[..offset + (kept_from - start)].to_vec() };
        messages.push(Message::User(UserMessage { text: SUMMARY_PROMPT.into(), ..Default::default() }));
        let tools = self.tool_specs(live);
        let (thinking, thinking_budget) = side_thinking(ep, live);
        let max_tokens = if ep.local { SUMMARY_MAX_TOKENS_LOCAL } else { SUMMARY_MAX_TOKENS_HOSTED };
        let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget, max_tokens };
        let mut body = stream::payload(ep, &req);
        if ep.local {
            body["return_progress"] = true.into();
        }

        // A local model continues the summary an interrupted attempt started, when it covers the same messages.
        let draft = conv.summary_draft.clone().filter(|d| ep.local && d.start == start && d.kept_from == kept_from);
        let label = if draft.is_some() { "Continuing the summary of earlier messages" } else { "Summarizing earlier messages to make room" };
        self.emit(Event::Activity { conv: id.into(), text: Some(label.into()) });
        let written = Mutex::new(draft.as_ref().map(|d| d.text.clone()).unwrap_or_default());
        let pieces = AtomicU64::new(0);
        let on_text = |t: &str| {
            written.lock().unwrap().push_str(t);
            let n = pieces.fetch_add(1, Ordering::SeqCst) + 1;
            if n % 8 == 1 {
                let words = written.lock().unwrap().split_whitespace().count();
                self.emit(Event::Activity { conv: id.into(), text: Some(format!("{label}: {} words written", thousands(words as u64))) });
            }
        };
        let result: anyhow::Result<bool> = match &draft {
            Some(d) => self.continue_summary(&body, &d.text, max_tokens, cancel, on_text).await,
            None => {
                let reply = stream::send(&self.http, ep, &body, cancel, |delta| match delta {
                    Delta::Text(t) => on_text(&t),
                    progress @ Delta::Progress { .. } => self.emit(Event::Delta { conv: id.into(), delta: progress }),
                    _ => {}
                })
                .await;
                match reply {
                    Ok(r) if r.stop == StopReason::Aborted => Err(crate::util::Cancelled.into()),
                    Ok(r) if r.text.trim().is_empty() => Err(anyhow::anyhow!("The model could not summarize the conversation. {}", r.error.unwrap_or_default())),
                    Ok(r) if r.stop == StopReason::Error => Err(anyhow::anyhow!("The summary stopped partway. {}", r.error.unwrap_or_default())),
                    Ok(r) => Ok(r.stop == StopReason::Length),
                    Err(e) => Err(e),
                }
            }
        };
        let mut written = written.into_inner().unwrap();
        let truncated = match result {
            Ok(truncated) => truncated,
            Err(err) => {
                // What was written so far is kept, so the next attempt continues from it rather than starting over.
                if ep.local && !written.trim().is_empty() {
                    let _ = live.conv.lock().unwrap().set_summary_draft(SummaryDraft { start, kept_from, text: written });
                }
                return Err(err);
            }
        };
        // A summary that hit the length limit ends mid-line, and a half line reads as a fact.
        if truncated && let Some(end) = written.trim_end().rfind('\n') {
            written.truncate(end);
        }
        let (named, mut summary) = memory::split_title(written.trim());
        let folder = self.settings().notes_folder;
        let vault = crate::notes::Vault::new(conv.cwd.join(&folder));
        // The model's name for the task, else the conversation's title, since the first message can be a paragraph.
        let title = named.clone().or(conv.title.clone()).unwrap_or_else(|| conversation::quick_title(&conv.display_title()));
        if conv.title.is_none()
            && let Some(name) = &named
            && live.conv.lock().unwrap().set_title(name).is_ok()
        {
            self.emit(Event::Titled { conv: id.to_string(), title: name.clone() });
        }
        match memory::save_task_summary(&vault, &conv.id, &title, &summary) {
            Ok(rel) => summary.push_str(&format!("\n\nThis summary is also saved in {folder}/{rel}.")),
            Err(err) => eprintln!("[notes] {err:#}"),
        }
        let environment = prompt::environment_block(&conv.cwd, &folder, &self.shell);
        let session = prompt::session_block(&conv.cwd, &folder);
        live.conv.lock().unwrap().set_compaction(summary, kept_from, environment, session)?;
        // Notes attached to the summarized messages are gone from the request, so they can be attached again.
        live.read_notes.lock().unwrap().clear();
        self.emit(Event::NotesChanged { cwd: conv.cwd.clone() });
        if matches!(target, Target::Local(_)) {
            self.llama.set_slot_owner(None);
            // The summary message opens with the same environment block as a new conversation, so the saved prompt
            // for new conversations here covers the instructions and tools, and only the summary and kept messages
            // are read again.
            let (cwd, thinking, environment) = {
                let c = live.conv.lock().unwrap();
                (c.cwd.clone(), c.thinking, c.compaction.as_ref().map(|k| k.environment.clone()).unwrap_or_default())
            };
            if let Ok(prefix) = self.opening_prefix(target, &cwd, thinking, &environment).await {
                self.llama.restore(&self.llama.slot_file("shared", &LlamaServer::hash(&prefix))).await;
            }
        }
        self.emit(Event::Compacted { conv: id.to_string(), kept_from });
        Ok(())
    }

    /// Continues a summary from its draft. A chat request cannot start the reply with given text while thinking is
    /// on, so this sends the rendered prompt with the draft already written to the completion endpoint.
    async fn continue_summary(&self, body: &Value, draft: &str, max_tokens: u32, cancel: &CancellationToken, on_text: impl FnMut(&str)) -> anyhow::Result<bool> {
        // The model's own template writes the draft as its reply, followed by a marker. Cutting at the marker gives
        // the prompt in whatever format the model uses. A later user message keeps the server from treating the
        // reply as a prefill, which it refuses while thinking is on.
        const MARK: &str = "\u{E000}scoobert-draft-end\u{E000}";
        let mut payload = body.clone();
        if let Some(messages) = payload["messages"].as_array_mut() {
            messages.push(serde_json::json!({ "role": "assistant", "content": format!("{draft}{MARK}") }));
            messages.push(serde_json::json!({ "role": "user", "content": "Continue." }));
        }
        let rendered = self.llama.render(&payload).await?;
        let prompt = &rendered[..rendered.find(MARK).context("The chat template left out the summary draft")?];
        let used = (draft.len() as f64 / CHARS_PER_TOKEN) as u32;
        self.llama.complete(&prompt, max_tokens.saturating_sub(used).max(64), cancel, on_text).await
    }

    /// The system prompt and the messages as the model sees them, with any summary in place of older messages.
    fn request_parts(&self, live: &Live) -> (String, Vec<Message>, Thinking) {
        let s = self.settings();
        let (system, _) = self.pinned_prompt(live);
        let mut c = live.conv.lock().unwrap();
        // A summary made before its details were kept gets them now, fixed for the rest of this session, so notes
        // that change after a task do not change the first message and invalidate the cached prompt.
        let cwd = c.cwd.clone();
        if let Some(comp) = c.compaction.as_mut()
            && comp.environment.is_empty()
        {
            comp.environment = prompt::environment_block(&cwd, &s.notes_folder, &self.shell);
            comp.context = prompt::session_block(&cwd, &s.notes_folder);
        }
        let messages = match &c.compaction {
            Some(comp) if comp.kept_from <= c.messages.len() => {
                let mut out = vec![Message::User(UserMessage {
                    text: format!("<summary>\n{}\n</summary>", comp.summary.trim()),
                    environment: comp.environment.clone(),
                    context: comp.context.clone(),
                    images: Vec::new(),
                    time: 0,
                })];
                out.extend(c.messages[comp.kept_from..].iter().cloned());
                out
            }
            _ => c.messages.clone(),
        };
        (system, shorten_saved_writes(messages), c.thinking)
    }

    async fn run_tool(&self, live: &Live, id: &str, cwd: &Path, call: &ToolCall, cancel: &CancellationToken) -> ToolResult {
        let make = |output: String, is_error: bool, diff: Option<String>| ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            output,
            is_error,
            diff,
            time: now_millis(),
        };
        if call.name == "new_project" {
            return self.start_project(live, id, call);
        }
        match self.approve(live, id, cwd, call, cancel).await {
            Verdict::Deny(reason) => return make(reason, true, None),
            Verdict::Always => {
                live.allowed.lock().unwrap().insert(call.name.clone());
            }
            Verdict::Allow => {}
        }
        self.emit(Event::ToolStarted { conv: id.to_string(), call: call.clone() });
        let events = self.events.clone();
        let (conv, call_id) = (id.to_string(), call.id.clone());
        let limits = tools::Limits {
            unattended: self.settings().approvals == Approvals::Project,
            isolation: self.isolation.clone(),
            max_output: self.max_tool_output(live),
            notes: Some(cwd.join(&self.settings().notes_folder)),
            web: self.settings().web_access,
            checkpoints: Some(checkpoint_dir(id)),
        };
        let outcome = tools::run(call, cwd, &self.shell, &limits, cancel, move |tail| {
            let _ = events.send(Event::ToolOutput { conv: conv.clone(), call_id: call_id.clone(), tail });
        })
        .await;
        if tools::changes_files(&call.name) && self.inside_notes(cwd, call) {
            self.emit(Event::NotesChanged { cwd: cwd.to_path_buf() });
        }
        make(outcome.output, outcome.is_error, outcome.diff)
    }

    async fn approve(&self, live: &Live, id: &str, cwd: &Path, call: &ToolCall, cancel: &CancellationToken) -> Verdict {
        let s = self.settings();
        if s.approvals == Approvals::Auto || tools::is_read_only(&call.name) {
            return Verdict::Allow;
        }
        if tools::changes_files(&call.name) && self.inside_notes(cwd, call) {
            return Verdict::Allow;
        }
        if s.approvals == Approvals::Project {
            if !tools::changes_files(&call.name) || inside(cwd, &tools::target_path(cwd, call, Some(&cwd.join(&s.notes_folder))).unwrap_or_default()) {
                return Verdict::Allow;
            }
            return Verdict::Deny(
                "Scoobert is working unattended, so it only changes files inside the project folder. Continue without this change, and tell the user about it when they are back."
                    .into(),
            );
        }
        if live.allowed.lock().unwrap().contains(&call.name) {
            return Verdict::Allow;
        }
        let approval = self.next_approval.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.approvals.lock().unwrap().insert(approval, (id.to_string(), tx));
        self.emit(Event::Approval { conv: id.to_string(), id: approval, call: call.clone() });
        let decision = tokio::select! {
            d = rx => d.unwrap_or(Decision::Deny),
            _ = cancel.cancelled() => Decision::Deny,
        };
        match decision {
            Decision::Allow => Verdict::Allow,
            Decision::Always => Verdict::Always,
            Decision::Deny => Verdict::Deny("The user declined this tool call. Ask them how to proceed.".into()),
        }
    }

    fn tool_specs(&self, live: &Live) -> Vec<Value> {
        self.pinned_prompt(live).1
    }

    /// The system prompt and tools the conversation is sent with. They stay as they were when its last summary was
    /// made or it started, so an update to Scoobert does not invalidate its cached prompt. A change to the settings
    /// they depend on replaces them.
    fn pinned_prompt(&self, live: &Live) -> (String, Vec<Value>) {
        let s = self.settings();
        let mut c = live.conv.lock().unwrap();
        let general = is_general(&c.cwd);
        if let Some(p) = &c.prompt
            && p.notes_folder == s.notes_folder
            && p.web == s.web_access
            && p.general == general
        {
            return (p.system.clone(), p.tools.clone());
        }
        let system = prompt::system_prompt(&s.notes_folder, self.shell.tool_name());
        let tools = tools::specs(&self.shell, general, s.web_access);
        // Only a running task pins them, so opening a conversation to read it does not write to its file.
        if live.running.load(Ordering::SeqCst) {
            let pin = conversation::PinnedPrompt { system: system.clone(), tools: tools.clone(), notes_folder: s.notes_folder.clone(), web: s.web_access, general };
            if let Err(err) = c.set_prompt(pin) {
                eprintln!("[prompt] {err:#}");
            }
        }
        (system, tools)
    }

    /// Creates a project folder for a conversation that has none and moves the conversation into it.
    fn start_project(&self, live: &Live, id: &str, call: &ToolCall) -> ToolResult {
        let make = |output: String, is_error: bool| ToolResult { call_id: call.id.clone(), name: call.name.clone(), output, is_error, diff: None, time: now_millis() };
        let current = live.conv.lock().unwrap().cwd.clone();
        if !is_general(&current) {
            return make("A project is already open, so keep working in it.".into(), true);
        }
        let name = crate::notes::sanitize_name(call.arg("name"));
        let root = crate::paths::projects_root();
        let mut path = root.join(&name);
        // A folder that already has files belongs to something else, so the new one gets a number.
        let mut n = 2;
        while path.read_dir().map(|mut d| d.next().is_some()).unwrap_or(false) {
            path = root.join(format!("{name} {n}"));
            n += 1;
        }
        if let Err(e) = std::fs::create_dir_all(&path) {
            return make(format!("Could not create {}: {e}", crate::paths::display(&path)), true);
        }
        let moved = live.conv.lock().unwrap().move_to(&path);
        if let Err(e) = moved {
            return make(format!("Could not move the conversation: {e:#}"), true);
        }
        let file = live.conv.lock().unwrap().file.clone();
        self.emit(Event::ProjectStarted { conv: id.to_string(), path: path.clone(), file });
        make(format!("Started the project folder {}. Relative paths now resolve inside it.", crate::paths::display(&path)), false)
    }

    /// Tool output that fits in about a quarter of the model's context.
    fn max_tool_output(&self, live: &Live) -> usize {
        let model = live.conv.lock().unwrap().model.clone();
        let ctx = self.target(&model).map(|t| self.context_window(&t)).unwrap_or(32_768);
        ((ctx as f64 * 0.25 * CHARS_PER_TOKEN) as usize).clamp(4 * 1024, 50 * 1024)
    }

    /// Resolves links on both sides, so a link inside the notes folder cannot point a write elsewhere.
    fn inside_notes(&self, cwd: &Path, call: &ToolCall) -> bool {
        let raw = call.arguments.get("path").and_then(Value::as_str).unwrap_or_default();
        if raw.is_empty() || raw.starts_with('~') || raw.starts_with('@') {
            return false;
        }
        let dir = cwd.join(&self.settings().notes_folder);
        let Some(target) = tools::target_path(cwd, call, Some(&dir)) else { return false };
        let (notes, target) = (real_path(&dir), real_path(&target));
        target != notes && target.starts_with(&notes)
    }

    // ---- prompt cache ----

    /// Makes the server's slot hold this conversation's processed prompt before the first reply of a run.
    async fn prepare_before(self: &Arc<Self>, live: &Arc<Live>, id: &str) -> anyhow::Result<()> {
        let conv = live.conv.lock().unwrap().clone();
        let Target::Local(model) = self.target(&conv.model)? else { return Ok(()) };
        self.llama.ensure(&model).await?;
        if self.llama.slot_owner().as_deref() == Some(id) {
            return Ok(());
        }
        // The new user message is already in the conversation; the cache covers what came before it.
        let history = conv.messages.len() > 1;
        if history && self.llama.restore(&self.llama.slot_file("chat", id)).await {
            self.llama.set_slot_owner(Some(id.to_string()));
            return Ok(());
        }
        let cancel = live.cache_cancel.lock().unwrap().clone().unwrap_or_default();
        self.persist(live, !history, true, &cancel).await
    }

    /// Loads the model and the cached prompt when a conversation opens, so the first message starts sooner.
    async fn prepare(self: &Arc<Self>, live: &Arc<Live>) -> anyhow::Result<()> {
        let conv = live.conv.lock().unwrap().clone();
        let Target::Local(model) = self.target(&conv.model)? else { return Ok(()) };
        self.llama.ensure(&model).await?;
        if self.llama.slot_owner().as_deref() == Some(&conv.id) {
            return Ok(());
        }
        let history = conv.has_user_message();
        if history && self.llama.restore(&self.llama.slot_file("chat", &conv.id)).await {
            self.llama.set_slot_owner(Some(conv.id.clone()));
            return Ok(());
        }
        let cancel = CancellationToken::new();
        *live.cache_cancel.lock().unwrap() = Some(cancel.clone());
        let result = self.persist(live, !history, false, &cancel).await;
        *live.cache_cancel.lock().unwrap() = None;
        result
    }

    /// Fills the slot with the part of the request that the next request starts with, and saves it.
    /// `exclude_last` leaves out the newest message, which the upcoming request sends anyway.
    async fn persist(&self, live: &Live, shared: bool, exclude_last: bool, cancel: &CancellationToken) -> anyhow::Result<()> {
        let conv = live.conv.lock().unwrap().clone();
        let Target::Local(model) = self.target(&conv.model)? else { return Ok(()) };
        // Another conversation may have swapped the model since this one loaded it, and filling the wrong model's
        // slot costs minutes and evicts that conversation's cache.
        if self.llama.loaded_model().as_deref() != Some(&model.name) {
            return Ok(());
        }
        let target = Target::Local(model);
        let (prefix, name) = if shared {
            // The first message's own environment block, so the prefix matches what the request sends.
            let environment = match conv.messages.first() {
                Some(Message::User(u)) if !u.environment.is_empty() => u.environment.clone(),
                _ => prompt::environment_block(&conv.cwd, &self.settings().notes_folder, &self.shell),
            };
            let prefix = self.opening_prefix(&target, &conv.cwd, conv.thinking, &environment).await?;
            let name = self.llama.slot_file("shared", &LlamaServer::hash(&prefix));
            (prefix, name)
        } else {
            let (system, mut messages, thinking) = self.request_parts(live);
            if exclude_last {
                messages.pop();
            }
            if messages.iter().any(|m| matches!(m, Message::User(u) if !u.images.is_empty())) {
                return Ok(());
            }
            let tools = self.tool_specs(live);
            let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget: None, max_tokens: self.max_tokens(&target) };
            let prefix = self.llama.shared_prefix(&stream::payload(&self.endpoint(&target), &req), "").await?;
            (prefix, self.llama.slot_file("chat", &conv.id))
        };
        if shared && self.llama.restore(&name).await {
            self.llama.set_slot_owner(Some(conv.id.clone()));
            return Ok(());
        }
        let _busy = self.llama.busy();
        // The activity appears with the first progress report, so a fill that takes a moment shows nothing.
        let label = if shared { "Reading Scoobert's instructions" } else { "Reading the conversation" };
        let mut shown = false;
        let filled = self
            .llama
            .fill(&prefix, cancel, |done, total| {
                if !shown {
                    self.emit(Event::Activity { conv: conv.id.clone(), text: Some(label.into()) });
                    shown = true;
                }
                self.emit(Event::Delta { conv: conv.id.clone(), delta: Delta::Progress { done, total, prompt: 0 } });
            })
            .await;
        // A running task replaces the activity itself. Work after a reply clears it.
        if shown && !live.running.load(Ordering::SeqCst) {
            self.emit(Event::Activity { conv: conv.id.clone(), text: None });
        }
        filled?;
        self.llama.save(&name).await?;
        self.llama.set_slot_owner(Some(conv.id.clone()));
        Ok(())
    }

    /// The prompt every new conversation in `cwd` starts with: instructions, tools, and the environment block that
    /// opens its first message.
    async fn opening_prefix(&self, target: &Target, cwd: &Path, thinking: Thinking, environment: &str) -> anyhow::Result<String> {
        let s = self.settings();
        let system = prompt::system_prompt(&s.notes_folder, self.shell.tool_name());
        let tools = tools::specs(&self.shell, is_general(cwd), s.web_access);
        let req = ChatRequest { system: &system, messages: &[], tools: &tools, thinking, thinking_budget: None, max_tokens: self.max_tokens(target) };
        self.llama.shared_prefix(&stream::payload(&self.endpoint(target), &req), &format!("{environment}\n\n")).await
    }

    // ---- baking: saved prompts built ahead of time ----

    /// Replaces any baking in progress with a fresh token, and returns it.
    fn bake_token(&self) -> CancellationToken {
        let token = CancellationToken::new();
        if let Some(old) = self.bake_cancel.lock().unwrap().replace(token.clone()) {
            old.cancel();
        }
        token
    }

    /// Stops building saved prompts, because the user is about to need the model.
    fn stop_baking(&self) {
        if let Some(t) = self.bake_cancel.lock().unwrap().take() {
            t.cancel();
        }
    }

    /// Loads `model` and builds the saved prompts that new conversations in `places` start from, in the background.
    /// Returns false when the model is not on this computer or does not fit in free memory.
    pub fn preload(self: &Arc<Self>, model: &str, places: Vec<PathBuf>) -> bool {
        let Ok(Target::Local(m)) = self.target(model) else { return false };
        let loaded = self.llama.loaded_model().as_deref() == Some(&m.name);
        if !loaded && self.llama.memory_needed(&m, self.llama.context_for(&m)) > crate::sys::available_memory() {
            return false;
        }
        let (host, cancel) = (self.clone(), self.bake_token());
        self.rt.spawn(async move {
            if host.llama.ensure(&m).await.is_ok() {
                host.bake_all(&m, &places, &cancel).await;
            }
        });
        true
    }

    /// Builds the saved prompts for new conversations in `places` with the loaded model, once nothing is using it.
    pub fn bake_soon(self: &Arc<Self>, model: &str, places: Vec<PathBuf>) {
        let Ok(Target::Local(m)) = self.target(model) else { return };
        let (host, cancel) = (self.clone(), self.bake_token());
        self.rt.spawn(async move { host.bake_all(&m, &places, &cancel).await });
    }

    async fn bake_all(&self, model: &LocalModel, places: &[PathBuf], cancel: &CancellationToken) {
        let mut done = HashSet::new();
        for place in places {
            if cancel.is_cancelled() {
                break;
            }
            if !done.insert(place.clone()) {
                continue;
            }
            if let Err(err) = self.bake(place, model, cancel).await
                && !is_cancelled(&err)
            {
                eprintln!("[bake] {err:#}");
            }
        }
    }

    /// Builds and saves the prompt new conversations in `cwd` start from, unless it is saved already. Does nothing
    /// while a conversation is using the server or when another model is loaded.
    async fn bake(&self, cwd: &Path, model: &LocalModel, cancel: &CancellationToken) -> anyhow::Result<()> {
        let busy = self.llama.in_use() || self.convs.lock().unwrap().values().any(|l| l.running.load(Ordering::SeqCst));
        if busy || self.llama.loaded_model().as_deref() != Some(&model.name) {
            return Ok(());
        }
        let s = self.settings();
        let target = Target::Local(model.clone());
        let environment = prompt::environment_block(cwd, &s.notes_folder, &self.shell);
        let prefix = self.opening_prefix(&target, cwd, s.thinking, &environment).await?;
        let name = self.llama.slot_file("shared", &LlamaServer::hash(&prefix));
        if self.llama.has_slot_file(&name) {
            return Ok(());
        }
        self.emit(Event::Preparing(Some((model.name.clone(), 0))));
        let _busy = self.llama.busy();
        let filled = self
            .llama
            .fill(&prefix, cancel, |done, total| {
                if total > 0 {
                    self.emit(Event::Preparing(Some((model.name.clone(), done * 100 / total))));
                }
            })
            .await;
        let result = match filled {
            Ok(()) => self.llama.save(&name).await,
            Err(e) => Err(e),
        };
        // The slot now holds this prompt, so the next conversation restores its own.
        self.llama.set_slot_owner(None);
        self.emit(Event::Preparing(None));
        result
    }

    // ---- after a run ----

    fn after_run(self: &Arc<Self>, live: Arc<Live>, id: ConvId) {
        let host = self.clone();
        self.rt.spawn(async move {
            let _guard = live.background.lock().await;
            let s = host.settings();
            let conv = live.conv.lock().unwrap().clone();
            let local = matches!(host.target(&conv.model), Ok(Target::Local(_))) && host.llama.loaded_model().is_some();
            let cancel = CancellationToken::new();
            *live.cache_cancel.lock().unwrap() = Some(cancel.clone());
            if local
                && let Err(err) = host.persist(&live, false, false, &cancel).await
                && !is_cancelled(&err)
            {
                eprintln!("[cache] {err:#}");
            }
            *live.cache_cancel.lock().unwrap() = None;
            let hosted = matches!(host.target(&conv.model), Ok(Target::Hosted(..)));
            // After the first task, the model names the conversation. It reads only the question, because the
            // conversation before it is cached, and a title the user set is never replaced.
            if (local || hosted) && conv.title.is_none() && conv.user_messages() == 1 {
                let cancel = CancellationToken::new();
                *live.note_cancel.lock().unwrap() = Some(cancel.clone());
                match host.ask(&live, TITLE_PROMPT, TITLE_MAX_TOKENS, &cancel).await {
                    Ok(answer) => {
                        if let Some(title) = clean_title(&answer) {
                            let saved = live.conv.lock().unwrap().set_title(&title);
                            if saved.is_ok() {
                                host.emit(Event::Titled { conv: id.clone(), title });
                            }
                        }
                    }
                    Err(err) if !is_cancelled(&err) => eprintln!("[title] {err:#}"),
                    Err(_) => {}
                }
                *live.note_cancel.lock().unwrap() = None;
                if local {
                    host.llama.set_slot_owner(None);
                }
            }
            // The model is loaded and idle now, so this is when saved prompts for new conversations get built.
            let bake = || {
                if local {
                    host.bake_soon(&conv.model, vec![conv.cwd.clone(), crate::paths::projects_root()]);
                }
            };
            let related = live.run_notes.lock().unwrap().clone();
            let Some(task) = finished_task(&conv, &s.notes_folder, related) else {
                bake();
                return;
            };
            let mut facts = Vec::new();
            if (local || hosted) && s.remember_step && memory::worth_noting(&task) {
                host.emit(Event::Activity { conv: id.clone(), text: Some("Taking notes...".into()) });
                let cancel = CancellationToken::new();
                *live.note_cancel.lock().unwrap() = Some(cancel.clone());
                let question = memory::remember_prompt(&memory::topic_names(&crate::notes::Vault::new(conv.cwd.join(&s.notes_folder))));
                match host.ask(&live, &question, REMEMBER_MAX_TOKENS, &cancel).await {
                    Ok(answer) => facts = memory::parse_facts(&answer),
                    Err(err) if !is_cancelled(&err) => eprintln!("[notes] {err:#}"),
                    Err(_) => {}
                }
                *live.note_cancel.lock().unwrap() = None;
                if local {
                    // The note step leaves its own question in the slot, so the next request restores the saved file.
                    host.llama.set_slot_owner(None);
                }
                host.emit(Event::Activity { conv: id.clone(), text: None });
            }
            let vault = crate::notes::Vault::new(conv.cwd.join(&s.notes_folder));
            match memory::record_task(&vault, s.activity_log, &task, &facts) {
                Ok(saved) => {
                    host.emit(Event::NotesChanged { cwd: conv.cwd.clone() });
                    if !saved.is_empty() {
                        host.emit(Event::NotesSaved { conv: id.clone(), notes: saved });
                    }
                }
                Err(err) => eprintln!("[notes] {err:#}"),
            }
            bake();
        });
    }

    /// Asks one short question at the end of the conversation, with thinking off. The request extends the cached
    /// conversation, so only the question is read, and the answer is short because each token takes about a
    /// second on a CPU.
    async fn ask(&self, live: &Live, question: &str, max_tokens: u32, cancel: &CancellationToken) -> anyhow::Result<String> {
        let conv = live.conv.lock().unwrap().clone();
        let target = self.target(&conv.model)?;
        let ep = self.endpoint(&target);
        let (system, mut messages, _) = self.request_parts(live);
        messages.push(Message::User(UserMessage { text: question.into(), ..Default::default() }));
        let tools = self.tool_specs(live);
        let (thinking, thinking_budget) = side_thinking(&ep, live);
        let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget, max_tokens };
        let body = stream::payload(&ep, &req);
        let _busy = ep.local.then(|| self.llama.busy());
        let reply = stream::send(&self.http, &ep, &body, cancel, |_| {}).await?;
        if reply.stop == StopReason::Aborted {
            return Err(crate::util::Cancelled.into());
        }
        Ok(reply.text)
    }

    pub async fn shutdown(&self) {
        let tokens: Vec<CancellationToken> = self
            .convs
            .lock()
            .unwrap()
            .values()
            .flat_map(|l| [&l.run_cancel, &l.cache_cancel, &l.note_cancel].into_iter().filter_map(|t| t.lock().unwrap().clone()).collect::<Vec<_>>())
            .collect();
        self.stop_baking();
        for t in tokens {
            t.cancel();
        }
        if tokio::time::timeout(std::time::Duration::from_secs(4), self.llama.stop()).await.is_err() {
            self.llama.kill_now();
        }
    }
}

/// Describes the run that just ended when it finished normally and changed files outside the notes folder.
fn finished_task(conv: &Conversation, notes_folder: &str, related: Vec<String>) -> Option<prompt::FinishedTask> {
    let start = conv.messages.iter().rposition(|m| matches!(m, Message::User(_)))?;
    let Message::User(request) = &conv.messages[start] else { return None };
    let run = &conv.messages[start + 1..];
    let last = run.iter().rev().find_map(|m| match m {
        Message::Assistant(a) => Some(a),
        _ => None,
    })?;
    if last.stop != StopReason::Stop {
        return None;
    }
    let notes_dir = real_path(&conv.cwd.join(notes_folder));
    let mut changed: Vec<String> = Vec::new();
    for m in run {
        let Message::Assistant(a) = m else { continue };
        for c in &a.tool_calls {
            if !tools::changes_files(&c.name) {
                continue;
            }
            let Some(path) = tools::target_path(&conv.cwd, c, Some(&conv.cwd.join(notes_folder))) else { continue };
            let real = real_path(&path);
            if real.starts_with(&notes_dir) {
                continue;
            }
            let shown = real.strip_prefix(real_path(&conv.cwd)).map(crate::paths::display).unwrap_or_else(|_| crate::paths::display(&path));
            if !changed.contains(&shown) {
                changed.push(shown);
            }
        }
    }
    if changed.is_empty() {
        return None;
    }
    Some(prompt::FinishedTask {
        request: request.text.split_whitespace().collect::<Vec<_>>().join(" "),
        summary: last.text.trim().to_string(),
        changed,
        related,
    })
}

/// The model's title answer, cleaned of labels, quotes, and trailing punctuation, when it is usable.
fn clean_title(answer: &str) -> Option<String> {
    let line = answer.lines().map(str::trim).find(|l| !l.is_empty())?;
    let line = line.trim_start_matches(['#', '*', ' ']);
    let line = line.strip_prefix("Title:").or_else(|| line.strip_prefix("title:")).unwrap_or(line);
    let (quotes, stops) = (['"', '\'', '*', '`'], ['.', '!', ':']);
    let title = line.trim().trim_end_matches(stops).trim_matches(quotes).trim_end_matches(stops).trim();
    let words = title.split_whitespace().count();
    (words >= 1 && words <= 10 && title.chars().count() <= 80).then(|| title.to_string())
}

/// Long file contents the model already saved are replaced by a line saying where they went. The file is on disk
/// for the model to read again, and keeping every written file in the prompt would fill a small context fast.
fn shorten_saved_writes(mut messages: Vec<Message>) -> Vec<Message> {
    const LONG: usize = 1500;
    let saved: HashSet<String> = messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) if !t.is_error => Some(t.call_id.clone()),
            _ => None,
        })
        .collect();
    for m in &mut messages {
        let Message::Assistant(a) = m else { continue };
        for call in &mut a.tool_calls {
            if !saved.contains(&call.id) {
                continue;
            }
            let path = call.arg("path").to_string();
            let Some(args) = call.arguments.as_object_mut() else { continue };
            for key in ["content", "new_text", "old_text"] {
                let Some(len) = args.get(key).and_then(Value::as_str).map(|s| s.chars().count()).filter(|&n| n > LONG) else { continue };
                let note = if key == "content" {
                    format!("[{len} characters written to {path}. Read the file to see them.]")
                } else {
                    format!("[{len} characters, now in {path}.]")
                };
                args.insert(key.into(), note.into());
            }
        }
    }
    messages
}

/// Thinking for a short side request, such as a summary, a title, or the note step. A local model keeps the
/// conversation's thinking level, because changing it changes the rendered prompt and loses the cache, and gets a
/// tiny budget instead. A hosted model simply does not think.
fn side_thinking(ep: &Endpoint, live: &Live) -> (Thinking, Option<u32>) {
    if ep.local { (live.conv.lock().unwrap().thinking, Some(SIDE_THINKING_TOKENS)) } else { (Thinking::Off, None) }
}

/// Where a conversation keeps the previous versions of files the model changed.
fn checkpoint_dir(id: &str) -> PathBuf {
    crate::paths::get().data.join("checkpoints").join(id)
}

/// Edit and write calls from message `from` on, oldest first.
fn changing_calls(messages: &[Message], from: usize) -> impl Iterator<Item = &ToolCall> {
    messages[from.min(messages.len())..]
        .iter()
        .filter_map(|m| match m {
            Message::Assistant(a) => Some(a.tool_calls.iter()),
            _ => None,
        })
        .flatten()
        .filter(|c| tools::changes_files(&c.name))
}

/// Whether a conversation has no project: it works in the shared folder where new projects are created.
pub fn is_general(cwd: &Path) -> bool {
    cwd == crate::paths::projects_root()
}

static PAST: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?i)\b(other|previous|past|earlier|last|older|another)\s+(conversations?|chats?|sessions?)\b|\b(conversations?|chats?|sessions?)\s+(about|where|when|from)\b|\bwhat did (we|you|i) (do|say|talk|discuss|decide|try)\b|\bconversation:[0-9a-f]{4,}").unwrap()
});

/// A list of the project's other conversations, attached only when the message asks about past conversations,
/// so the model never reads them otherwise.
fn past_conversations(cwd: &Path, current: &str, message: &str) -> Option<String> {
    if !PAST.is_match(message) {
        return None;
    }
    let list: Vec<String> = Conversation::list(cwd)
        .into_iter()
        .filter(|s| s.id != current)
        .take(15)
        .map(|s| {
            let date = chrono::DateTime::from_timestamp_millis(s.modified).map(|d| d.with_timezone(&chrono::Local).format("%Y-%m-%d").to_string()).unwrap_or_default();
            format!("- {} ({date}): read conversation:{}", s.title, &s.id[..s.id.len().min(8)])
        })
        .collect();
    if list.is_empty() {
        return Some("<past_conversations>There are no other conversations in this project.</past_conversations>".into());
    }
    Some(format!("<past_conversations>
The user asked about earlier conversations. Read one with the read tool, such as read conversation:<id>.
{}
</past_conversations>", list.join("\n")))
}

/// Whether the conversation ends in the middle of a task: a message with no reply, a tool result the model has
/// not answered, or a reply that was stopped, failed, or still had tool calls to make.
pub fn needs_continue(messages: &[Message]) -> bool {
    match messages.last() {
        None => false,
        Some(Message::User(_)) | Some(Message::Tool(_)) => true,
        Some(Message::Assistant(a)) => a.stop != StopReason::Stop || !a.tool_calls.is_empty(),
    }
}

/// Adds a result for each tool call that never ran because Scoobert closed first. Providers reject a conversation
/// with a tool call and no result.
fn answer_dangling_calls(conv: &mut Conversation) -> anyhow::Result<()> {
    let Some(last) = conv.messages.iter().rposition(|m| matches!(m, Message::Assistant(a) if !a.tool_calls.is_empty())) else {
        return Ok(());
    };
    let Message::Assistant(a) = &conv.messages[last] else { return Ok(()) };
    let answered: HashSet<String> = conv.messages[last + 1..]
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) => Some(t.call_id.clone()),
            _ => None,
        })
        .collect();
    let missing: Vec<ToolCall> = a.tool_calls.iter().filter(|c| !answered.contains(&c.id)).cloned().collect();
    if missing.is_empty() || conv.messages[last + 1..].iter().any(|m| matches!(m, Message::User(_))) {
        return Ok(());
    }
    for c in missing {
        conv.push(Message::Tool(ToolResult {
            call_id: c.id,
            name: c.name,
            output: "Scoobert closed before this ran.".into(),
            is_error: true,
            diff: None,
            time: now_millis(),
        }))?;
    }
    Ok(())
}

/// Whether the last run ended because the user pressed Stop.
fn was_stopped(messages: &[Message]) -> bool {
    match messages.last() {
        Some(Message::Assistant(a)) => a.stop == StopReason::Aborted,
        Some(Message::Tool(t)) => t.is_error && (t.output.starts_with("The user stopped") || t.output.ends_with("[The user stopped the command.]")),
        _ => false,
    }
}

/// Where the kept part of a conversation starts when older messages are summarized: the newest messages that fit
/// in `keep_chars`, starting on a user or model turn and never on a tool result, after `start`.
fn kept_from(messages: &[Message], start: usize, keep_chars: usize) -> Option<usize> {
    let len = messages.len();
    let mut size = 0;
    let mut cut = len;
    while cut > start {
        size += messages_chars(&messages[cut - 1..cut]);
        if size > keep_chars {
            break;
        }
        cut -= 1;
    }
    let cut = cut.min(len.saturating_sub(1)).max(start + 1);
    let turn = |i: &usize| !matches!(messages[*i], Message::Tool(_));
    (cut..len).find(turn).or_else(|| (start + 1..cut).rev().find(turn))
}

fn messages_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| match m {
            Message::User(u) => u.text.len() + u.context.len() + u.images.len() * 3000,
            Message::Assistant(a) => a.thinking.len() + a.text.len() + a.tool_calls.iter().map(|c| c.arguments.to_string().len() + 40).sum::<usize>(),
            Message::Tool(t) => t.output.len() + 40,
        })
        .sum()
}

/// Errors that mean the request did not fit in the model's context.
fn context_overflow(error: &str) -> bool {
    let e = error.to_lowercase();
    ["exceeds the available context", "context_length_exceeded", "maximum context length", "prompt is too long", "too many tokens", "context window", "context size"]
        .iter()
        .any(|p| e.contains(p))
}

/// Errors that usually pass: dropped connections, rate limits, overloaded servers, a model server restarting.
fn transient(error: &str) -> bool {
    let e = error.to_lowercase();
    !context_overflow(&e)
        && ["could not reach", "connection", "dropped", "timed out", "limiting requests", " 429", " 500", " 502", " 503", " 504", " 529", "overloaded"]
            .iter()
            .any(|p| e.contains(p))
}

/// Whether `path` is inside the project folder, after resolving links.
fn inside(cwd: &Path, path: &Path) -> bool {
    !path.as_os_str().is_empty() && real_path(path).starts_with(real_path(cwd))
}

/// The canonical path, resolving links in the part of the path that exists.
fn real_path(p: &Path) -> PathBuf {
    let mut existing = p.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        match (existing.file_name().map(|n| n.to_os_string()), existing.parent()) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => return p.to_path_buf(),
        }
    }
    let mut out = dunce_canonical(&existing);
    for part in rest.into_iter().rev() {
        out.push(part);
    }
    out
}

/// Canonicalizes without the `\\?\` prefix Windows adds, so paths compare equal to ones built by joining.
fn dunce_canonical(p: &Path) -> PathBuf {
    let c = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let s = c.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::conversation::{AssistantMessage, Message, ToolCall, ToolResult, UserMessage};
    use super::conversation::StopReason;
    use super::{clean_title, kept_from, needs_continue};

    #[test]
    fn shortens_long_saved_writes_only() {
        let big = "x".repeat(3000);
        let write = |id: &str| ToolCall { id: id.into(), name: "write".into(), arguments: serde_json::json!({ "path": "a.ts", "content": big }) };
        let messages = vec![
            Message::Assistant(AssistantMessage { tool_calls: vec![write("ok"), write("failed")], ..Default::default() }),
            Message::Tool(ToolResult { call_id: "ok".into(), name: "write".into(), output: "Created".into(), is_error: false, diff: None, time: 0 }),
            Message::Tool(ToolResult { call_id: "failed".into(), name: "write".into(), output: "Could not write".into(), is_error: true, diff: None, time: 0 }),
        ];
        let out = super::shorten_saved_writes(messages);
        let Message::Assistant(a) = &out[0] else { panic!() };
        assert_eq!(a.tool_calls[0].arg("content"), "[3000 characters written to a.ts. Read the file to see them.]");
        assert_eq!(a.tool_calls[1].arg("content").len(), 3000);
    }

    #[test]
    fn spots_unfinished_tasks() {
        let done = Message::Assistant(AssistantMessage { text: "Done.".into(), stop: StopReason::Stop, ..Default::default() });
        let stopped = Message::Assistant(AssistantMessage { text: "Half".into(), stop: StopReason::Aborted, ..Default::default() });
        assert!(!needs_continue(&[]));
        assert!(!needs_continue(&[user(10), done.clone()]));
        assert!(needs_continue(&[user(10)]));
        assert!(needs_continue(&[user(10), call(5)]));
        assert!(needs_continue(&[user(10), call(5), result(5)]));
        assert!(needs_continue(&[user(10), stopped]));
    }

    #[test]
    fn cleans_model_titles() {
        assert_eq!(clean_title("Title: \"RPG sprite system\".\n").as_deref(), Some("RPG sprite system"));
        assert_eq!(clean_title("**Secure auth cookies**").as_deref(), Some("Secure auth cookies"));
        assert_eq!(clean_title(""), None);
        assert_eq!(clean_title(&"word ".repeat(20)), None);
    }

    fn user(n: usize) -> Message {
        Message::User(UserMessage { text: "u".repeat(n), ..Default::default() })
    }
    fn call(n: usize) -> Message {
        let call = ToolCall { id: "c".into(), name: "read".into(), arguments: serde_json::json!({ "path": "x".repeat(n) }) };
        Message::Assistant(AssistantMessage { tool_calls: vec![call], ..Default::default() })
    }
    fn result(n: usize) -> Message {
        Message::Tool(ToolResult { call_id: "c".into(), name: "read".into(), output: "o".repeat(n), is_error: false, diff: None, time: 0 })
    }

    #[test]
    fn keeps_newest_turns_without_splitting_tool_results() {
        let m = vec![user(100), call(10), result(5000), call(10), result(100), user(50)];
        // Room for the last three messages: the cut lands on the second call, not its result.
        assert_eq!(kept_from(&m, 0, 400), Some(3));
        // A huge newest result still keeps the call that produced it.
        let m = vec![user(100), call(10), result(50_000)];
        assert_eq!(kept_from(&m, 0, 1000), Some(1));
        // Nothing after the previous summary to summarize.
        assert_eq!(kept_from(&m, 2, 1000), None);
    }
}
