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

use conversation::{AssistantMessage, Conversation, Message, StopReason, ToolCall, ToolResult, UserMessage};
use providers::{Api, Provider};
use stream::{ChatRequest, Delta, Endpoint};
use tools::Shell;

use crate::llama::{LlamaServer, LocalModel, ServerStatus, SharedSettings};
use crate::store::{Approvals, HostedModel, Settings, Thinking};
use crate::util::{is_cancelled, now_millis};

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

const SUMMARY_PROMPT: &str = "Scoobert context step. The conversation is too long for the model's context, so older messages will be replaced by your summary and only the most recent ones stay. Write what you need to continue the work without the older ones, as short bullet points under these headings: ## Goal, ## Done (files changed and why, with exact paths), ## Decisions (with reasons), ## Next, ## Open problems. Reply in plain text without tools.";
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
        let c = live.conv.lock().unwrap();
        Snapshot {
            id: c.id.clone(),
            cwd: c.cwd.clone(),
            file: c.file.clone(),
            title: c.display_title(),
            model: c.model.clone(),
            thinking: c.thinking,
            messages: c.messages.clone(),
            context: c.context_tokens(),
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
        self.convs.lock().unwrap().insert(id, live.clone());
        let host = self.clone();
        let warm = live.clone();
        self.rt.spawn(async move {
            let _guard = warm.background.lock().await;
            if let Err(err) = host.prepare(&warm).await
                && !is_cancelled(&err)
            {
                eprintln!("[cache] {err:#}");
            }
        });
        Ok(self.snapshot(&live, notice))
    }

    /// Closes conversations that are not running, except `keep`.
    pub fn close_idle(&self, keep: &str) {
        self.convs.lock().unwrap().retain(|id, l| id == keep || l.running.load(Ordering::SeqCst));
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
            let (context, interrupted) = {
                let c = live.conv.lock().unwrap();
                (c.context_tokens(), needs_continue(&c.messages))
            };
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
        if first {
            let env = prompt::environment_block(&cwd, &s.notes_folder, &self.shell);
            context = if context.is_empty() { env } else { format!("{context}\n\n{env}") };
        }
        let user = Message::User(UserMessage { text, context, images, time: now_millis() });
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
                self.emit(Event::Activity { conv: id.into(), text: Some(format!("Loading {}...", model.name)) });
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
        let (system, messages, thinking) = self.request_parts(live);
        let chars = system.len() + self.tool_specs(live).iter().map(|t| t.to_string().len()).sum::<usize>() + messages_chars(&messages);
        let estimate = (chars as f64 / CHARS_PER_TOKEN) as u64;
        let reserve = (thinking.budget() as u64 + 2048).min(ctx / 3);
        estimate + reserve > ctx
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
        self.emit(Event::Activity { conv: id.into(), text: Some("Summarizing earlier messages to make room...".into()) });
        let (system, view, _) = self.request_parts(live);
        // The view starts with the previous summary when there is one, and maps conversation indexes after it.
        let offset = if conv.compaction.is_some() { 1 } else { 0 };
        let mut messages: Vec<Message> = if extend { view } else { view[..offset + (kept_from - start)].to_vec() };
        messages.push(Message::User(UserMessage { text: SUMMARY_PROMPT.into(), context: String::new(), images: Vec::new(), time: 0 }));
        let tools = self.tool_specs(live);
        let (thinking, thinking_budget) = side_thinking(ep, live);
        let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget, max_tokens: if ep.local { SUMMARY_MAX_TOKENS_LOCAL } else { SUMMARY_MAX_TOKENS_HOSTED } };
        let body = stream::payload(ep, &req);
        let reply = stream::send(&self.http, ep, &body, cancel, |_| {}).await?;
        if reply.stop == StopReason::Aborted {
            return Err(crate::util::Cancelled.into());
        }
        if reply.text.trim().is_empty() {
            bail!("The model could not summarize the conversation. {}", reply.error.unwrap_or_default());
        }
        let mut summary = reply.text.trim().to_string();
        let folder = self.settings().notes_folder;
        let vault = crate::notes::Vault::new(conv.cwd.join(&folder));
        match memory::save_task_summary(&vault, &conv.display_title(), &summary) {
            Ok(rel) => summary.push_str(&format!("\n\nThis summary is also saved in {folder}/{rel}.")),
            Err(err) => eprintln!("[notes] {err:#}"),
        }
        live.conv.lock().unwrap().set_compaction(summary, kept_from)?;
        // Notes attached to the summarized messages are gone from the request, so they can be attached again.
        live.read_notes.lock().unwrap().clear();
        self.emit(Event::NotesChanged { cwd: conv.cwd.clone() });
        if matches!(target, Target::Local(_)) {
            self.llama.set_slot_owner(None);
        }
        self.emit(Event::Compacted { conv: id.to_string(), kept_from });
        Ok(())
    }

    /// The system prompt and the messages as the model sees them, with any summary in place of older messages.
    fn request_parts(&self, live: &Live) -> (String, Vec<Message>, Thinking) {
        let s = self.settings();
        let c = live.conv.lock().unwrap();
        let system = prompt::system_prompt(&s.notes_folder, self.shell.tool_name());
        let messages = match &c.compaction {
            Some(comp) if comp.kept_from <= c.messages.len() => {
                let mut out = vec![Message::User(UserMessage {
                    text: format!("<summary>\n{}\n</summary>", comp.summary.trim()),
                    context: prompt::environment_block(&c.cwd, &s.notes_folder, &self.shell),
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
        let cwd = live.conv.lock().unwrap().cwd.clone();
        tools::specs(&self.shell, is_general(&cwd), self.settings().web_access)
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
        let target = Target::Local(model);
        let ep = self.endpoint(&target);
        let (system, mut messages, thinking) = self.request_parts(live);
        if shared {
            messages.clear();
        } else if exclude_last {
            messages.pop();
        }
        if messages.iter().any(|m| matches!(m, Message::User(u) if !u.images.is_empty())) {
            return Ok(());
        }
        let tools = self.tool_specs(live);
        let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget: None, max_tokens: self.max_tokens(&target) };
        let body = stream::payload(&ep, &req);
        let prefix = self.llama.shared_prefix(&body).await?;
        let name = if shared {
            self.llama.slot_file("shared", &LlamaServer::hash(&prefix))
        } else {
            self.llama.slot_file("chat", &conv.id)
        };
        if shared && self.llama.restore(&name).await {
            self.llama.set_slot_owner(Some(conv.id.clone()));
            return Ok(());
        }
        let _busy = self.llama.busy();
        self.llama.fill(&prefix, cancel).await?;
        self.llama.save(&name).await?;
        self.llama.set_slot_owner(Some(conv.id.clone()));
        Ok(())
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
            let related = live.run_notes.lock().unwrap().clone();
            let Some(task) = finished_task(&conv, &s.notes_folder, related) else { return };
            let mut facts = Vec::new();
            if (local || hosted) && s.remember_step && memory::worth_noting(&task) {
                host.emit(Event::Activity { conv: id.clone(), text: Some("Taking notes...".into()) });
                let cancel = CancellationToken::new();
                *live.note_cancel.lock().unwrap() = Some(cancel.clone());
                match host.ask(&live, memory::REMEMBER_PROMPT, REMEMBER_MAX_TOKENS, &cancel).await {
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
        messages.push(Message::User(UserMessage { text: question.into(), context: String::new(), images: Vec::new(), time: 0 }));
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
        for t in tokens {
            t.cancel();
        }
        self.llama.stop().await;
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
        Message::User(UserMessage { text: "u".repeat(n), context: String::new(), images: Vec::new(), time: 0 })
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
