//! Runs conversations: sends them to a local or hosted model, runs the tools it calls, and keeps the local
//! model's prompt cache and the project notes up to date.

pub mod browser;
pub mod conversation;
pub mod jobs;
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

use crate::i18n::{tr, trf};
use crate::llama::{LlamaServer, LocalModel, ServerStatus, SharedSettings};
use crate::store::{Approvals, HostedModel, PlanFirst, Settings, Thinking};
use crate::util::{is_cancelled, now_millis, thousands};

const MAX_OUTPUT_TOKENS: u32 = 16_384;

pub type ConvId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Always,
    Deny,
    /// The answer to the question a new project asks instead of an approval.
    Plan(PlanFirst),
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
    /// The stopped reply at the end of the conversation is being continued, and the next reply takes its place.
    Replacing { conv: ConvId },
    /// The model did not fit in free memory and loading from disk is off, so the window can offer to turn it on.
    DiskOffer { conv: ConvId },
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
    /// A task finished with the local model, so a task in another conversation can use it.
    ModelFree,
    /// The commands the model left running in the background, by job number.
    Jobs { conv: ConvId, jobs: Vec<(u32, String)> },
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
    /// Commands the model left running in the background.
    pub jobs: Vec<(u32, String)>,
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
    /// The entry in `Settings::model_files` for a model file the user added from another folder, which leaving out
    /// of the list does not delete.
    pub added: Option<String>,
    /// A model downloaded into the models folder, which Settings can delete. Variants are deleted in the model lab.
    pub deletable: bool,
    /// Model lab variants that run on this model's file and stop working without it.
    pub variants: usize,
    /// A local model that is not a variant, so it can learn from ratings.
    pub learnable: bool,
    /// The local model's architecture, since a draft model must share it, and whether its file holds layers that
    /// predict several words at once.
    pub arch: String,
    pub mtp: bool,
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
    /// Messages the user sent while a task ran, which the task reads after its current step.
    queued: Mutex<Vec<(String, Vec<conversation::Image>)>>,
    /// The running task updates the notes, so the note step after it is skipped.
    notes_update: AtomicBool,
    /// This conversation's hold on the local model, from the start of a task to the end of the steps after it.
    turn: Mutex<Option<Turn>>,
    /// Stops only the request in flight, so a message sent while the model thinks reaches it without ending the task.
    step_cancel: Mutex<Option<CancellationToken>>,
    /// When the request in flight last streamed thinking.
    last_thought: Mutex<Option<std::time::Instant>>,
    /// The request in flight was stopped because a file write grew past `stream::WRITE_PART_CHARS`.
    split: AtomicBool,
    /// Commands the model started in the background, which end when the conversation closes.
    jobs: Arc<jobs::Jobs>,
    /// The browser the model tests pages in, which closes when the task ends.
    browser: tools::BrowserSlot,
    /// The next run answers the conversation again instead of sending a message, after Retry took back a reply.
    retry: AtomicBool,
}

/// Why a file write was cut off partway, which decides what is saved and what the model is told.
#[derive(Clone, Copy, PartialEq)]
enum Cut {
    /// The user pressed Stop.
    Stopped,
    /// Scoobert stopped a write that grew past one part.
    Split,
    /// The reply ran out of room in the context.
    OutOfRoom,
}

/// A conversation's hold on the local model. Dropping it lets the next waiting task start.
struct Turn {
    holder: Arc<Mutex<Option<ConvId>>>,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

impl Drop for Turn {
    fn drop(&mut self) {
        *self.holder.lock().unwrap() = None;
    }
}

/// The local model held by a benchmark. Dropping it lets waiting tasks run.
pub struct ModelHold(#[allow(dead_code)] Turn);

/// Stands in `turn_holder` for a benchmark, and cannot be a conversation's id.
const BENCHMARK_HOLDER: &str = "\u{0}benchmark";

/// Another conversation's task that holds the local model.
#[derive(Debug, Clone)]
pub struct Elsewhere {
    pub title: String,
    pub cwd: PathBuf,
    pub file: PathBuf,
    /// The task is still running, rather than saving its cache and taking notes after it.
    pub running: bool,
    pub same_model: bool,
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
    /// Held by one task on the local model at a time. The server has one slot, so two tasks would take turns on it
    /// and read their whole conversation again at every turn.
    local_turn: Arc<tokio::sync::Mutex<()>>,
    /// The conversation that holds `local_turn`.
    turn_holder: Arc<Mutex<Option<ConvId>>>,
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

const SUMMARY_PROMPT: &str = "Context step. The conversation is too long for the model's context, so older messages will be replaced by your summary and only the most recent ones stay. Start with one line that begins with Title: and names the whole task in three to six words. Then write what you need to continue the work without the older ones, as short bullet points under these headings, in this order: ## Goal, ## Next, ## Open problems, ## Decisions (with reasons), ## Done (files changed and why, with exact paths). Keep the whole summary under 400 words. Reply in plain text without tools.";
/// Facts from the note step are three labeled lines of up to 160 characters.
const REMEMBER_MAX_TOKENS: u32 = 160;
const IMAGE_HELPER: &str = "You describe images for another AI model that cannot see them. Report only what the image shows.";
const HELPER_TOKENS: u32 = 800;
/// Room for one screenshot, the request, and the description.
const HELPER_CONTEXT: u32 = 8192;
/// Free memory left over when the image helper runs beside the conversation's model.
const HELPER_MARGIN: u64 = 1_000_000_000;
const TITLE_PROMPT: &str = "Title step. Reply with a short title for this conversation: 3 to 6 words that name the task, with no quotes and no period. Reply in plain text without tools.";
const TITLE_MAX_TOKENS: u32 = 24;
/// Summary length caps. A laptop CPU writes one to four tokens per second, so local summaries stay short.
const SUMMARY_MAX_TOKENS_LOCAL: u32 = 700;
const SUMMARY_MAX_TOKENS_HOSTED: u32 = 1500;
/// Share of the context the newest messages keep when older ones are summarized.
const KEEP_SHARE: f64 = 0.35;
/// Characters per token when estimating, set low so the estimate errs toward summarizing early.
const CHARS_PER_TOKEN: f64 = 3.2;
const MAX_RETRIES: u32 = 3;
/// Thinking streams a token every second or so, even on a CPU. A longer pause means the model is writing a tool call,
/// which the server sends only once it is complete, so stopping then would lose it.
const THOUGHT_GAP: std::time::Duration = std::time::Duration::from_secs(3);
/// How long a task runs between saves of its prompt cache. A save of a large model's cache writes about a gigabyte.
const SAVE_DURING_TASK: std::time::Duration = std::time::Duration::from_secs(300);
/// Thinking tokens a local model may spend on a side request before it must answer.
const SIDE_THINKING_TOKENS: u32 = 16;
/// Sent with the Continue button, after a crash, a close, or Stop left a task unfinished.
const RESUME: &str = "The system was closed or stopped before the last task finished. Continue that task from where it stopped. Files you already read in this conversation have not changed, so do not read them again.";
/// Added to RESUME when the last reply was stopped while it wrote a file.
const RESUME_WRITE: &str = "A file you were writing when it stopped was saved up to its last complete line, as its result above says.";
/// Added to RESUME when the last reply was stopped before it called a tool.
const RESUME_CUT: &str = "Your last reply above was cut off partway, so its reasoning and text end where it stopped. Pick up from there.";
/// Attached to the first message after the user pressed Stop, which otherwise only shows a reply cut short.
/// Instructions for the Update notes button. `{notes}` is the notes folder.
const NOTES_UPDATE: &str = "<notes_update>Update the project notes from this whole conversation, then stop. Read {notes}/Features.md first if it exists. Keep it in this form: a # Features heading, a ## Done section, and a ## To do section, with one line per feature written as - **Short name**: one sentence about what it does or what it still needs. Add every feature this conversation finished to Done, and every feature it planned, started, or left unfinished to To do. Move an item from To do to Done when it is finished instead of listing it twice, and keep the items from earlier conversations. Link a feature to its topic page with [[Topic]] when one exists. Then add each decision, convention, or known problem a later conversation needs to the topic page it belongs to, or to Decisions.md, Conventions.md, or Problems.md in {notes}/. Use edit for small changes and write for new files, change only files in {notes}/, and reply with a short list of what you changed.</notes_update>";
/// The planning half of the Plan first option, which keeps the plan in notes that later conversations and summaries
/// can read. `{notes}` is the notes folder.
const PLAN: &str = "Before you write any code, plan this project in its notes. Think through the features the request needs, including the ones it implies but does not name. Write {notes}/Features.md with a # Features heading, an empty ## Done section, and a ## To do section with one line per feature, written as - **Short name**: one sentence about what it does, linked to its page as [[Short name]]. For each feature that needs more than a few lines of code, write {notes}/Short name.md with what it does, the files it will add or change, how it connects to other features through [[links]], and how to check that it works. Put decisions that affect the whole project in {notes}/Decisions.md, with the reason for each. Keep each page short, since you will read them again later instead of the whole conversation.";
const PLAN_DISCUSS: &str = "Then stop without writing code, and reply with a short summary of the plan and the questions the user should answer before you build.";
const PLAN_BUILD: &str = "Then build the project from the plan, one feature at a time, in an order where each step can be checked. After each feature works, move its line from To do to Done in Features.md, and correct its page where the build differs from the plan.";

/// Instructions to plan the project in its notes first, then stop for the user or build from the plan.
fn plan_instructions(folder: &str, then_build: bool) -> String {
    let next = if then_build { PLAN_BUILD } else { PLAN_DISCUSS };
    format!("<plan_first>{PLAN} {next}</plan_first>").replace("{notes}", folder)
}
/// Lines from the end of a file that a cut-off write's result shows.
const FILE_END_LINES: usize = 6;

/// The last lines of a file with their line numbers, so the model sees exactly where a cut-off write stopped.
fn file_end(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(n);
    let shown: Vec<String> = lines[start..].iter().enumerate().map(|(i, l)| format!("{}: {}", start + i + 1, crate::util::clip(l, 160))).collect();
    format!("\nThe file now has {} lines and ends with:\n{}", lines.len(), shown.join("\n"))
}

#[cfg(test)]
mod file_end_tests {
    #[test]
    fn the_end_of_a_file_is_numbered() {
        let text: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        assert_eq!(super::file_end(&text, 2), "\nThe file now has 10 lines and ends with:\n9: line 9\n10: line 10");
    }
}

/// Context for a message the user sent while a task ran.
pub const QUEUED: &str = "<queued>The user sent this while you were working. Take it into account, and carry on with the task unless it asks you to change course.</queued>";
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
            local_turn: Arc::new(tokio::sync::Mutex::new(())),
            turn_holder: Arc::new(Mutex::new(None)),
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

    /// Whether a task, or the steps after one, is using the local model, so the model lab must not unload it.
    pub fn model_in_use(&self) -> bool {
        self.turn_holder.lock().unwrap().is_some() || self.llama.in_use()
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
            (Err(err), Some(_)) => Some(trf("{error}. The key works until Scoobert closes.", &[("error", &format!("{err:#}"))])),
            (Err(err), None) => Some(format!("{err:#}")),
            (Ok(()), _) => None,
        }
    }

    pub fn models(&self) -> Vec<ModelOption> {
        let s = self.settings();
        let local = self.llama.models();
        let models_dir = self.llama.models_dir();
        let variants_of = |m: &LocalModel| local.iter().filter(|v| v.variant.is_some() && v.path == m.path).count();
        let mut out: Vec<ModelOption> = local
            .iter()
            .cloned()
            .map(|m| {
                let ctx = self.llama.context_for(&m);
                let (arch, mtp) = crate::llama::gguf::kind(&m.path);
                ModelOption {
                    arch,
                    mtp,
                    deletable: m.variant.is_none() && m.path.starts_with(&models_dir),
                    learnable: m.variant.is_none(),
                    variants: if m.variant.is_none() { variants_of(&m) } else { 0 },
                    label: m.name.clone(),
                    provider: String::new(),
                    usable: true,
                    context: ctx,
                    vision: m.mmproj.is_some(),
                    // A variant made in the model lab carries the name it was given, and its family says what it is.
                    reasoning: REASONING_LOCAL.is_match(&m.name) || REASONING_LOCAL.is_match(&m.family),
                    size: m.size,
                    memory_needed: self.llama.memory_needed(&m, ctx),
                    added: s.model_files.iter().find(|f| std::path::Path::new(f) == m.path).cloned(),
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
                added: None,
                deletable: false,
                variants: 0,
                learnable: false,
                arch: String::new(),
                mtp: false,
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
                reasoning: REASONING_LOCAL.is_match(&m.name) || REASONING_LOCAL.is_match(&m.family),
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
            jobs: live.jobs.running(),
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
                notice = Some(trf("{model} is not available, so this conversation now uses {fallback}.", &[("model", &conv.model), ("fallback", &fallback)]));
            }
            let _ = conv.set_model(&fallback);
        }
        let id = conv.id.clone();
        let (events, conv_id) = (self.events.clone(), id.clone());
        let notify: jobs::Notify = Arc::new(move |jobs| {
            let _ = events.send(Event::Jobs { conv: conv_id.clone(), jobs });
        });
        let live = Arc::new(Live {
            jobs: Arc::new(jobs::Jobs::new(notify)),
            browser: tools::BrowserSlot::default(),
            retry: AtomicBool::new(false),
            conv: Mutex::new(conv),
            running: AtomicBool::new(false),
            run_cancel: Mutex::new(None),
            cache_cancel: Mutex::new(None),
            note_cancel: Mutex::new(None),
            background: tokio::sync::Mutex::new(()),
            allowed: Mutex::new(HashSet::new()),
            read_notes: Mutex::new(BTreeSet::new()),
            run_notes: Mutex::new(Vec::new()),
            queued: Mutex::new(Vec::new()),
            notes_update: AtomicBool::new(false),
            turn: Mutex::new(None),
            step_cancel: Mutex::new(None),
            last_thought: Mutex::new(None),
            split: AtomicBool::new(false),
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
        // Reading this conversation into the slot, or loading its model, would undo the work of the task that holds it.
        if self.turn_holder.lock().unwrap().as_deref().is_some_and(|holder| holder != id) {
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
        Conversation::delete(file)?;
        // The file name ends with the conversation's id, which names its saved prompts.
        if let Some(id) = file.file_stem().and_then(|s| s.to_str()).and_then(|s| s.rsplit('-').next()) {
            crate::llama::forget_chat(id);
        }
        Ok(())
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
    /// Holds a message sent while a task runs. The task reads it after its current step.
    pub fn queue(&self, id: &str, text: String, images: Vec<conversation::Image>) -> anyhow::Result<()> {
        self.live(id)?.queued.lock().unwrap().push((text, images));
        Ok(())
    }

    /// Delivers the held messages now when the model is thinking. The request stops, its thinking so far stays in
    /// the conversation, and the next request adds the messages. Returns whether it did.
    pub fn deliver_now(&self, id: &str) -> bool {
        let Ok(live) = self.live(id) else { return false };
        if !live.last_thought.lock().unwrap().is_some_and(|t| t.elapsed() < THOUGHT_GAP) {
            return false;
        }
        match live.step_cancel.lock().unwrap().as_ref() {
            Some(step) => {
                step.cancel();
                true
            }
            None => false,
        }
    }

    /// Drops the held message at `index`, unless the task has read the held messages already. Returns whether it did.
    pub fn unqueue(&self, id: &str, index: usize) -> bool {
        let Ok(live) = self.live(id) else { return false };
        let mut queued = live.queued.lock().unwrap();
        if index < queued.len() {
            queued.remove(index);
            true
        } else {
            false
        }
    }

    /// Takes back the messages a task ended without reading.
    pub fn take_queued(&self, id: &str) -> Vec<(String, Vec<conversation::Image>)> {
        self.live(id).map(|l| std::mem::take(&mut *l.queued.lock().unwrap())).unwrap_or_default()
    }

    /// Adds the messages the user sent during the task to the conversation, so the next request reads them.
    fn deliver_queued(&self, live: &Live, id: &str) -> anyhow::Result<()> {
        let queued = std::mem::take(&mut *live.queued.lock().unwrap());
        if queued.is_empty() {
            return Ok(());
        }
        let text = queued.iter().map(|(t, _)| t.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n\n");
        let images: Vec<conversation::Image> = queued.into_iter().flat_map(|(_, i)| i).collect();
        // Describing images would stop the task's model, so a model that cannot see them hears that they came.
        let model = live.conv.lock().unwrap().model.clone();
        let described = !images.is_empty() && !self.sees_images(&model);
        let context = if described {
            format!("{QUEUED}\n\nThe user also attached {} images, which you cannot see. Ask the user to send them again after this task, when another model can describe them.", images.len())
        } else {
            QUEUED.into()
        };
        let context = prompt::system_block(&context);
        let user = Message::User(UserMessage { text, context, images, described, time: now_millis(), ..Default::default() });
        live.conv.lock().unwrap().push(user.clone())?;
        self.emit(Event::Message { conv: id.to_string(), message: user });
        Ok(())
    }

    pub fn prompt(self: &Arc<Self>, id: &str, text: String, images: Vec<conversation::Image>, resume: bool) -> anyhow::Result<()> {
        self.start(id, text, images, resume, None, false)
    }

    /// Takes back the last reply and asks the model again, keeping any tool results before it. With `bad`, the reply
    /// is saved as a bad example for learning first.
    pub fn retry(self: &Arc<Self>, id: &str, bad: bool) -> anyhow::Result<()> {
        let live = self.live(id)?;
        if live.running.load(Ordering::SeqCst) {
            bail!("Scoobert is still working on the last message.");
        }
        {
            let mut c = live.conv.lock().unwrap();
            let Some(Message::Assistant(_)) = c.messages.last() else { bail!(tr("There is no reply to retry.")) };
            let at = c.messages.len() - 1;
            if c.compaction.as_ref().is_some_and(|k| k.kept_from > at) {
                bail!(tr("That reply is part of a summary, so it cannot be retried."));
            }
            if bad && let Some(rating) = last_rating(&c, false) {
                crate::llama::learn::record(&c.model, rating)?;
            }
            c.rewind(at)?;
        }
        self.emit(Event::Replacing { conv: id.into() });
        live.retry.store(true, Ordering::SeqCst);
        let started = self.start(id, String::new(), Vec::new(), false, None, false);
        if started.is_err() {
            live.retry.store(false, Ordering::SeqCst);
        }
        started
    }

    /// Saves the last reply as a good example for learning.
    pub fn rate_good(&self, id: &str) -> anyhow::Result<()> {
        let live = self.live(id)?;
        let c = live.conv.lock().unwrap();
        let rating = last_rating(&c, true).context(tr("There is no reply to rate."))?;
        crate::llama::learn::record(&c.model, rating)
    }

    /// Sends the first message of a conversation with instructions to plan the project in its notes first, and then
    /// either stop for the user or build from the plan.
    pub fn plan(self: &Arc<Self>, id: &str, text: String, images: Vec<conversation::Image>, then_build: bool) -> anyhow::Result<()> {
        let instructions = plan_instructions(&self.settings().notes_folder, then_build);
        // A plan that stops for the user wrote the notes already, so the note step after it has nothing to add.
        self.start(id, text, images, false, Some(instructions), !then_build)
    }

    /// The task in another conversation that holds the local model, which a task in `id` would wait for or, on
    /// another model, would have to stop first. `None` when this conversation uses a hosted model.
    pub fn local_task_elsewhere(&self, id: &str) -> Option<Elsewhere> {
        let holder = self.turn_holder.lock().unwrap().clone()?;
        if holder == id {
            return None;
        }
        let (mine, other) = {
            let convs = self.convs.lock().unwrap();
            (convs.get(id).cloned()?, convs.get(&holder).cloned()?)
        };
        let model = mine.conv.lock().unwrap().model.clone();
        if self.settings().hosted_models.iter().any(|h| h.reference() == model) {
            return None;
        }
        let c = other.conv.lock().unwrap();
        Some(Elsewhere {
            title: c.display_title(),
            cwd: c.cwd.clone(),
            file: c.file.clone(),
            running: other.running.load(Ordering::SeqCst),
            same_model: c.model == model,
        })
    }

    /// Waits for `wait`, with the activity line saying what for: the model being loaded and the time so far, or
    /// `text`. It updates once a second, since a first load from a slow disk can take minutes.
    async fn wait_showing<T>(&self, id: &str, text: impl Fn() -> String, wait: impl std::future::Future<Output = T>, cancel: &CancellationToken) -> anyhow::Result<T> {
        tokio::pin!(wait);
        loop {
            let line = match self.llama.loading_for() {
                Some((model, took, first)) => loading_line(&model, took.as_secs(), first),
                None => text(),
            };
            self.emit(Event::Activity { conv: id.into(), text: Some(line) });
            tokio::select! {
                out = &mut wait => return Ok(out),
                _ = cancel.cancelled() => return Err(crate::util::Cancelled.into()),
                _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
            }
        }
    }

    /// Waits until no other conversation's task holds the local model, then holds it for this task and the steps
    /// after it.
    async fn take_turn(&self, live: &Live, id: &str, cancel: &CancellationToken) -> anyhow::Result<()> {
        if live.turn.lock().unwrap().is_some() {
            return Ok(());
        }
        let guard = match self.local_turn.clone().try_lock_owned() {
            Ok(guard) => guard,
            Err(_) => {
                // The holder is another conversation's task, or this conversation's steps after its last task.
                let text = || match self.local_task_elsewhere(id) {
                    Some(other) => trf("Waiting for “{title}” to finish", &[("title", &other.title)]),
                    None if self.turn_holder.lock().unwrap().as_deref() == Some(BENCHMARK_HOLDER) => tr("Waiting for the benchmark to finish").into(),
                    None => tr("Getting ready...").into(),
                };
                self.wait_showing(id, text, self.local_turn.clone().lock_owned(), cancel).await?
            }
        };
        *self.turn_holder.lock().unwrap() = Some(id.to_string());
        *live.turn.lock().unwrap() = Some(Turn { holder: self.turn_holder.clone(), _guard: guard });
        // Reads that only prepare other conversations would take turns with this task on the slot.
        self.stop_baking();
        for (other, l) in self.convs.lock().unwrap().iter() {
            if other != id
                && let Some(t) = l.cache_cancel.lock().unwrap().as_ref()
            {
                t.cancel();
            }
        }
        Ok(())
    }

    /// Waits until no conversation's task holds the local model, then holds it for a benchmark until the hold drops.
    pub async fn hold_model(&self) -> ModelHold {
        let guard = self.local_turn.clone().lock_owned().await;
        *self.turn_holder.lock().unwrap() = Some(BENCHMARK_HOLDER.to_string());
        self.stop_baking();
        for l in self.convs.lock().unwrap().values() {
            if let Some(t) = l.cache_cancel.lock().unwrap().as_ref() {
                t.cancel();
            }
        }
        ModelHold(Turn { holder: self.turn_holder.clone(), _guard: guard })
    }

    /// Lets the next waiting task use the local model.
    fn release_turn(&self, live: &Live) {
        if live.turn.lock().unwrap().take().is_some() {
            self.emit(Event::ModelFree);
        }
    }

    /// Asks the model to update the project's notes from the whole conversation: the features it finished and the
    /// ones still to do on the Features page, and other lasting facts on their topic pages.
    pub fn update_notes(self: &Arc<Self>, id: &str) -> anyhow::Result<()> {
        let folder = self.settings().notes_folder;
        self.start(id, tr("Update the notes from this conversation.").into(), Vec::new(), false, Some(NOTES_UPDATE.replace("{notes}", &folder)), true)
    }

    /// Starts a task. `instructions` go with the message for the model only. `notes_written` skips the note step
    /// after the task, because the task itself wrote the notes.
    fn start(
        self: &Arc<Self>,
        id: &str,
        text: String,
        images: Vec<conversation::Image>,
        resume: bool,
        instructions: Option<String>,
        notes_written: bool,
    ) -> anyhow::Result<()> {
        let live = self.live(id)?;
        // Loading another model would stop the task that runs on the loaded one.
        if let Some(other) = self.local_task_elsewhere(id)
            && other.running
            && !other.same_model
        {
            bail!("{} {}", trf("Scoobert is working in “{title}”.", &[("title", &other.title)]), tr("That task uses another model. Stop it before you send a message here."));
        }
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
        live.notes_update.store(notes_written, Ordering::SeqCst);
        self.rt.spawn(async move {
            let result = host.run(&live, &conv_id, text, images, resume, instructions, &cancel).await;
            // The browser holds hundreds of megabytes, so it closes with the task, and the next task opens a new one.
            drop(live.browser.lock().await.take());
            if let Err(err) = &result
                && !is_cancelled(err)
            {
                host.emit(Event::Error { conv: Some(conv_id.clone()), message: format!("{err:#}") });
                if err.chain().any(|e| e.downcast_ref::<crate::llama::NotEnoughMemory>().is_some_and(|m| !m.from_disk)) {
                    host.emit(Event::DiskOffer { conv: conv_id.clone() });
                }
            }
            host.emit(Event::Activity { conv: conv_id.clone(), text: None });
            live.running.store(false, Ordering::SeqCst);
            *live.run_cancel.lock().unwrap() = None;
            let context = host.context_used(&live);
            let interrupted = needs_continue(&live.conv.lock().unwrap().messages);
            host.emit(Event::Settled { conv: conv_id.clone(), context, interrupted });
            if result.is_ok() {
                host.after_run(live, conv_id);
            } else {
                host.release_turn(&live);
            }
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        self: &Arc<Self>,
        live: &Arc<Live>,
        id: &str,
        text: String,
        images: Vec<conversation::Image>,
        resume: bool,
        instructions: Option<String>,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        let (model_name, last, summarized) = {
            let c = live.conv.lock().unwrap();
            // A reply that is the first one a summary kept cannot be replaced without dropping the summary.
            let summarized = c.compaction.as_ref().is_some_and(|k| k.kept_from + 1 >= c.messages.len());
            (c.model.clone(), c.messages.last().cloned(), summarized)
        };
        // A message whose run failed before the model answered it is answered now rather than sent again, and so is
        // the conversation after Retry took back its last reply.
        let retry = live.retry.swap(false, Ordering::SeqCst);
        let unanswered = retry || resume && matches!(last, Some(Message::User(_)));
        let target = self.target(&model_name)?;
        let ep = self.endpoint(&target);
        // A local model can start its reply with given text, so Continue picks up a stopped reply where it stopped.
        let try_continue = !retry && resume && !summarized && matches!(target, Target::Local(_)) && matches!(&last, Some(Message::Assistant(a)) if continuable(a));
        let mut message = (!unanswered).then_some((text, images, instructions));
        // A model that cannot see images gets the user's images described by one that can, before it loads.
        let mut described = false;
        if let Some((_, images, instructions)) = message.as_mut()
            && !images.is_empty()
            && !self.sees_images(&model_name)
        {
            if matches!(target, Target::Local(_)) {
                self.take_turn(live, id, cancel).await?;
            }
            let mut notes = Vec::new();
            for (i, image) in images.iter().enumerate() {
                let (helper, text) = self.describe_image(live, id, image, "image", "", cancel).await?;
                notes.push(format!("<image_description>Image {} that the user attached, described by {helper}, because you cannot see images:\n{text}</image_description>", i + 1));
            }
            let notes = notes.join("\n\n");
            *instructions = Some(match instructions.take() {
                Some(i) => format!("{i}\n\n{notes}"),
                None => notes,
            });
            described = true;
        }
        if !try_continue && let Some((text, images, instructions)) = message.take() {
            self.add_message(live, id, text, images, described, resume, instructions, last.as_ref(), cancel).await?;
        }
        if matches!(target, Target::Local(_)) {
            self.take_turn(live, id, cancel).await?;
        }
        let _busy = matches!(target, Target::Local(_)).then(|| self.llama.busy());
        let mut continuing = false;
        if let Target::Local(model) = &target {
            let _guard = match live.background.try_lock() {
                Ok(guard) => guard,
                Err(_) => self.wait_showing(id, || tr("Getting ready...").into(), live.background.lock(), cancel).await?,
            };
            if self.llama.loaded_model().as_deref() != Some(&model.name) {
                // Loading takes from seconds to minutes and reports no progress, so the time so far is shown.
                let (events, conv, name, stop) = (self.events.clone(), id.to_string(), model.name.clone(), CancellationToken::new());
                let first = !self.llama.loaded_before(&model.name);
                let ticker = stop.clone();
                self.rt.spawn(async move {
                    let start = std::time::Instant::now();
                    loop {
                        let text = loading_line(&name, start.elapsed().as_secs(), first);
                        let _ = events.send(Event::Activity { conv: conv.clone(), text: Some(text) });
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
            continuing = try_continue && self.can_continue(live, id, &target).await;
            if !continuing && let Some((text, images, instructions)) = message.take() {
                self.add_message(live, id, text, images, described, resume, instructions, last.as_ref(), cancel).await?;
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
        self.steps(live, id, &target, &ep, continuing, cancel).await
    }

    /// Adds the user's message, with the notes that match it and what the model needs to know about the last run.
    #[allow(clippy::too_many_arguments)]
    async fn add_message(
        &self,
        live: &Arc<Live>,
        id: &str,
        text: String,
        images: Vec<conversation::Image>,
        described: bool,
        resume: bool,
        instructions: Option<String>,
        last: Option<&Message>,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        let s = self.settings();
        let interrupted = was_stopped(&live.conv.lock().unwrap().messages);
        self.save_stopped_writes(live, id, cancel).await?;
        let (cwd, first) = {
            let mut c = live.conv.lock().unwrap();
            answer_dangling_calls(&mut c)?;
            (c.cwd.clone(), !c.has_user_message())
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
            let note = if resume { resume_note(last) } else { INTERRUPTED.to_string() };
            context = [note.as_str(), &context].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join("\n\n");
        }
        if let Some(instructions) = instructions {
            context = if context.is_empty() { instructions } else { format!("{instructions}\n\n{context}") };
        }
        let mut environment = String::new();
        if first {
            environment = prompt::environment_block(&cwd, &s.notes_folder, &self.shell);
            let session = prompt::session_block(&cwd, &s.notes_folder, s.language());
            context = if context.is_empty() { session } else { format!("{context}\n\n{session}") };
        } else {
            // A language picked after the conversation started is announced on the next message, which keeps the
            // cached start of the conversation as it is.
            let language = s.language();
            let stated = prompt::stated_language(&live.conv.lock().unwrap());
            if stated.as_deref().unwrap_or("English") != language.english {
                let note = format!("<language>{}</language>", prompt::language_line(language));
                context = if context.is_empty() { note } else { format!("{note}\n\n{context}") };
            }
        }
        // Everything Scoobert adds after the user's words is framed as its own, so the model does not answer it as the
        // user's request.
        let context = if context.trim().is_empty() { context } else { prompt::system_block(&context) };
        // Surrounding whitespace would change how the start of the text tokenizes and miss the saved cache.
        let user = Message::User(UserMessage { text: text.trim().to_string(), environment, context, images, described, time: now_millis() });
        live.conv.lock().unwrap().push(user.clone())?;
        self.emit(Event::Message { conv: id.to_string(), message: user });
        Ok(())
    }

    /// Whether continuing the stopped reply at the end reads about as little as a new message would. A prompt saved
    /// before Scoobert closed may hold that reply already closed, which a continued reply cannot start from.
    async fn can_continue(&self, live: &Live, id: &str, target: &Target) -> bool {
        if self.llama.slot_owner().as_deref() == Some(id) {
            return true;
        }
        let (Ok(Some(before)), Ok(Some(with))) = (self.history_tokens(live, target, true).await, self.history_tokens(live, target, false).await) else {
            return false;
        };
        let unread = |tokens: &[i32]| tokens.len().saturating_sub(self.llama.longest_saved(tokens));
        // One batch of slack, since a continued reply reads its own text again.
        unread(&before) <= unread(&with) + 512
    }

    /// Requests the model's replies and runs their tool calls until the task ends. `continuing` sends the stopped
    /// reply at the end of the conversation for the model to carry on, and the first reply takes its place.
    async fn steps(self: &Arc<Self>, live: &Arc<Live>, id: &str, target: &Target, ep: &Endpoint, mut continuing: bool, cancel: &CancellationToken) -> anyhow::Result<()> {
        let mut compacted_for_error = false;
        // One summary per state of the conversation; summarizing again without new messages cannot help.
        let mut compacted_at = None;
        loop {
            self.deliver_queued(live, id)?;
            // A message the user sent meanwhile follows the stopped reply, so the model reads that reply as it is.
            if continuing && !matches!(live.conv.lock().unwrap().messages.last(), Some(Message::Assistant(a)) if continuable(a)) {
                continuing = false;
            }
            let len = live.conv.lock().unwrap().messages.len();
            if compacted_at != Some(len) && self.too_long(live, &target) {
                self.compact(live, id, &target, &ep, true, cancel).await?;
                compacted_at = Some(len);
            }
            // A read after a summary goes in saved steps, so closing Scoobert partway does not start it over. A long
            // task saves every few minutes for the same reason.
            if ep.local && (self.llama.slot_owner().as_deref() != Some(id) || self.llama.since_save() > SAVE_DURING_TASK)
                && let Err(err) = self.persist(live, false, false, cancel).await
            {
                if is_cancelled(&err) {
                    return Err(err);
                }
                eprintln!("[cache] {err:#}");
            }
            self.emit(Event::Activity { conv: id.into(), text: Some(tr("Reading...").into()) });
            let (system, messages, thinking) = self.request_parts(live);
            let tools = self.tool_specs(live);
            // The reply stops early enough that the summary after it can still be asked with the whole conversation,
            // which a local model has cached.
            let room = self.context_window(target).saturating_sub(self.prompt_tokens(live) + summary_room(ep.local));
            let max_tokens = self.max_tokens(target).min(room.max(1024) as u32);
            let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget: None, max_tokens };
            let mut body = stream::payload(&ep, &req);
            if ep.local {
                body["return_progress"] = true.into();
            }
            let step = cancel.child_token();
            *live.step_cancel.lock().unwrap() = Some(step.clone());
            let continued = std::mem::take(&mut continuing);
            if continued {
                self.emit(Event::Replacing { conv: id.into() });
            }
            let result = self.send_with_retries(id, &target, &ep, &body, &step).await;
            *live.step_cancel.lock().unwrap() = None;
            *live.last_thought.lock().unwrap() = None;
            // A continued reply starts with everything the stopped one held, so it takes that reply's place. Without
            // anything new, the stopped reply stays and the window shows it again.
            let replaces = continued && result.as_ref().is_ok_and(|r| r.stop != StopReason::Error && !(r.thinking.is_empty() && r.text.is_empty() && r.tool_calls.is_empty()));
            if continued && !replaces {
                let stopped = live.conv.lock().unwrap().messages.last().cloned();
                if let Some(message) = stopped {
                    self.emit(Event::Message { conv: id.to_string(), message });
                }
            }
            // A write that grew past one part was stopped. Its complete lines are saved, and the model writes the rest
            // in the next part.
            if live.split.swap(false, Ordering::SeqCst) && !cancel.is_cancelled() {
                if let Ok(mut cut) = result
                    && !cut.tool_calls.is_empty()
                {
                    cut.resend_thinking = false;
                    let calls = cut.tool_calls.clone();
                    self.add_reply(live, id, cut, replaces)?;
                    self.save_cut_writes(live, id, calls, Cut::Split, cancel).await?;
                }
                continue;
            }
            // The user sent a message while the model thought. The thinking so far stays, and the next request
            // adds the message after it.
            if step.is_cancelled() && !cancel.is_cancelled() {
                if let Ok(mut thought) = result
                    && (!thought.thinking.is_empty() || replaces)
                {
                    thought.tool_calls.clear();
                    thought.resend_thinking = true;
                    self.add_reply(live, id, thought, replaces)?;
                }
                continue;
            }
            let mut reply = match result {
                Ok(r) => r,
                Err(err) if is_cancelled(&err) => AssistantMessage { stop: StopReason::Aborted, model: ep.model.clone(), time: now_millis(), ..Default::default() },
                Err(err) if !compacted_for_error && compacted_at != Some(len) && context_overflow(&format!("{err:#}")) => {
                    // The estimate was short; summarize now and send the request again.
                    compacted_for_error = true;
                    self.compact(live, id, &target, &ep, false, cancel).await?;
                    compacted_at = Some(len);
                    continuing = continued;
                    continue;
                }
                Err(err) => AssistantMessage { stop: StopReason::Error, error: Some(format!("{err:#}")), model: ep.model.clone(), time: now_millis(), ..Default::default() },
            };
            if reply.stop == StopReason::Error && reply.error.is_none() {
                reply.error = Some("The reply ended with an error.".into());
            }
            // The next turn builds on a stopped reply's reasoning instead of repeating it.
            reply.resend_thinking = reply.stop == StopReason::Aborted;
            // A continued reply stopped before it added anything leaves the stopped reply as it was.
            if continued && !replaces && reply.stop == StopReason::Aborted {
                return Ok(());
            }
            let calls = reply.tool_calls.clone();
            let stop = reply.stop;
            let error = reply.error.clone();
            self.add_reply(live, id, reply, replaces)?;
            if let Some(err) = error {
                self.emit(Event::Error { conv: Some(id.to_string()), message: err });
                return Ok(());
            }
            // The context filled while the model wrote a file. The complete lines are saved, and the next step
            // summarizes the conversation to make room before the model continues the file.
            if stop == StopReason::Length && !calls.is_empty() && !cancel.is_cancelled() {
                self.save_cut_writes(live, id, calls, Cut::OutOfRoom, cancel).await?;
                continue;
            }
            if stop != StopReason::ToolUse || calls.is_empty() || cancel.is_cancelled() {
                // A message sent during the final reply gets its own reply before the task ends.
                if stop == StopReason::Stop && !cancel.is_cancelled() && !live.queued.lock().unwrap().is_empty() {
                    continue;
                }
                return Ok(());
            }
            for call in calls {
                // A new project moves the conversation, so each call reads the current folder.
                let cwd = live.conv.lock().unwrap().cwd.clone();
                let result = if cancel.is_cancelled() {
                    ToolResult { call_id: call.id.clone(), name: call.name.clone(), output: "The user stopped the task before this ran.".into(), is_error: true, diff: None, time: now_millis(), ..Default::default() }
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

    /// Adds a reply to the conversation and the window. `replaces` puts it in place of the stopped reply it continues.
    fn add_reply(&self, live: &Live, id: &str, reply: AssistantMessage, replaces: bool) -> anyhow::Result<()> {
        let message = Message::Assistant(reply);
        {
            let mut c = live.conv.lock().unwrap();
            if replaces {
                let at = c.messages.len().saturating_sub(1);
                c.rewind(at)?;
            }
            c.push(message.clone())?;
        }
        self.emit(Event::Message { conv: id.to_string(), message });
        Ok(())
    }

    /// Saves the complete lines of file writes that Stop cut off, which the reply before this run left.
    async fn save_stopped_writes(&self, live: &Live, id: &str, cancel: &CancellationToken) -> anyhow::Result<()> {
        let calls = match live.conv.lock().unwrap().messages.last() {
            Some(Message::Assistant(a)) if a.stop == StopReason::Aborted && !a.tool_calls.is_empty() => a.tool_calls.clone(),
            _ => return Ok(()),
        };
        self.save_cut_writes(live, id, calls, Cut::Stopped, cancel).await
    }

    /// Saves the complete lines of file writes cut off partway, through the usual approval, and tells the model where
    /// to continue. A cut rewrite of an existing file goes to its draft, which the model finishes and moves into
    /// place. The tool lists mode before content, so a write cut without it came from a conversation that only has
    /// append, where leaving it out replaces the file.
    async fn save_cut_writes(&self, live: &Live, id: &str, calls: Vec<ToolCall>, cut: Cut, cancel: &CancellationToken) -> anyhow::Result<()> {
        let cwd = live.conv.lock().unwrap().cwd.clone();
        let notes = cwd.join(&self.settings().notes_folder);
        for call in calls {
            let append = tools::appends(&call);
            let exists = tools::target_path(&cwd, &call, Some(&notes)).is_some_and(|p| p.exists());
            let lines = call.arg("content").lines().count();
            let path = call.arg("path").to_string();
            // A cut rewrite of an existing file goes to a draft beside it, so the file keeps working, the wipe guard
            // has nothing to refuse, and the lines written are kept.
            let draft = (!append && exists).then(|| tools::draft_path(&path));
            let mut saved = call.clone();
            if let (Some(draft), Some(args)) = (&draft, saved.arguments.as_object_mut()) {
                args.insert("path".into(), draft.clone().into());
                args.insert("mode".into(), "create".into());
                args.remove("append");
                if let Some(old) = tools::target_path(&cwd, &saved, Some(&notes)) {
                    let _ = tokio::fs::remove_file(old).await;
                }
            }
            let mut result = self.run_tool(live, id, &cwd, &saved, cancel).await;
            if !result.is_error {
                let why = match cut {
                    Cut::Stopped => format!("The reply was stopped while writing this file, so only the first {lines} lines of this call were saved."),
                    Cut::Split => format!("One write holds about 200 lines, so the system stopped this one and saved the first {lines} lines of this call."),
                    Cut::OutOfRoom => format!("The reply ran out of room in the context while writing this file, so only the first {lines} lines of this call were saved."),
                };
                let next = match &draft {
                    Some(draft) => format!(
                        "{path} is unchanged, and the lines went to the draft {draft}. Add the rest of the new version to {draft} with write and mode set to append, without writing the saved lines again, then move {draft} to {path}."
                    ),
                    None => "Continue from where the file now ends with write and mode set to append, and do not write the saved lines again.".into(),
                };
                result.output = format!("{} {why} {next}", result.output);
                // The conversation later shows this call without its text, so the end of the file is the only record
                // of where to continue. The outline above leaves out everything inside a class.
                if let Some(file) = tools::target_path(&cwd, &saved, Some(&notes))
                    && let Ok(text) = tokio::fs::read_to_string(&file).await
                {
                    result.output.push_str(&file_end(&text, FILE_END_LINES));
                }
            }
            let message = Message::Tool(result);
            live.conv.lock().unwrap().push(message.clone())?;
            self.emit(Event::Message { conv: id.to_string(), message });
        }
        Ok(())
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
            let live = self.live(id).ok();
            let result = stream::send(&self.http, ep, body, cancel, |delta| {
                if let Some(live) = &live {
                    match &delta {
                        Delta::Thinking(_) => *live.last_thought.lock().unwrap() = Some(std::time::Instant::now()),
                        Delta::Text(_) | Delta::ToolCall(_) | Delta::ToolInput(_) => *live.last_thought.lock().unwrap() = None,
                        // The write so far is saved and the model continues it in another part. A write that has not
                        // named its file yet runs on, since nothing could be saved.
                        Delta::LongWrite { path: Some(_) } => {
                            if let Some(step) = live.step_cancel.lock().unwrap().as_ref() {
                                live.split.store(true, Ordering::SeqCst);
                                step.cancel();
                            }
                        }
                        _ => {}
                    }
                }
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
            self.emit(Event::Activity { conv: id.into(), text: Some(trf("The request failed. Trying again in {secs} seconds...", &[("secs", &wait.as_secs())])) });
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
        // A write the context cuts off keeps its complete lines, so the reserve only covers the thinking, a short
        // reply, and the summary request after them, and the summary waits as long as it can.
        let reserve = (thinking.budget() as u64 + 2048).min(ctx / 3) + summary_room(matches!(target, Target::Local(_)));
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
    /// It is false after the server rejected the conversation as too long, when only the older part fits. A failed
    /// attempt is tried again with only the older part, so a long task does not stop for the user.
    async fn compact(self: &Arc<Self>, live: &Arc<Live>, id: &str, target: &Target, ep: &Endpoint, extend: bool, cancel: &CancellationToken) -> anyhow::Result<()> {
        // A tool result can fill the context after the reply, leaving no room for the summary after the whole
        // conversation.
        let extend = extend && self.prompt_tokens(live) + summary_room(ep.local) <= self.context_window(target);
        match self.compact_once(live, id, target, ep, extend, cancel).await {
            Err(err) if !is_cancelled(&err) => {
                eprintln!("[summary] {err:#}");
                self.compact_once(live, id, target, ep, false, cancel).await
            }
            done => done,
        }
    }

    async fn compact_once(self: &Arc<Self>, live: &Arc<Live>, id: &str, target: &Target, ep: &Endpoint, extend: bool, cancel: &CancellationToken) -> anyhow::Result<()> {
        let conv = live.conv.lock().unwrap().clone();
        let start = conv.compaction.as_ref().map(|c| c.kept_from).unwrap_or(0);
        let keep_chars = (self.context_window(target) as f64 * KEEP_SHARE * CHARS_PER_TOKEN) as usize;
        let Some(kept_from) = kept_from(&conv.messages, start, keep_chars) else {
            bail!(trf("The last message is too long for {model}'s context. Start a new conversation, or pick a model with a larger context in Settings.", &[("model", &ep.model)]));
        };
        let (system, view, _) = self.request_parts(live);
        // The view starts with the previous summary when there is one, and maps conversation indexes after it.
        let offset = if conv.compaction.is_some() { 1 } else { 0 };
        let mut messages: Vec<Message> = if extend { view } else { view[..offset + (kept_from - start)].to_vec() };
        let question = in_language(SUMMARY_PROMPT, "Write the summary in {language}, and keep the Title: label in English.", &self.settings());
        messages.push(Message::User(UserMessage { text: prompt::system_block(&question), ..Default::default() }));
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
        let label = tr(if draft.is_some() { "Continuing the summary of earlier messages" } else { "Summarizing earlier messages to make room" });
        self.emit(Event::Activity { conv: id.into(), text: Some(label.into()) });
        let written = Mutex::new(draft.as_ref().map(|d| d.text.clone()).unwrap_or_default());
        let pieces = AtomicU64::new(0);
        let on_text = |t: &str| {
            written.lock().unwrap().push_str(t);
            let n = pieces.fetch_add(1, Ordering::SeqCst) + 1;
            if n % 8 == 1 {
                let words = written.lock().unwrap().split_whitespace().count();
                self.emit(Event::Activity { conv: id.into(), text: Some(trf("{activity}: {count} words written", &[("activity", &label), ("count", &thousands(words as u64))])) });
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
        let session = prompt::system_block(&prompt::session_block(&conv.cwd, &folder, self.settings().language()));
        live.conv.lock().unwrap().set_compaction(summary, kept_from, environment, session)?;
        // Notes attached to the summarized messages are gone from the request, so they can be attached again.
        live.read_notes.lock().unwrap().clear();
        self.emit(Event::NotesChanged { cwd: conv.cwd.clone() });
        // The next request reads the summary in saved steps. The summary message opens with the same environment
        // block as a new conversation, so that read starts from the saved prompt for new conversations here.
        if matches!(target, Target::Local(_)) {
            self.llama.set_slot_owner(None);
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
            comp.context = prompt::system_block(&prompt::session_block(&cwd, &s.notes_folder, s.language()));
        }
        let messages = match &c.compaction {
            Some(comp) if comp.kept_from <= c.messages.len() => {
                let mut out = vec![Message::User(UserMessage {
                    text: format!("<summary>\n{}\n</summary>", comp.summary.trim()),
                    environment: comp.environment.clone(),
                    context: comp.context.clone(),
                    ..Default::default()
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
            ..Default::default()
        };
        if call.name == "new_project" {
            return self.start_project(live, id, call, cancel).await;
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
            jobs: Some(live.jobs.clone()),
            browser: Some(live.browser.clone()),
        };
        let outcome = tools::run(call, cwd, &self.shell, &limits, cancel, move |tail| {
            let _ = events.send(Event::ToolOutput { conv: conv.clone(), call_id: call_id.clone(), tail });
        })
        .await;
        if tools::changes_files(&call.name) && self.inside_notes(cwd, call) {
            self.emit(Event::NotesChanged { cwd: cwd.to_path_buf() });
        }
        let mut result = make(outcome.output, outcome.is_error, outcome.diff);
        if let Some(image) = outcome.image {
            self.add_screenshot(live, id, call, &mut result, image, cancel).await;
        }
        result
    }

    /// Gives a screenshot to a model that can see images, and has another model describe it for one that cannot.
    async fn add_screenshot(&self, live: &Live, id: &str, call: &ToolCall, result: &mut ToolResult, image: conversation::Image, cancel: &CancellationToken) {
        let model = live.conv.lock().unwrap().model.clone();
        result.images = vec![image.clone()];
        if self.sees_images(&model) {
            result.output.push_str(" The screenshot follows.");
            return;
        }
        result.described = true;
        match self.describe_image(live, id, &image, "screenshot of a web page", call.arg("question"), cancel).await {
            Ok((helper, text)) => {
                result.output.push_str(&format!("\n\n{}", prompt::system_block(&format!("You cannot see images, so {helper} looked at the screenshot and described it:\n{text}"))));
            }
            Err(err) if is_cancelled(&err) => result.output.push_str("\n\nThe user stopped the task before the screenshot was described."),
            Err(err) => result.output.push_str(&format!("\n\nThe screenshot could not be described: {err:#} Check the page with browser_read and browser_script instead.")),
        }
    }

    fn sees_images(&self, model: &str) -> bool {
        match self.target(model) {
            Ok(Target::Local(m)) => m.mmproj.is_some(),
            Ok(Target::Hosted(h, ..)) => h.vision,
            Err(_) => false,
        }
    }

    /// The model that describes images for models that cannot see them: the smallest local model with an image
    /// projector, or else a hosted model that sees images.
    fn image_helper(&self) -> Option<Target> {
        let local = self.llama.models().into_iter().filter(|m| m.mmproj.is_some()).min_by_key(|m| m.size);
        if let Some(m) = local {
            return Some(Target::Local(m));
        }
        let hosted = self.models().into_iter().find(|m| m.usable && m.vision && !m.provider.is_empty())?;
        self.target(&hosted.name).ok()
    }

    /// Has a model that sees images describe one for a model that cannot, and returns the helper's name with the
    /// description. A local helper runs in a server of its own beside the conversation's model when memory allows.
    /// Otherwise the conversation's model saves what it has read, makes room, and loads again afterward.
    async fn describe_image(
        &self,
        live: &Live,
        id: &str,
        image: &conversation::Image,
        what: &str,
        question: &str,
        cancel: &CancellationToken,
    ) -> anyhow::Result<(String, String)> {
        let helper = self.image_helper().context("No model that can see images is set up. Download Qwen3.5 9B in Settings so the system can describe images.")?;
        let mut prompt = format!(
            "Describe this {what} for a programmer who cannot see it. Give the text it shows word for word, then the layout, colors, and sizes of what is on it. Point out anything that looks wrong, such as overlapping or cut-off parts, blank areas, missing images, or error messages. Be specific and brief."
        );
        if !question.trim().is_empty() {
            prompt.push_str(&format!(" Also answer this question about it: {}", question.trim()));
        }
        let message = Message::User(UserMessage { text: prompt, images: vec![image.clone()], time: now_millis(), ..Default::default() });
        let model = match helper {
            Target::Hosted(h, ..) => {
                self.emit(Event::Activity { conv: id.into(), text: Some(trf("{model} is looking at the image", &[("model", &h.name)])) });
                let target = self.target(&h.reference())?;
                let ep = self.endpoint(&target);
                let req = ChatRequest {
                    system: IMAGE_HELPER,
                    messages: std::slice::from_ref(&message),
                    tools: &[],
                    thinking: Thinking::Off,
                    thinking_budget: None,
                    max_tokens: HELPER_TOKENS,
                };
                let reply = stream::send(&self.http, &ep, &stream::payload(&ep, &req), cancel, |_| {}).await?;
                if reply.stop == StopReason::Aborted {
                    return Err(crate::util::Cancelled.into());
                }
                return Ok((h.name.clone(), prompt::outside(reply.text.trim())));
            }
            Target::Local(m) => m,
        };
        let need = self.llama.memory_needed(&model, HELPER_CONTEXT);
        let conv_model = match self.target(&live.conv.lock().unwrap().model.clone()) {
            Ok(Target::Local(m)) => Some(m),
            _ => None,
        };
        let swap = crate::sys::available_memory() < need + HELPER_MARGIN;
        if swap {
            if conv_model.is_none() && self.turn_holder.lock().unwrap().as_deref().is_some_and(|holder| holder != id) {
                bail!("{} needs about {} of free memory to look at the image, and a task in another conversation is using the local model.", model.name, crate::util::gb(need));
            }
            self.stop_baking();
            if self.llama.slot_owner().as_deref() == Some(id)
                && let Err(err) = self.persist(live, false, false, cancel).await
            {
                if is_cancelled(&err) {
                    return Err(err);
                }
                eprintln!("[cache] {err:#}");
            }
            self.llama.stop().await;
            self.llama.set_slot_owner(None);
        }
        self.emit(Event::Activity { conv: id.into(), text: Some(trf("{model} is looking at the image", &[("model", &model.name)])) });
        let body = stream::payload(
            &self.endpoint(&Target::Local(model.clone())),
            &ChatRequest { system: IMAGE_HELPER, messages: std::slice::from_ref(&message), tools: &[], thinking: Thinking::Off, thinking_budget: None, max_tokens: HELPER_TOKENS },
        );
        let reply = self.llama.run_side(&model, HELPER_CONTEXT, &body, cancel).await;
        if swap && let Some(main) = &conv_model {
            self.emit(Event::Activity { conv: id.into(), text: Some(trf("Loading {model} again", &[("model", &main.name)])) });
            self.llama.ensure(main).await?;
            if let Err(err) = self.persist(live, false, false, cancel).await {
                if is_cancelled(&err) {
                    return Err(err);
                }
                eprintln!("[cache] {err:#}");
            }
        }
        let text = reply?["choices"][0]["message"]["content"].as_str().unwrap_or_default().trim().to_string();
        if text.is_empty() {
            bail!("{} returned no description.", model.name);
        }
        // An image can show text written to pass for the system's.
        Ok((model.name.clone(), prompt::outside(&text)))
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
                "This task runs unattended, so the system only changes files inside the project folder. Continue without this change, and tell the user about it when they are back."
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
            Decision::Deny | Decision::Plan(_) => Verdict::Deny("The user declined this tool call. Ask them how to proceed.".into()),
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
        let custom = s.model_prompt(&c.model).unwrap_or_default().to_string();
        if let Some(p) = &c.prompt
            && p.notes_folder == s.notes_folder
            && p.web == s.web_access
            && p.general == general
            && p.custom == custom
        {
            return (p.system.clone(), p.tools.clone());
        }
        let system = self.system_prompt(&s, &c.model);
        let tools = tools::specs(&self.shell, general, s.web_access, web::has_browser());
        // Only a running task pins them, so opening a conversation to read it does not write to its file.
        if live.running.load(Ordering::SeqCst) {
            let pin = conversation::PinnedPrompt { system: system.clone(), tools: tools.clone(), notes_folder: s.notes_folder.clone(), web: s.web_access, general, custom };
            if let Err(err) = c.set_prompt(pin) {
                eprintln!("[prompt] {err:#}");
            }
        }
        (system, tools)
    }

    /// The system prompt for `model`: the user's own for it, or Scoobert's.
    fn system_prompt(&self, s: &Settings, model: &str) -> String {
        prompt::system_prompt(s.model_prompt(model).unwrap_or(prompt::DEFAULT_PROMPT), &s.notes_folder, self.shell.tool_name())
    }

    /// Creates a project folder for a conversation that has none and moves the conversation into it, after the user
    /// chooses whether Scoobert plans the project first.
    async fn start_project(&self, live: &Live, id: &str, call: &ToolCall, cancel: &CancellationToken) -> ToolResult {
        let make = |output: String, is_error: bool| ToolResult { call_id: call.id.clone(), name: call.name.clone(), output, is_error, diff: None, time: now_millis(), ..Default::default() };
        let current = live.conv.lock().unwrap().cwd.clone();
        if !is_general(&current) {
            return make("A project is already open, so keep working in it.".into(), true);
        }
        let Some(plan) = self.choose_plan(id, call, cancel).await else {
            return make("The user stopped before the project started.".into(), true);
        };
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
        let folder_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let shown = crate::paths::display(&path);
        // Models tend to keep the folder name in the paths they planned before the folder existed.
        let mut output = format!(
            "Started the project folder {shown}, and commands now run in it. Give paths relative to it, such as main.py for {shown}/main.py. Do not start a path with {folder_name}/, which would make a second {folder_name} folder inside it."
        );
        if plan != PlanFirst::Off {
            output.push_str(&format!(
                "\n\n{}",
                prompt::system_block(&format!("The user chose to plan first. {}", plan_instructions(&self.settings().notes_folder, plan == PlanFirst::Build)))
            ));
        }
        if plan == PlanFirst::Discuss {
            live.notes_update.store(true, Ordering::SeqCst);
        }
        make(output, false)
    }

    /// Asks the user whether to plan a new project first. Unattended work uses the last choice, since nobody is there
    /// to pick. None when the user stopped the task instead.
    async fn choose_plan(&self, id: &str, call: &ToolCall, cancel: &CancellationToken) -> Option<PlanFirst> {
        let s = self.settings();
        if s.approvals == Approvals::Project {
            return Some(s.plan_first);
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
            Decision::Plan(plan) => Some(plan),
            _ => None,
        }
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
        // The newest message, the user's or a stopped reply being continued, goes with the request, and the cache
        // covers what came before it.
        let history = conv.messages.len() > 1;
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
        let cancel = CancellationToken::new();
        *live.cache_cancel.lock().unwrap() = Some(cancel.clone());
        let result = self.persist(live, !history, false, &cancel).await;
        *live.cache_cancel.lock().unwrap() = None;
        result
    }

    /// Fills the slot with the part of the request that the next request starts with, and saves it. The fill
    /// starts from the saved prompt that matches the most of it, so only the difference is read.
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
            let Some(prefix) = self.history_prefix(live, &target, exclude_last).await? else { return Ok(()) };
            (prefix, self.llama.slot_file("chat", &conv.id))
        };
        let tokens = self.llama.tokenize(&prefix).await?;
        // A slot that already holds this conversation holds it followed by the last reply, which the server
        // matches itself.
        let held = if !shared && self.llama.slot_owner().as_deref() == Some(&conv.id) {
            None
        } else {
            let held = self.llama.restore_longest(&tokens).await;
            if held == tokens.len() {
                self.llama.set_slot_owner(Some(conv.id.clone()));
                return Ok(());
            }
            Some(held)
        };
        let _busy = self.llama.busy();
        // The activity appears with the first progress report, so a fill that takes a moment shows nothing.
        let label = tr(if shared { "Reading Scoobert's instructions" } else { "Reading the conversation" });
        let mut shown = false;
        let filled = self
            .llama
            .fill(&tokens, held, &name, cancel, |done, total| {
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
        self.llama.set_slot_owner(Some(conv.id.clone()));
        Ok(())
    }

    /// The rendered start of the next request: instructions, tools, and messages, without the newest message when
    /// `exclude_last`. None when a message has images, which the saved prompts leave out.
    async fn history_prefix(&self, live: &Live, target: &Target, exclude_last: bool) -> anyhow::Result<Option<String>> {
        let (system, mut messages, thinking) = self.request_parts(live);
        if exclude_last {
            messages.pop();
        }
        if messages.iter().any(Message::sends_images) {
            return Ok(None);
        }
        let tools = self.tool_specs(live);
        let req = ChatRequest { system: &system, messages: &messages, tools: &tools, thinking, thinking_budget: None, max_tokens: self.max_tokens(target) };
        Ok(Some(self.llama.shared_prefix(&stream::payload(&self.endpoint(target), &req), "").await?))
    }

    async fn history_tokens(&self, live: &Live, target: &Target, exclude_last: bool) -> anyhow::Result<Option<Vec<i32>>> {
        match self.history_prefix(live, target, exclude_last).await? {
            Some(prefix) => Ok(Some(self.llama.tokenize(&prefix).await?)),
            None => Ok(None),
        }
    }

    /// The prompt every new conversation in `cwd` starts with: instructions, tools, and the environment block that
    /// opens its first message.
    async fn opening_prefix(&self, target: &Target, cwd: &Path, thinking: Thinking, environment: &str) -> anyhow::Result<String> {
        let s = self.settings();
        let model = match target {
            Target::Local(m) => m.name.clone(),
            Target::Hosted(h, ..) => h.reference(),
        };
        let system = self.system_prompt(&s, &model);
        let tools = tools::specs(&self.shell, is_general(cwd), s.web_access, web::has_browser());
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
        // Loading a model while a task runs would stop the task's model.
        if self.turn_holder.lock().unwrap().is_some() {
            return false;
        }
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
        let busy = self.llama.in_use()
            || self.turn_holder.lock().unwrap().is_some()
            || self.convs.lock().unwrap().values().any(|l| l.running.load(Ordering::SeqCst));
        if busy || self.llama.loaded_model().as_deref() != Some(&model.name) {
            return Ok(());
        }
        let s = self.settings();
        let target = Target::Local(model.clone());
        let environment = prompt::environment_block(cwd, &s.notes_folder, &self.shell);
        let prefix = self.opening_prefix(&target, cwd, s.thinking, &environment).await?;
        let name = self.llama.slot_file("shared", &LlamaServer::hash(&prefix));
        let tokens = self.llama.tokenize(&prefix).await?;
        if self.llama.holds(&name, &tokens) {
            return Ok(());
        }
        self.emit(Event::Preparing(Some((model.name.clone(), 0))));
        let _busy = self.llama.busy();
        // A bake that was stopped partway saved what it had read, and continues from there.
        let held = self.llama.restore_longest(&tokens).await;
        let result = if held == tokens.len() {
            self.llama.save(&name, &tokens).await
        } else {
            self.llama
                .fill(&tokens, Some(held), &name, cancel, |done, total| {
                    if total > 0 {
                        self.emit(Event::Preparing(Some((model.name.clone(), done * 100 / total))));
                    }
                })
                .await
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
            let turn = live.turn.lock().unwrap().take();
            let local = host.after_steps(&live, id).await;
            if turn.is_some() {
                drop(turn);
                host.emit(Event::ModelFree);
            }
            // The model is loaded and idle now, so this is when saved prompts for new conversations get built.
            if local {
                let conv = live.conv.lock().unwrap().clone();
                host.bake_soon(&conv.model, vec![conv.cwd.clone(), crate::paths::projects_root()]);
            }
        });
    }

    /// Saves the cache, names the conversation, and takes notes after a task. Returns whether it ran on the local model.
    async fn after_steps(self: &Arc<Self>, live: &Arc<Live>, id: ConvId) -> bool {
        let host = self;
        {
            let _guard = live.background.lock().await;
            let s = host.settings();
            let conv = live.conv.lock().unwrap().clone();
            let local = matches!(host.target(&conv.model), Ok(Target::Local(_))) && host.llama.loaded_model().is_some();
            let cancel = CancellationToken::new();
            *live.cache_cancel.lock().unwrap() = Some(cancel.clone());
            // Continue carries on a stopped reply from where it starts, which a save that holds it closed cannot give.
            let stopped = matches!(conv.messages.last(), Some(Message::Assistant(a)) if continuable(a));
            if local
                && let Err(err) = host.persist(&live, false, stopped, &cancel).await
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
                let question = in_language(TITLE_PROMPT, "Write the title in {language}.", &s);
                match host.ask(&live, &question, TITLE_MAX_TOKENS, &cancel).await {
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
            let Some(task) = finished_task(&conv, &s.notes_folder, related) else {
                return local;
            };
            let mut facts = Vec::new();
            let updated_notes = live.notes_update.swap(false, Ordering::SeqCst);
            if (local || hosted) && s.remember_step && !updated_notes && memory::worth_noting(&task) {
                host.emit(Event::Activity { conv: id.clone(), text: Some(tr("Taking notes...").into()) });
                let cancel = CancellationToken::new();
                *live.note_cancel.lock().unwrap() = Some(cancel.clone());
                let question = memory::remember_prompt(&memory::topic_names(&crate::notes::Vault::new(conv.cwd.join(&s.notes_folder))));
                let question = in_language(&question, "Write each topic and fact in {language}, and keep the words Decision, Convention, Problem, and NONE in English.", &s);
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
            local
        }
    }

    /// Asks one short question at the end of the conversation, with thinking off. The request extends the cached
    /// conversation, so only the question is read, and the answer is short because each token takes about a
    /// second on a CPU.
    async fn ask(&self, live: &Live, question: &str, max_tokens: u32, cancel: &CancellationToken) -> anyhow::Result<String> {
        let conv = live.conv.lock().unwrap().clone();
        let target = self.target(&conv.model)?;
        let ep = self.endpoint(&target);
        let (system, mut messages, _) = self.request_parts(live);
        messages.push(Message::User(UserMessage { text: prompt::system_block(question), ..Default::default() }));
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

    /// Stops a background command the user chose to stop.
    pub fn stop_job(&self, conv: &str, job: u32) {
        if let Ok(live) = self.live(conv) {
            live.jobs.stop_quietly(job);
        }
    }

    pub async fn shutdown(&self) {
        let lives: Vec<Arc<Live>> = self.convs.lock().unwrap().values().cloned().collect();
        for live in lives {
            live.jobs.stop_all();
            drop(live.browser.lock().await.take());
        }
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

/// The last reply with the user message it answered, for learning. None when the reply holds no text.
fn last_rating(c: &Conversation, good: bool) -> Option<crate::llama::learn::Rating> {
    let Some(Message::Assistant(a)) = c.messages.last() else { return None };
    if a.text.trim().is_empty() {
        return None;
    }
    let asked = c.messages.iter().rev().find_map(|m| match m {
        Message::User(u) => Some(u.text.clone()),
        _ => None,
    })?;
    Some(crate::llama::learn::Rating { good, asked, reply: a.text.trim().to_string(), time: now_millis() })
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

/// A side step's question, followed by `ask` with the language filled in when the language is not English.
fn in_language(question: &str, ask: &str, settings: &Settings) -> String {
    let language = settings.language();
    if language.code == "en" {
        return question.to_string();
    }
    format!("{question} {}", ask.replace("{language}", language.english))
}

/// The model's title answer, cleaned of labels, quotes, and trailing punctuation, when it is usable.
fn clean_title(answer: &str) -> Option<String> {
    // A small model can repeat the step's label before its answer, with the frame it came in or the older
    // "Scoobert title step" label.
    const ECHO: &str = "Title step";
    const OLD: &str = "Scoobert ";
    let line = answer
        .lines()
        .map(|l| {
            let l = l.trim().trim_start_matches("⟦System:").trim_end_matches('⟧').trim();
            let l = match l.get(..OLD.len()) {
                Some(start) if start.eq_ignore_ascii_case(OLD) => &l[OLD.len()..],
                _ => l,
            };
            match l.get(..ECHO.len()) {
                Some(start) if start.eq_ignore_ascii_case(ECHO) => l[ECHO.len()..].trim_start_matches(['.', ':', '-', ' ']),
                _ => l,
            }
        })
        .find(|l| !l.is_empty())?;
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
    const LONG: usize = tools::SHORTENED_WRITE;
    // Each saved call, and which note its result promised: 2 for the first and last lines framed as Scoobert's text,
    // 1 for them unframed, as before 0.4.7, and 0 for a count, as before 0.2.6. Notes whose result says they stay in
    // full are left out.
    let saved: HashMap<String, u8> = messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) if !t.is_error && !t.output.contains(tools::KEPT_IN_FULL.trim()) => {
                let style = if t.output.contains(tools::SHORTENED_SYSTEM.trim()) {
                    2
                } else if t.output.contains(tools::SHORTENED_NOTICE.trim()) {
                    1
                } else {
                    0
                };
                Some((t.call_id.clone(), style))
            }
            _ => None,
        })
        .collect();
    for m in &mut messages {
        let Message::Assistant(a) = m else { continue };
        for call in &mut a.tool_calls {
            let Some(&style) = saved.get(&call.id) else { continue };
            let path = call.arg("path").to_string();
            let Some(args) = call.arguments.as_object_mut() else { continue };
            // The text moves to a field of another name, so the history never shows a note where file content goes,
            // which a model copied into new writes.
            for key in ["content", "new_text", "old_text"] {
                let Some(text) = args.get(key).and_then(Value::as_str).filter(|s| s.chars().count() > LONG).map(String::from) else { continue };
                // A note alone read to a model as if it had sent the note instead of code, so it wrote the file again.
                // Its own first and last lines show that the code went out.
                let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
                let first = crate::util::clip(lines.first().copied().unwrap_or_default(), 120);
                let last = crate::util::clip(lines.last().copied().unwrap_or_default(), 120);
                let note = match style {
                    2 => format!(
                        "{}\nFirst line: {first}\nLast line: {last}",
                        prompt::system_note(&format!(
                            "{} lines saved in full in {path}, shown here by their first and last line to save room. Read the file for its exact text.",
                            text.lines().count()
                        ))
                    ),
                    1 => format!("{} lines saved in full in {path}, shown here by their first and last line to save room.\nFirst line: {first}\nLast line: {last}", text.lines().count()),
                    _ => format!("{} characters, saved in full in {path}", text.chars().count()),
                };
                args.remove(key);
                args.insert(format!("{key}_saved"), note.into());
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

/// Tokens a summary request adds after the whole conversation: the question, the thinking, the summary, and the
/// chat template around them.
fn summary_room(local: bool) -> u64 {
    let summary = if local { SUMMARY_MAX_TOKENS_LOCAL } else { SUMMARY_MAX_TOKENS_HOSTED };
    (SUMMARY_PROMPT.len() as f64 / CHARS_PER_TOKEN) as u64 + (summary + SIDE_THINKING_TOKENS) as u64 + 256
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
            format!("- {} ({date}): read conversation:{}", prompt::outside(&s.title), &s.id[..s.id.len().min(8)])
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
            output: "The system closed before this ran.".into(),
            is_error: true,
            time: now_millis(),
            ..Default::default()
        }))?;
    }
    Ok(())
}

/// Whether the last run ended because the user pressed Stop.
/// The activity line while a model loads, with a note on the first load, which reads the whole model from disk.
fn loading_line(model: &str, secs: u64, first: bool) -> String {
    let line = trf("Loading {model}: {secs} s", &[("model", &model), ("secs", &secs)]);
    if first { format!("{line}. {}", tr("The first load of a model takes longer, because it reads the whole model from disk.")) } else { line }
}

/// Whether Continue can carry on this reply where it stopped: it stopped before it called a tool and holds some
/// reasoning or text.
fn continuable(a: &AssistantMessage) -> bool {
    a.stop == StopReason::Aborted && a.tool_calls.is_empty() && !(a.thinking.is_empty() && a.text.is_empty())
}

/// The note sent with Continue, which says what the last run left behind.
fn resume_note(last: Option<&Message>) -> String {
    let left = match last {
        Some(Message::Assistant(a)) if a.stop == StopReason::Aborted && !a.tool_calls.is_empty() => Some(RESUME_WRITE),
        Some(Message::Assistant(a)) if continuable(a) => Some(RESUME_CUT),
        _ => None,
    };
    match left {
        Some(more) => format!("<interrupted>{RESUME} {more}</interrupted>"),
        None => format!("<interrupted>{RESUME}</interrupted>"),
    }
}

#[cfg(test)]
mod resume_note_tests {
    use super::*;

    fn stopped(thinking: &str, calls: usize) -> Message {
        let call = ToolCall { id: "1".into(), name: "write".into(), arguments: serde_json::json!({}) };
        Message::Assistant(AssistantMessage { stop: StopReason::Aborted, thinking: thinking.into(), tool_calls: vec![call; calls], ..Default::default() })
    }

    #[test]
    fn the_note_names_only_what_the_stop_left() {
        assert!(resume_note(Some(&stopped("half a thought", 0))).contains(RESUME_CUT));
        assert!(resume_note(Some(&stopped("", 1))).contains(RESUME_WRITE));
        let plain = resume_note(Some(&Message::Tool(ToolResult { call_id: "1".into(), name: "bash".into(), output: String::new(), is_error: false, diff: None, time: 0, ..Default::default() })));
        assert!(!plain.contains(RESUME_CUT) && !plain.contains(RESUME_WRITE));
        assert!(!continuable(match &stopped("", 0) {
            Message::Assistant(a) => a,
            _ => unreachable!(),
        }));
    }
}

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
            Message::User(u) => u.text.len() + u.context.len() + u.sent_images().len() * 3000,
            Message::Assistant(a) => a.thinking.len() + a.text.len() + a.tool_calls.iter().map(|c| c.arguments.to_string().len() + 40).sum::<usize>(),
            Message::Tool(t) => t.output.len() + 40 + t.sent_images().len() * 3000,
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
            Message::Tool(ToolResult { call_id: "ok".into(), name: "write".into(), output: "Created".into(), is_error: false, diff: None, time: 0, ..Default::default() }),
            Message::Tool(ToolResult { call_id: "failed".into(), name: "write".into(), output: "Could not write".into(), is_error: true, diff: None, time: 0, ..Default::default() }),
        ];
        let out = super::shorten_saved_writes(messages);
        let Message::Assistant(a) = &out[0] else { panic!() };
        assert!(a.tool_calls[0].arguments.get("content").is_none());
        assert_eq!(a.tool_calls[0].arg("content_saved"), "3000 characters, saved in full in a.ts");
        assert_eq!(a.tool_calls[1].arg("content").len(), 3000);
    }

    #[test]
    fn new_long_writes_show_their_first_and_last_lines() {
        let code = format!("import x from \"y\";\n{}console.log(done);\n", "let a = 1;\n".repeat(200));
        let call = ToolCall { id: "w".into(), name: "write".into(), arguments: serde_json::json!({ "path": "a.ts", "content": code }) };
        let output = format!("Created a.ts (now 9 bytes).{}", super::tools::SHORTENED_NOTICE);
        let messages = vec![
            Message::Assistant(AssistantMessage { tool_calls: vec![call], ..Default::default() }),
            Message::Tool(ToolResult { call_id: "w".into(), name: "write".into(), output, is_error: false, diff: None, time: 0, ..Default::default() }),
        ];
        let out = super::shorten_saved_writes(messages);
        let Message::Assistant(a) = &out[0] else { panic!() };
        let note = a.tool_calls[0].arg("content_saved").to_string();
        assert!(note.starts_with("202 lines saved in full in a.ts"), "{note}");
        assert!(note.ends_with("First line: import x from \"y\";\nLast line: console.log(done);"), "{note}");
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
        assert_eq!(clean_title("Scoobert title step"), None);
        assert_eq!(clean_title("Scoobert title step.\nFind the first public function").as_deref(), Some("Find the first public function"));
        assert_eq!(clean_title("scoobert title step: Save loader").as_deref(), Some("Save loader"));
        assert_eq!(clean_title("⟦System:\nTitle step\n⟧\nBattle engine fixes").as_deref(), Some("Battle engine fixes"));
    }

    fn user(n: usize) -> Message {
        Message::User(UserMessage { text: "u".repeat(n), ..Default::default() })
    }
    fn call(n: usize) -> Message {
        let call = ToolCall { id: "c".into(), name: "read".into(), arguments: serde_json::json!({ "path": "x".repeat(n) }) };
        Message::Assistant(AssistantMessage { tool_calls: vec![call], ..Default::default() })
    }
    fn result(n: usize) -> Message {
        Message::Tool(ToolResult { call_id: "c".into(), name: "read".into(), output: "o".repeat(n), is_error: false, diff: None, time: 0, ..Default::default() })
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
