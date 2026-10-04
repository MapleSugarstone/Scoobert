//! The Settings dialog: general options, models on this computer, and hosted models with API keys.

use std::path::PathBuf;
use std::sync::Arc;

use iced::widget::scrollable::RelativeOffset;
use iced::widget::{Column, button, checkbox, column, container, operation, pick_list, progress_bar, row, rule, scrollable, space, text, text_editor, text_input};
use iced::{Alignment, Element, Fill, Length, Task};

use super::Message;
use super::fonts;
use super::icons::{Icon, icon};
use super::setup::{self, Download, Job};
use super::theme;
use crate::agent::prompt::DEFAULT_PROMPT;
use crate::agent::providers::{self, Provider};
use crate::agent::{Host, ModelOption};
use crate::i18n::{tr, trf};
use crate::llama::catalog::{CATALOG, memory_needed};
use crate::llama::cuda;
use tokio_util::sync::CancellationToken;
use crate::store::{Approvals, CustomProvider, HostedModel, State, ThemeChoice};
use crate::util::gb;

const KEEP_ALIVE: [KeepAlive; 4] = [KeepAlive(5), KeepAlive(30), KeepAlive(120), KeepAlive(0)];
const CONTEXT_SIZES: [u32; 6] = [8192, 16_384, 32_768, 65_536, 131_072, 262_144];
const SUPPORT_URL: &str = "https://ko-fi.com/krazvalt";
const PAGE_ID: &str = "settings-page";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    General,
    Local,
    Hosted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAlive(u32);

impl std::fmt::Display for KeepAlive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            0 => f.write_str(tr("Until Scoobert closes")),
            m if m < 60 => f.write_str(&trf("{minutes} minutes", &[("minutes", &m)])),
            m => f.write_str(&trf("{hours} hours", &[("hours", &(m / 60))])),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderChoice(pub Provider);

impl std::fmt::Display for ProviderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextChoice(pub u32);

impl std::fmt::Display for ContextChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&trf("{size}K tokens", &[("size", &(self.0 / 1024))]))
    }
}

pub enum Effect {
    None,
    Saved,
    ModelsChanged,
    Toast(String),
}

pub struct Ctx<'a> {
    pub state: &'a mut State,
    pub host: Option<&'a Arc<Host>>,
}

pub struct ViewCtx<'a> {
    pub state: &'a State,
    /// How unattended commands are isolated on this computer.
    pub isolation: &'a str,
    pub models: &'a [ModelOption],
    pub download: &'a Option<Download>,
    pub queue: &'a [Job],
    pub choices: &'a Option<crate::llama::download::ChooseSize>,
}

pub struct Panel {
    section: Section,
    notes_folder: String,
    models_dir: String,
    server_path: String,
    provider: Option<Provider>,
    key_input: String,
    has_key: bool,
    listing: bool,
    available: Vec<HostedModel>,
    filter: String,
    error: Option<String>,
    custom_name: String,
    custom_url: String,
    custom_spec: String,
    /// The model lab page, which replaces the model list while it is open.
    lab: Option<super::lab::Lab>,
    /// The system prompt page for one model, which replaces the section while it is open.
    prompt: Option<PromptPage>,
    /// Bytes of NVIDIA support downloaded so far and in all, while it downloads.
    cuda_progress: Option<(u64, u64)>,
    cuda_cancel: Option<CancellationToken>,
    cuda_error: Option<String>,
    /// Ratings and learning for each local model, read when the panel opens and after each change.
    learned: std::collections::HashMap<String, crate::llama::learn::Summary>,
    learn_run: Option<LearnRun>,
}

struct PromptPage {
    /// The model's name, or a hosted model's reference.
    model: String,
    label: String,
    content: text_editor::Content,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Init,
    Section(Section),
    Theme(ThemeChoice),
    Language(&'static crate::i18n::Language),
    Approvals(Approvals),
    NotesFolder(String),
    CommitNotesFolder,
    ActivityLog(bool),
    RememberStep(bool),
    WebAccess(bool),
    KeepAlive(KeepAlive),
    PreloadModel(bool),
    UseGpu(bool),
    ModelsFromDisk(bool),
    CompactContext(bool),
    /// How a model predicts words ahead for itself to check.
    Predict(String, PredictChoice),
    ModelsDir(String),
    CommitModelsDir,
    BrowseModelsDir,
    ModelsDirPicked(Option<PathBuf>),
    ServerPath(String),
    CommitServerPath,
    Context(String, ContextChoice),
    Download(usize),
    CustomSpec(String),
    DownloadCustom,
    /// Downloads one of the sizes a repository offered.
    DownloadSize(String),
    AddModelFile,
    ModelFilePicked(Option<std::path::PathBuf>),
    /// Leaves an added model file out of the list, without deleting it.
    ForgetModelFile(String),
    /// Asks before deleting a downloaded model's files.
    DeleteModel { name: String, size: u64, variants: usize },
    CancelDownload,
    RevealModels,
    Provider(ProviderChoice),
    KeyInput(String),
    SaveKey,
    RemoveKey,
    ListModels,
    Listed(String, Result<Vec<HostedModel>, String>),
    Filter(String),
    ToggleModel(HostedModel, bool),
    RemoveHosted(String),
    CustomName(String),
    CustomUrl(String),
    AddCustom,
    RemoveCustom(String),
    OpenUrl(String),
    Uninstall,
    CheckUpdates,
    ShowFile(PathBuf),
    OpenLab(String),
    Lab(super::lab::Msg),
    LabDeleted,
    LabFailed(String),
    CudaDownload,
    CudaProgress(u64, u64),
    /// NVIDIA support finished downloading or was removed, or the error that stopped it. An empty error is a stop.
    CudaDone(Result<(), String>),
    CudaCancel,
    CudaRemove,
    /// Opens the system prompt page for a model, by name or reference, with the label to show.
    OpenPrompt(String, String),
    PromptEdit(text_editor::Action),
    SavePrompt,
    ResetPrompt,
    ClosePrompt,
    /// Turns learning from ratings on or off for a model.
    Learning(String, bool),
    LearnStrength(String, LearnStrength),
    ApplyLearning(String),
    LearnProgress(u64, u64),
    LearnDone(String, Result<(), String>),
    LearnCancel,
    ResetLearning(String),
    Close,
}

impl Panel {
    pub fn new(section: Section, settings: &crate::store::Settings) -> Panel {
        let provider = providers::all(&settings.custom_providers).into_iter().next();
        Panel {
            section,
            notes_folder: settings.notes_folder.clone(),
            models_dir: settings.models_dir.clone(),
            server_path: settings.llama_server_path.clone(),
            provider,
            key_input: String::new(),
            has_key: false,
            listing: false,
            available: Vec::new(),
            filter: String::new(),
            error: None,
            custom_name: String::new(),
            custom_url: String::new(),
            custom_spec: String::new(),
            lab: None,
            prompt: None,
            cuda_progress: None,
            cuda_cancel: None,
            cuda_error: None,
            learned: std::collections::HashMap::new(),
            learn_run: None,
        }
    }

    fn select_provider(&mut self, p: Provider, host: Option<&Arc<Host>>) -> Task<Message> {
        self.has_key = host.is_some_and(|h| h.key(&p.id).is_some());
        self.provider = Some(p);
        self.key_input.clear();
        self.available.clear();
        self.error = None;
        self.filter.clear();
        if self.has_key || self.provider.as_ref().is_some_and(|p| p.id == "openrouter" || p.base_url.starts_with("http://")) {
            return Task::done(Message::Settings(Msg::ListModels));
        }
        Task::none()
    }

    pub fn update(&mut self, msg: Msg, ctx: &mut Ctx) -> (Task<Message>, Effect) {
        let s = &mut ctx.state.settings;
        match msg {
            Msg::Init => {
                self.refresh_learned(ctx.host, s);
                if let Some(p) = self.provider.clone() {
                    return (self.select_provider(p, ctx.host), Effect::None);
                }
            }
            Msg::Section(section) => {
                self.section = section;
                self.prompt = None;
                if section == Section::Hosted {
                    return self.update(Msg::Init, ctx);
                }
            }
            Msg::Theme(t) => {
                s.theme = t;
                return (Task::none(), Effect::Saved);
            }
            Msg::Language(language) => {
                crate::i18n::set(language.code);
                s.language = language.code.to_string();
                return (Task::none(), Effect::Saved);
            }
            Msg::Approvals(a) => {
                s.approvals = a;
                return (Task::none(), Effect::Saved);
            }
            Msg::NotesFolder(v) => self.notes_folder = v,
            Msg::CommitNotesFolder => {
                let name = crate::notes::sanitize_name(&self.notes_folder);
                self.notes_folder = name.clone();
                if name != s.notes_folder {
                    s.notes_folder = name;
                    return (Task::none(), Effect::Saved);
                }
            }
            Msg::ActivityLog(on) => {
                s.activity_log = on;
                return (Task::none(), Effect::Saved);
            }
            Msg::RememberStep(on) => {
                s.remember_step = on;
                return (Task::none(), Effect::Saved);
            }
            Msg::WebAccess(on) => {
                s.web_access = on;
                return (Task::none(), Effect::Saved);
            }
            Msg::PreloadModel(on) => {
                s.preload_model = on;
                return (Task::none(), Effect::Saved);
            }
            Msg::UseGpu(on) => {
                s.use_gpu = on;
                // Turning the card on again tries it for every model, including ones it failed before.
                if on {
                    s.gpu_failed.clear();
                    if let Some(host) = ctx.host {
                        host.llama.forget_gpu_failures();
                    }
                }
                return (Task::none(), Effect::Saved);
            }
            Msg::ModelsFromDisk(on) => {
                s.models_from_disk = on;
                return (Task::none(), Effect::Saved);
            }
            Msg::CompactContext(on) => {
                s.compact_context = on;
                // A saved prompt holds the context in the precision it was read with, so none can be restored.
                crate::llama::forget_all_slots();
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::Predict(model, choice) => {
                // Off is stored too, since a model left out uses its own prediction layers when it has them.
                s.speculation.insert(model.clone(), choice.value);
                if let Some(host) = ctx.host {
                    host.llama.forget_spec_failure(&model);
                }
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::CudaDownload => {
                let Some(Some(gpu)) = cuda::detected() else { return (Task::none(), Effect::None) };
                let stop = CancellationToken::new();
                self.cuda_cancel = Some(stop.clone());
                self.cuda_progress = Some((0, gpu.download_size()));
                self.cuda_error = None;
                let stream = iced::stream::channel(16, async move |mut out: iced::futures::channel::mpsc::Sender<Message>| {
                    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(u64, u64)>();
                    let job = tokio::spawn(async move { cuda::install(&gpu, &stop, |done, total| {
                        let _ = tx.send((done, total));
                    })
                    .await });
                    let mut job = std::pin::pin!(job);
                    let mut shown = u64::MAX;
                    let result = loop {
                        tokio::select! {
                            Some((done, total)) = rx.recv() => {
                                let percent = done * 100 / total.max(1);
                                if percent != shown {
                                    shown = percent;
                                    let _ = iced::futures::SinkExt::send(&mut out, Message::Settings(Msg::CudaProgress(done, total))).await;
                                }
                            }
                            r = &mut job => break r,
                        }
                    };
                    let result = match result {
                        Ok(r) => r.map_err(|e| if crate::util::is_cancelled(&e) { String::new() } else { format!("{e:#}") }),
                        Err(e) => Err(e.to_string()),
                    };
                    let _ = iced::futures::SinkExt::send(&mut out, Message::Settings(Msg::CudaDone(result))).await;
                });
                return (Task::run(stream, std::convert::identity), Effect::None);
            }
            Msg::CudaProgress(done, total) => self.cuda_progress = Some((done, total)),
            Msg::CudaDone(result) => {
                self.cuda_progress = None;
                self.cuda_cancel = None;
                self.cuda_error = result.err().filter(|e| !e.is_empty());
            }
            Msg::CudaCancel => {
                if let Some(stop) = &self.cuda_cancel {
                    stop.cancel();
                }
            }
            Msg::CudaRemove => {
                let llama = ctx.host.map(|h| h.llama.clone());
                // The server may be running from the CUDA folder, so it stops before the folder goes.
                return (
                    Task::perform(
                        async move {
                            if let Some(l) = llama {
                                l.stop().await;
                            }
                            cuda::remove().map_err(|e| format!("{e:#}"))
                        },
                        |r| Message::Settings(Msg::CudaDone(r)),
                    ),
                    Effect::None,
                );
            }
            Msg::KeepAlive(k) => {
                s.keep_alive_minutes = k.0;
                return (Task::none(), Effect::Saved);
            }
            Msg::ModelsDir(v) => self.models_dir = v,
            Msg::CommitModelsDir => {
                s.models_dir = self.models_dir.trim().to_string();
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::BrowseModelsDir => {
                return (
                    Task::perform(
                        async { rfd::AsyncFileDialog::new().set_title(tr("Choose the models folder")).pick_folder().await.map(|h| h.path().to_path_buf()) },
                        |p| Message::Settings(Msg::ModelsDirPicked(p)),
                    ),
                    Effect::None,
                );
            }
            Msg::ModelsDirPicked(Some(p)) => {
                self.models_dir = p.to_string_lossy().into_owned();
                s.models_dir = self.models_dir.clone();
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::ModelsDirPicked(None) => {}
            Msg::ServerPath(v) => self.server_path = v,
            Msg::CommitServerPath => {
                s.llama_server_path = self.server_path.trim().to_string();
                return (Task::none(), Effect::Saved);
            }
            Msg::Context(model, c) => {
                s.context_sizes.insert(model, c.0);
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::Download(i) => return (Task::done(Message::Setup(setup::Msg::StartOne(i))), Effect::None),
            Msg::CustomSpec(v) => self.custom_spec = v,
            Msg::DownloadCustom => {
                let spec = self.custom_spec.trim().to_string();
                if spec.is_empty() {
                    return (Task::none(), Effect::None);
                }
                self.custom_spec.clear();
                return (Task::done(Message::Setup(setup::Msg::StartCustom(spec))), Effect::None);
            }
            Msg::DownloadSize(spec) => return (Task::done(Message::Setup(setup::Msg::StartCustom(spec))), Effect::None),
            Msg::AddModelFile => {
                return (
                    Task::perform(
                        async { rfd::AsyncFileDialog::new().set_title(tr("Choose a model file")).add_filter("GGUF", &["gguf"]).pick_file().await.map(|h| h.path().to_path_buf()) },
                        |p| Message::Settings(Msg::ModelFilePicked(p)),
                    ),
                    Effect::None,
                );
            }
            Msg::ModelFilePicked(Some(path)) => {
                // Any part of a split model stands for its first part, which is the one llama.cpp opens.
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let first = regex::Regex::new(r"(?i)-\d{5}(-of-\d{5}\.gguf)$").unwrap().replace(&name, "-00001$1").into_owned();
                let path = path.with_file_name(first);
                if crate::llama::added_model(&path).is_none() {
                    return (Task::none(), Effect::Toast(tr("That file is not a model Scoobert can run. Choose a GGUF model file, not its image projector.").into()));
                }
                let entry = path.to_string_lossy().into_owned();
                if !s.model_files.contains(&entry) {
                    s.model_files.push(entry);
                }
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::ModelFilePicked(None) => {}
            Msg::Learning(name, on) => {
                if on {
                    s.learning.insert(name.clone(), 2.0);
                } else {
                    s.learning.remove(&name);
                }
                // Prompts saved with or without the learned direction no longer match how the model runs.
                crate::llama::forget_slots(&name);
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::LearnStrength(name, strength) => {
                s.learning.insert(name.clone(), strength.0);
                crate::llama::forget_slots(&name);
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::ApplyLearning(name) => {
                let Some(host) = ctx.host.cloned() else { return (Task::none(), Effect::None) };
                let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                self.learn_run = Some(LearnRun { model: name.clone(), done: 0, total: 1, cancel: cancel.clone() });
                let stream = iced::stream::channel(16, async move |mut out: iced::futures::channel::mpsc::Sender<Message>| {
                    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(u64, u64)>();
                    let model = name.clone();
                    let job = tokio::spawn(async move {
                        host.llama
                            .learn(&model, &cancel, move |p| {
                                let _ = tx.send((p.done, p.total));
                            })
                            .await
                    });
                    let mut job = std::pin::pin!(job);
                    let result = loop {
                        tokio::select! {
                            Some((done, total)) = rx.recv() => {
                                let _ = iced::futures::SinkExt::send(&mut out, Message::Settings(Msg::LearnProgress(done, total))).await;
                            }
                            r = &mut job => break r,
                        }
                    };
                    let result = match result {
                        Ok(r) => r.map_err(|e| if crate::util::is_cancelled(&e) { String::new() } else { format!("{e:#}") }),
                        Err(e) => Err(e.to_string()),
                    };
                    let _ = iced::futures::SinkExt::send(&mut out, Message::Settings(Msg::LearnDone(name, result))).await;
                });
                return (Task::run(stream, std::convert::identity), Effect::None);
            }
            Msg::LearnProgress(done, total) => {
                if let Some(run) = &mut self.learn_run {
                    (run.done, run.total) = (done, total);
                }
            }
            Msg::LearnDone(name, result) => {
                self.learn_run = None;
                self.refresh_learned(ctx.host, s);
                return match result {
                    Ok(()) => {
                        // Learning that was off turns on, since applying the ratings is asking for it.
                        s.learning.entry(name.clone()).or_insert(2.0);
                        (Task::none(), Effect::Toast(trf("{model} now leans toward the replies you rated good.", &[("model", &name)])))
                    }
                    Err(e) if e.is_empty() => (Task::none(), Effect::None),
                    Err(e) => (Task::none(), Effect::Toast(e)),
                };
            }
            Msg::LearnCancel => {
                if let Some(run) = &self.learn_run {
                    run.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
            Msg::ResetLearning(name) => {
                crate::llama::learn::reset(&s.models_dir(), &name);
                crate::llama::forget_slots(&name);
                self.refresh_learned(ctx.host, s);
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::DeleteModel { name, size, variants } => {
                return (Task::done(Message::AskConfirm(super::Confirm::DeleteModel { name, size, variants })), Effect::None);
            }
            Msg::ForgetModelFile(entry) => {
                // The file stays, and the prompts Scoobert saved for it go.
                if let Some(m) = ctx.host.map(|h| h.models()).unwrap_or_default().into_iter().find(|m| m.added.as_ref() == Some(&entry)) {
                    crate::llama::forget_saved(&m.name);
                }
                s.model_files.retain(|f| *f != entry);
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::CancelDownload => return (Task::done(Message::Setup(setup::Msg::Cancel)), Effect::None),
            Msg::RevealModels => {
                let dir = s.models_dir();
                let _ = std::fs::create_dir_all(&dir);
                let _ = opener::reveal(dir);
            }
            Msg::Provider(ProviderChoice(p)) => return (self.select_provider(p, ctx.host), Effect::None),
            Msg::KeyInput(v) => self.key_input = v,
            Msg::SaveKey => {
                let (Some(host), Some(p)) = (ctx.host, &self.provider) else { return (Task::none(), Effect::None) };
                let key = self.key_input.trim().to_string();
                if key.is_empty() {
                    return (Task::none(), Effect::None);
                }
                let note = host.set_key(&p.id, Some(&key));
                self.has_key = true;
                self.key_input.clear();
                let effect = note.map(Effect::Toast).unwrap_or(Effect::ModelsChanged);
                return (Task::done(Message::Settings(Msg::ListModels)), effect);
            }
            Msg::RemoveKey => {
                let (Some(host), Some(p)) = (ctx.host, &self.provider) else { return (Task::none(), Effect::None) };
                self.has_key = false;
                let effect = host.set_key(&p.id, None).map(Effect::Toast).unwrap_or(Effect::ModelsChanged);
                return (Task::none(), effect);
            }
            Msg::ListModels => {
                let (Some(host), Some(p)) = (ctx.host.cloned(), self.provider.clone()) else { return (Task::none(), Effect::None) };
                self.listing = true;
                self.error = None;
                let key = host.key(&p.id);
                let id = p.id.clone();
                return (
                    Task::perform(
                        async move { providers::list_models(host.http(), &p, key.as_deref()).await.map_err(|e| format!("{e:#}")) },
                        move |r| Message::Settings(Msg::Listed(id.clone(), r)),
                    ),
                    Effect::None,
                );
            }
            Msg::Listed(id, result) => {
                if self.provider.as_ref().is_some_and(|p| p.id == id) {
                    self.listing = false;
                    match result {
                        Ok(list) => self.available = list,
                        Err(e) => self.error = Some(e),
                    }
                }
            }
            Msg::Filter(f) => self.filter = f,
            Msg::ToggleModel(m, on) => {
                let reference = m.reference();
                s.hosted_models.retain(|h| h.reference() != reference);
                if on {
                    s.hosted_models.push(m);
                }
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::RemoveHosted(reference) => {
                s.hosted_models.retain(|h| h.reference() != reference);
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::CustomName(v) => self.custom_name = v,
            Msg::CustomUrl(v) => self.custom_url = v,
            Msg::AddCustom => {
                let url = self.custom_url.trim().trim_end_matches('/').to_string();
                let name = self.custom_name.trim().to_string();
                if name.is_empty() || !(url.starts_with("https://") || url.starts_with("http://")) {
                    self.error = Some(tr("Give the provider a name and an address that starts with https:// or http://.").into());
                    return (Task::none(), Effect::None);
                }
                let id = format!("custom-{}", crate::util::random_hex(4));
                let provider = CustomProvider { id: id.clone(), name, base_url: url };
                s.custom_providers.push(provider);
                self.custom_name.clear();
                self.custom_url.clear();
                let p = providers::find(&id, &s.custom_providers);
                let task = p.map(|p| self.select_provider(p, ctx.host)).unwrap_or_else(Task::none);
                return (task, Effect::ModelsChanged);
            }
            Msg::RemoveCustom(id) => {
                s.custom_providers.retain(|c| c.id != id);
                s.hosted_models.retain(|h| h.provider != id);
                if let Some(host) = ctx.host {
                    let _ = host.set_key(&id, None);
                }
                self.provider = providers::all(&s.custom_providers).into_iter().next();
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::OpenUrl(url) => return (Task::done(Message::OpenUrl(url)), Effect::None),
            Msg::Uninstall => {
                if let Some(uninstaller) = crate::paths::uninstaller() {
                    match std::process::Command::new(&uninstaller).spawn() {
                        Ok(_) => return (Task::done(Message::Quit), Effect::None),
                        Err(e) => return (Task::none(), Effect::Toast(trf("Could not start the uninstaller: {error}", &[("error", &e)]))),
                    }
                }
            }
            Msg::CheckUpdates => return (Task::done(Message::CheckUpdates), Effect::None),
            Msg::ShowFile(path) => {
                let _ = opener::reveal(path);
            }
            Msg::OpenLab(name) => {
                let Some(model) = ctx.host.and_then(|h| h.llama.models().into_iter().find(|m| m.name == name)) else {
                    return (Task::none(), Effect::Toast(tr("That model is no longer in the models folder.").into()));
                };
                let (lab, task) = super::lab::Lab::open(model);
                self.lab = Some(lab);
                return (Task::batch([task, operation::snap_to(PAGE_ID, RelativeOffset::START)]), Effect::None);
            }
            Msg::Lab(m) => {
                let Some(lab) = &mut self.lab else { return (Task::none(), Effect::None) };
                let (closed, task, effect) = lab.update(m, ctx.host);
                if closed {
                    self.lab = None;
                }
                return (task, effect);
            }
            Msg::LabDeleted => return (Task::none(), Effect::ModelsChanged),
            Msg::LabFailed(e) => return (Task::none(), Effect::Toast(trf("The variant could not be moved to the trash: {error}", &[("error", &e)]))),
            Msg::OpenPrompt(model, label) => {
                let mut content = text_editor::Content::with_text(s.model_prompt(&model).unwrap_or(DEFAULT_PROMPT));
                content.perform(text_editor::Action::Move(text_editor::Motion::DocumentStart));
                self.prompt = Some(PromptPage { model, label, content });
                return (operation::snap_to(PAGE_ID, RelativeOffset::START), Effect::None);
            }
            Msg::PromptEdit(action) => {
                if let Some(p) = &mut self.prompt {
                    p.content.perform(action);
                }
            }
            Msg::ResetPrompt => {
                if let Some(p) = &mut self.prompt {
                    p.content = text_editor::Content::with_text(DEFAULT_PROMPT);
                }
            }
            Msg::SavePrompt => {
                let Some(p) = self.prompt.take() else { return (Task::none(), Effect::None) };
                let written = p.content.text();
                let written = written.trim_end();
                if written.trim().is_empty() || written == DEFAULT_PROMPT {
                    s.model_prompts.remove(&p.model);
                } else {
                    s.model_prompts.insert(p.model, written.to_string());
                }
                return (Task::none(), Effect::Saved);
            }
            Msg::ClosePrompt => self.prompt = None,
            Msg::Close => {
                // Closing Settings stops a variant that is being made, and its half-written files are removed.
                if let Some(lab) = &mut self.lab
                    && lab.is_running()
                {
                    let _ = lab.update(super::lab::Msg::Cancel, ctx.host);
                }
                return (Task::done(Message::CloseModal), Effect::None);
            }
        }
        (Task::none(), Effect::None)
    }

    pub fn view<'a>(&'a self, ctx: ViewCtx<'a>) -> Element<'a, Msg> {
        let nav = |label: &'static str, section: Section| {
            button(text(label).size(14))
                .width(Fill)
                .padding([8, 12])
                .style(theme::list_item(self.section == section))
                .on_press(Msg::Section(section))
        };
        // The author's wording, kept as written in every language.
        let support = iced::widget::rich_text![
            iced::widget::span("Support my free games and software!: "),
            iced::widget::span(SUPPORT_URL).link(SUPPORT_URL.to_string()).underline(true),
        ]
        .size(12)
        .on_link_click(Msg::OpenUrl);
        let sidebar = column![
            text(tr("Settings")).size(18).font(fonts::ui_semibold()),
            space().height(8),
            nav(tr("General"), Section::General),
            nav(tr("Models on this computer"), Section::Local),
            nav(tr("Hosted models"), Section::Hosted),
            space::vertical(),
            support,
        ]
        .spacing(4)
        .width(210)
        .height(Fill)
        .padding(16);
        let content = match (&self.prompt, self.section) {
            (Some(p), _) => prompt_view(p),
            (None, Section::General) => self.general(ctx.state, ctx.isolation),
            (None, Section::Local) => self.local(ctx),
            (None, Section::Hosted) => self.hosted(ctx.state),
        };
        let close = button(icon(Icon::Close, 16.0)).padding(6).style(theme::ghost).on_press(Msg::Close);
        let page = scrollable(container(content).padding(iced::Padding { top: 0.0, right: 24.0, bottom: 24.0, left: 8.0 })).id(PAGE_ID).height(Fill).style(theme::scrollbar);
        let body = column![row![space::horizontal(), close], page];
        container(row![sidebar, rule::vertical(1).style(theme::divider), body.width(Fill)]).height(600).into()
    }

    fn general<'a>(&'a self, state: &'a State, isolation: &'a str) -> Element<'a, Msg> {
        let s = &state.settings;
        let mode_help = match s.approvals {
            Approvals::Ask => tr("Scoobert asks before it edits a file or runs a command. Edits to the notes folder never ask.").to_string(),
            Approvals::Project => trf(
                "For leaving a task running. Edits inside the project run without asking, and edits elsewhere are refused. {isolation} Every command's memory is capped.",
                &[("isolation", &isolation)],
            ),
            Approvals::Auto => tr("Everything runs without asking, including commands that download or install software.").to_string(),
        };
        column![
            heading(tr("General")),
            column![
                field(
                    tr("When Scoobert makes changes"),
                    None,
                    pick_list(Approvals::ALL, Some(s.approvals), Msg::Approvals).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu).into(),
                ),
                text(mode_help).size(12).style(theme::muted),
            ]
            .spacing(4),
            field(
                tr("Language"),
                Some(tr("Scoobert's menus and labels, and the language the model writes in.")),
                pick_list(crate::i18n::LANGUAGES.iter().collect::<Vec<_>>(), Some(crate::i18n::current()), Msg::Language)
                    .padding([5, 10])
                    .text_size(13)
                    .style(theme::select)
                    .menu_style(theme::menu)
                    .into(),
            ),
            field(
                tr("Appearance"),
                None,
                pick_list(ThemeChoice::ALL, Some(s.theme), Msg::Theme).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu).into(),
            ),
            field(
                tr("Notes folder"),
                Some(tr("The folder inside each project where Scoobert keeps its notes.")),
                text_input("Notes", &self.notes_folder)
                    .on_input(Msg::NotesFolder)
                    .on_submit(Msg::CommitNotesFolder)
                    .size(13)
                    .padding([6, 10])
                    .width(220)
                    .style(theme::input)
                    .into(),
            ),
            switch_row(
                tr("Log finished tasks in today's note"),
                tr("Adds what changed to the Daily folder after a task that edited files."),
                s.activity_log,
                Msg::ActivityLog
            ),
            switch_row(
                tr("Ask the model what to remember"),
                tr("After a task that changed several files or stated a preference, the model picks up to three facts, and Scoobert files them in the notes. With a local model this takes about a minute."),
                s.remember_step,
                Msg::RememberStep
            ),
            switch_row(
                tr("Let Scoobert search the web with DuckDuckGo"),
                tr("Off until you turn it on. When you ask it to research something, it searches DuckDuckGo and reads pages. Searches and page addresses leave your computer, even with a local model."),
                s.web_access,
                Msg::WebAccess
            ),
            version(),
            removal(),
        ]
        .spacing(18)
        .into()
    }

    fn local<'a>(&'a self, ctx: ViewCtx<'a>) -> Element<'a, Msg> {
        if let Some(lab) = &self.lab {
            return lab.view().map(Msg::Lab);
        }
        let s = &ctx.state.settings;
        let installed: Vec<&ModelOption> = ctx.models.iter().filter(|m| m.provider.is_empty()).collect();
        let mut list = Column::new().spacing(8);
        for m in &installed {
            let mut sizes: Vec<ContextChoice> = CONTEXT_SIZES.iter().map(|&c| ContextChoice(c)).collect();
            if !CONTEXT_SIZES.contains(&m.context) {
                sizes.push(ContextChoice(m.context));
            }
            let free = crate::sys::available_memory();
            let args: &[(&str, &dyn std::fmt::Display)] = &[("memory", &gb(m.memory_needed)), ("card", &gb(m.card_memory)), ("free", &gb(free))];
            let warn = match (m.memory_needed > free, m.card_memory > 0) {
                (true, true) => text(trf("Needs about {memory} of free memory besides {card} on the graphics card. {free} is free now.", args)).size(12).style(theme::warn_text),
                (true, false) => text(trf("Needs about {memory} of free memory; {free} is free now.", args)).size(12).style(theme::warn_text),
                (false, true) => text(trf("Needs about {memory} of free memory besides {card} on the graphics card.", args)).size(12).style(theme::muted),
                (false, false) => text(trf("Needs about {memory} of free memory.", args)).size(12).style(theme::muted),
            };
            let name = m.name.clone();
            let forget: Element<'a, Msg> = match &m.added {
                Some(entry) => button(text(tr("Remove from list")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::ForgetModelFile(entry.clone())).into(),
                None if m.deletable => button(icon(Icon::Trash, 14.0))
                    .padding([4, 6])
                    .style(theme::ghost)
                    .on_press(Msg::DeleteModel { name: m.name.clone(), size: m.size, variants: m.variants })
                    .into(),
                None => space().width(0).into(),
            };
            list = list.push(
                container(
                    row![
                        column![
                            text(m.label.clone()).size(14).font(fonts::ui_semibold()),
                            text(if m.vision { trf("{size}, reads images", &[("size", &gb(m.size))]) } else { gb(m.size) }).size(12).style(theme::muted),
                            warn,
                        ]
                        .spacing(2)
                        .width(Fill),
                        text(tr("Context")).size(12).style(theme::muted),
                        pick_list(sizes, Some(ContextChoice(m.context)), move |c| Msg::Context(name.clone(), c))
                            .padding([4, 8])
                            .text_size(12)
                            .style(theme::select)
                            .menu_style(theme::menu),
                        button(text(tr("System prompt")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::OpenPrompt(m.name.clone(), m.label.clone())),
                        button(text(tr("Model lab")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::OpenLab(m.name.clone())),
                        forget,
                    ]
                    .spacing(10)
                    .align_y(Alignment::Center),
                )
                .padding(12)
                .style(theme::card),
            );
            if m.learnable {
                list = list.push(predict_row(m, s, ctx.models));
                list = list.push(self.learning_row(m, s));
            }
        }
        if installed.is_empty() {
            list = list.push(text(tr("No models yet. Download one below.")).size(13).style(theme::muted));
        }

        let mut catalog = Column::new().spacing(8);
        for (i, m) in CATALOG.iter().enumerate() {
            let have = installed.iter().any(|x| x.name == m.name);
            let status: Element<'a, Msg> = if have {
                text(tr("Downloaded")).size(12).style(theme::accent_text).into()
            } else if let Some(d) = ctx.download.as_ref().filter(|d| d.job == Job::Catalog(i)) {
                let (done, total) = d.progress.as_ref().map(|p| (p.done, p.total.max(1))).unwrap_or((0, 1));
                row![
                    progress_bar(0.0..=total as f32, done as f32).length(120).girth(6).style(theme::meter),
                    text(trf("{done} of {total}", &[("done", &gb(done)), ("total", &gb(total))])).size(12).style(theme::muted),
                    button(text(tr("Stop")).size(12)).padding([3, 10]).style(theme::secondary).on_press(Msg::CancelDownload),
                ]
                .spacing(8)
                .align_y(Alignment::Center)
                .into()
            } else if ctx.queue.contains(&Job::Catalog(i)) {
                text(tr("Waiting...")).size(12).style(theme::muted).into()
            } else {
                button(row![icon(Icon::Download, 14.0), text(trf("Download {size}", &[("size", &gb(m.download_bytes))])).size(12)].spacing(6).align_y(Alignment::Center))
                    .padding([4, 10])
                    .style(theme::secondary)
                    .on_press(Msg::Download(i))
                    .into()
            };
            let fit = if setup::fits(i) {
                text(trf("Needs about {memory} of free memory.", &[("memory", &gb(memory_needed(m)))])).size(12).style(theme::muted)
            } else {
                text(trf("Needs about {memory} of free memory, more than this computer has.", &[("memory", &gb(memory_needed(m)))])).size(12).style(theme::warn_text)
            };
            catalog = catalog.push(
                container(
                    row![
                        column![
                            row![text(tr(m.label)).size(14).font(fonts::ui_semibold()), container(text(tr(m.tier)).size(11)).padding([1, 6]).style(theme::chip)]
                                .spacing(8)
                                .align_y(Alignment::Center),
                            text(tr(m.summary)).size(12).style(theme::muted),
                            fit,
                        ]
                        .spacing(3)
                        .width(Fill),
                        status,
                    ]
                    .spacing(12)
                    .align_y(Alignment::Center),
                )
                .padding(12)
                .style(theme::card),
            );
        }

        column![
            heading(tr("Models on this computer")),
            list,
            subheading(tr("Download")),
            catalog,
            subheading(tr("Download another model")),
            text(tr("Paste the address of any GGUF model on Hugging Face, or write owner/repository. Add a size after a colon to choose one, as in :Q4_K_M.")).size(12).style(theme::muted),
            row![
                text_input("https://huggingface.co/unsloth/Qwen3.5-9B-GGUF", &self.custom_spec)
                    .on_input(Msg::CustomSpec)
                    .on_submit(Msg::DownloadCustom)
                    .size(13)
                    .padding([6, 10])
                    .style(theme::input),
                button(text(tr("Download")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::DownloadCustom),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            custom_progress(ctx.download),
            size_choices(ctx.choices),
            subheading(tr("Add a model file")),
            text(tr("Use a GGUF model you downloaded yourself. Scoobert runs it from its folder, and an image projector beside it lets it read images.")).size(12).style(theme::muted),
            button(text(tr("Choose a model file")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::AddModelFile),
            subheading(tr("Where models are stored")),
            row![
                text_input(&crate::paths::display(&crate::paths::get().default_models_dir()), &self.models_dir)
                    .on_input(Msg::ModelsDir)
                    .on_submit(Msg::CommitModelsDir)
                    .size(13)
                    .padding([6, 10])
                    .style(theme::input),
                button(text(tr("Browse")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::BrowseModelsDir),
                button(icon(Icon::External, 15.0)).padding(6).style(theme::ghost).on_press(Msg::RevealModels),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            field(
                tr("Unload the model after"),
                Some(tr("An idle model still holds its memory. Saved caches make the next load quick.")),
                pick_list(&KEEP_ALIVE[..], Some(KeepAlive(s.keep_alive_minutes)), Msg::KeepAlive)
                    .padding([5, 10])
                    .text_size(13)
                    .style(theme::select)
                    .menu_style(theme::menu)
                    .into(),
            ),
            switch_row(
                tr("Load the model when Scoobert starts"),
                tr("The first reply starts sooner, but the model holds its memory from the start until it has been idle for the time above."),
                s.preload_model,
                Msg::PreloadModel
            ),
            switch_row(
                tr("Use the graphics card"),
                tr("Runs as much of the model as fits on the graphics card, which is much faster with a dedicated card. A model the card cannot load runs on the processor instead. Applies the next time a model loads."),
                s.use_gpu,
                Msg::UseGpu
            ),
            self.nvidia(),
            switch_row(
                tr("Load models larger than memory from disk"),
                tr("Lets a model that does not fit in free memory keep part of its weights on the disk and read them as virtual memory while it works. It runs much slower: a mixture-of-experts model manages a few words per second from an SSD, and other models can take several seconds per word."),
                s.models_from_disk,
                Msg::ModelsFromDisk
            ),
            switch_row(
                tr("Compact context memory"),
                tr("Keeps the conversation's context at 8 bits instead of 16, which halves the memory it takes, so more of a model fits on the graphics card or in memory. Replies change very little. Saved conversations are read again once after you change it."),
                s.compact_context,
                Msg::CompactContext
            ),
            field(
                tr("llama-server program"),
                Some(tr("Leave empty to use the copy that comes with Scoobert.")),
                text_input(tr("Bundled"), &self.server_path)
                    .on_input(Msg::ServerPath)
                    .on_submit(Msg::CommitServerPath)
                    .size(13)
                    .padding([6, 10])
                    .style(theme::input)
                    .into(),
            ),
        ]
        .spacing(14)
        .into()
    }

    /// NVIDIA support for a computer with an NVIDIA card: the offer to download it, its progress, or what is installed.
    fn nvidia(&self) -> Element<'_, Msg> {
        let Some(Some(gpu)) = cuda::detected() else { return space().into() };
        let mut col = Column::new().spacing(6);
        if let Some(version) = cuda::installed_version() {
            col = col.push(
                text(trf("NVIDIA support with CUDA {version} is installed. Scoobert uses it while Use the graphics card is on, from the next time a model loads.", &[("version", &version)]))
                    .size(12)
                    .style(theme::muted),
            );
            col = col.push(button(text(tr("Remove NVIDIA support")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::CudaRemove));
        } else if let Some((done, total)) = self.cuda_progress {
            col = col.push(
                row![
                    progress_bar(0.0..=total.max(1) as f32, done as f32).girth(6).style(theme::meter),
                    text(format!("{}%", done * 100 / total.max(1))).size(12).width(44),
                    button(text(tr("Stop")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::CudaCancel),
                ]
                .spacing(10)
                .align_y(Alignment::Center),
            );
        } else if gpu.cuda().is_none() {
            col = col.push(
                text(trf(
                    "NVIDIA support needs driver version {min} or later, and this computer has {driver}. Update the driver from NVIDIA's website first.",
                    &[("min", &cuda::MIN_DRIVER), ("driver", &gpu.driver)],
                ))
                .size(12)
                .style(theme::muted),
            );
        } else {
            col = col.push(
                text(trf("NVIDIA support runs models through CUDA on your {gpu}, which usually reads prompts much faster than the graphics support Scoobert includes.", &[("gpu", &gpu.name)]))
                    .size(12)
                    .style(theme::muted),
            );
            col = col.push(
                button(text(trf("Download NVIDIA support ({size})", &[("size", &gb(gpu.download_size()))])).size(12))
                    .padding([4, 10])
                    .style(theme::secondary)
                    .on_press(Msg::CudaDownload),
            );
        }
        if let Some(e) = &self.cuda_error {
            col = col.push(text(e.clone()).size(12).style(theme::warn_text));
        }
        col.into()
    }

    fn hosted<'a>(&'a self, state: &'a State) -> Element<'a, Msg> {
        let s = &state.settings;
        let all = providers::all(&s.custom_providers);
        let choices: Vec<ProviderChoice> = all.iter().cloned().map(ProviderChoice).collect();
        let selected = self.provider.clone().map(ProviderChoice);
        let mut col = column![
            heading(tr("Hosted models")),
            text(tr("Hosted models run on the provider's servers, so Scoobert sends them your conversations and the files it reads. Keys are kept in the system credential store."))
                .size(13)
                .style(theme::muted),
            row![
                text(tr("Provider")).size(13).width(90),
                pick_list(choices, selected, Msg::Provider).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu).width(Length::Fixed(260.0)),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        ]
        .spacing(14);
        if let Some(p) = &self.provider {
            let key_row: Element<'a, Msg> = if self.has_key {
                row![
                    icons_ok(),
                    text(tr("Key saved")).size(13),
                    space::horizontal(),
                    button(text(tr("Remove key")).size(13)).padding([5, 12]).style(theme::danger).on_press(Msg::RemoveKey),
                ]
                .spacing(8)
                .align_y(Alignment::Center)
                .into()
            } else {
                let mut r = row![
                    text_input(tr("Paste an API key"), &self.key_input)
                        .secure(true)
                        .on_input(Msg::KeyInput)
                        .on_submit(Msg::SaveKey)
                        .size(13)
                        .padding([6, 10])
                        .style(theme::input),
                    button(text(tr("Save key")).size(13)).padding([6, 12]).style(theme::primary).on_press_maybe((!self.key_input.trim().is_empty()).then_some(Msg::SaveKey)),
                ]
                .spacing(8)
                .align_y(Alignment::Center);
                if !p.key_url.is_empty() {
                    r = r.push(button(text(tr("Get a key")).size(13)).style(theme::link).on_press(Msg::OpenUrl(p.key_url.clone())));
                }
                r.into()
            };
            col = col.push(row![text(tr("API key")).size(13).width(90), key_row].spacing(10).align_y(Alignment::Center));
            if p.id.starts_with("custom-") {
                col = col.push(
                    row![
                        text(p.base_url.clone()).size(12).style(theme::muted).font(fonts::mono()),
                        space::horizontal(),
                        button(text(tr("Remove this provider")).size(12)).padding([4, 10]).style(theme::danger).on_press(Msg::RemoveCustom(p.id.clone())),
                    ]
                    .align_y(Alignment::Center),
                );
            }
            col = col.push(self.model_list(state, p));
        }
        let added: Vec<&HostedModel> = s.hosted_models.iter().collect();
        if !added.is_empty() {
            let mut list = Column::new().spacing(4);
            for m in added {
                let provider = all.iter().find(|p| p.id == m.provider).map(|p| p.name.clone()).unwrap_or_else(|| m.provider.clone());
                list = list.push(
                    row![
                        text(format!("{} ({provider})", m.name)).size(13).width(Fill),
                        button(text(tr("System prompt")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::OpenPrompt(m.reference(), m.name.clone())),
                        button(icon(Icon::Close, 14.0)).padding(4).style(theme::ghost).on_press(Msg::RemoveHosted(m.reference())),
                    ]
                    .spacing(6)
                    .align_y(Alignment::Center),
                );
            }
            col = col.push(subheading(tr("In the model menu")));
            col = col.push(list);
        }
        col = col.push(subheading(tr("Add a provider by address")));
        col = col.push(text(tr("Any server that accepts OpenAI-style chat requests, such as a model server on another computer.")).size(12).style(theme::muted));
        col = col.push(
            row![
                text_input(tr("Name"), &self.custom_name).on_input(Msg::CustomName).size(13).padding([6, 10]).width(160).style(theme::input),
                text_input("https://example.com/v1", &self.custom_url).on_input(Msg::CustomUrl).on_submit(Msg::AddCustom).size(13).padding([6, 10]).style(theme::input),
                button(text(tr("Add")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::AddCustom),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        );
        col.into()
    }

    fn model_list<'a>(&'a self, state: &'a State, p: &'a Provider) -> Element<'a, Msg> {
        if let Some(e) = &self.error {
            return container(text(e.clone()).size(13)).padding([8, 12]).width(Fill).style(theme::error_box).into();
        }
        if self.listing {
            return text(tr("Loading the model list...")).size(13).style(theme::muted).into();
        }
        if self.available.is_empty() {
            let hint = if self.has_key { tr("Refresh the list to see this provider's models.") } else { tr("Save a key to see this provider's models.") };
            return row![
                text(hint).size(13).style(theme::muted),
                button(icon(Icon::Refresh, 14.0)).padding(4).style(theme::ghost).on_press(Msg::ListModels),
            ]
            .spacing(6)
            .align_y(Alignment::Center)
            .into();
        }
        let filter = self.filter.to_lowercase();
        let mut list = Column::new().spacing(2);
        let mut shown = 0;
        for m in &self.available {
            if !filter.is_empty() && !m.name.to_lowercase().contains(&filter) && !m.id.to_lowercase().contains(&filter) {
                continue;
            }
            shown += 1;
            if shown > 200 {
                break;
            }
            let on = state.settings.hosted_models.iter().any(|h| h.provider == p.id && h.id == m.id);
            let model = m.clone();
            let mut tags = vec![trf("{size}K context", &[("size", &(m.context / 1000))])];
            if m.vision {
                tags.push(tr("images").into());
            }
            if m.reasoning {
                tags.push(tr("thinking").into());
            }
            list = list.push(
                row![
                    checkbox(on).label(m.name.clone()).on_toggle(move |v| Msg::ToggleModel(model.clone(), v)).text_size(13).style(theme::check).width(Fill),
                    text(tags.join(", ")).size(12).style(theme::muted),
                ]
                .align_y(Alignment::Center),
            );
        }
        column![
            row![
                text_input(tr("Search models"), &self.filter).on_input(Msg::Filter).size(13).padding([6, 10]).style(theme::input),
                button(icon(Icon::Refresh, 14.0)).padding(6).style(theme::ghost).on_press(Msg::ListModels),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            text(tr("Tick the models to show in the model menu.")).size(12).style(theme::muted),
            container(scrollable(list.padding([4, 8])).style(theme::scrollbar)).height(260).style(theme::card),
        ]
        .spacing(8)
        .into()
    }
}

impl Panel {
    /// Under a model's card: learning from ratings, how many there are, and building or resetting what it learned.
    fn learning_row<'a>(&'a self, m: &ModelOption, s: &crate::store::Settings) -> Element<'a, Msg> {
        let summary = self.learned.get(&m.name).copied().unwrap_or_default();
        let strength = s.learning.get(&m.name).copied();
        let name = m.name.clone();
        let mut line = row![
            text(tr("Learn from ratings")).size(13),
            super::switch::switch(strength.is_some(), move |on| Msg::Learning(name.clone(), on)),
            text(trf("{good} good and {bad} bad ratings", &[("good", &summary.good), ("bad", &summary.bad)])).size(12).style(theme::muted),
            space::horizontal(),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        if let Some(value) = strength {
            let name = m.name.clone();
            line = line.push(
                pick_list(LEARN_STRENGTHS, Some(LearnStrength(value)), move |c| Msg::LearnStrength(name.clone(), c))
                    .padding([4, 8])
                    .text_size(12)
                    .style(theme::select)
                    .menu_style(theme::menu),
            );
        }
        match &self.learn_run {
            Some(run) if run.model == m.name => {
                line = line.push(progress_bar(0.0..=run.total.max(1) as f32, run.done as f32).length(100).girth(6).style(theme::meter));
                line = line.push(button(text(tr("Stop")).size(12)).padding([3, 10]).style(theme::secondary).on_press(Msg::LearnCancel));
            }
            _ => {
                let label = if summary.learned { trf("Apply {count} new ratings", &[("count", &summary.new)]) } else { tr("Apply ratings").to_string() };
                let can = summary.can_learn() && self.learn_run.is_none();
                line = line.push(button(text(label).size(12)).padding([4, 10]).style(theme::secondary).on_press_maybe(can.then(|| Msg::ApplyLearning(m.name.clone()))));
                if summary.good + summary.bad > 0 || summary.learned {
                    line = line.push(button(text(tr("Reset")).size(12)).padding([4, 10]).style(theme::ghost).on_press(Msg::ResetLearning(m.name.clone())));
                }
            }
        }
        let help = if summary.good < crate::llama::learn::MIN_EACH || summary.bad < crate::llama::learn::MIN_EACH {
            trf("Rate replies with Good and Bad, retry in the conversation. Learning needs at least {count} of each.", &[("count", &crate::llama::learn::MIN_EACH)])
        } else {
            tr("Applying unloads the model for several minutes while Scoobert compares the good replies with the bad ones. It changes the model's tone and style, not what it knows.").to_string()
        };
        container(column![line, text(help).size(12).style(theme::muted)].spacing(4)).padding(iced::Padding { top: 0.0, right: 12.0, bottom: 4.0, left: 24.0 }).into()
    }

    /// Reads how many ratings each local model has, for its learning row.
    fn refresh_learned(&mut self, host: Option<&Arc<Host>>, settings: &crate::store::Settings) {
        let Some(h) = host else { return };
        let dir = settings.models_dir();
        self.learned = h.llama.models().into_iter().filter(|m| m.variant.is_none()).map(|m| (m.name.clone(), crate::llama::learn::summary(&dir, &m.name))).collect();
    }
}

/// A way for a model to predict words ahead: the setting's stored value, and its label.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictChoice {
    value: String,
    label: String,
}

impl std::fmt::Display for PredictChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label)
    }
}

/// Under a model's card: how it predicts words ahead, with only the ways its file and the other models allow. A draft
/// model must share the model's architecture, so the two read text the same way, and be much smaller to be quicker.
fn predict_row<'a>(m: &ModelOption, s: &crate::store::Settings, models: &[ModelOption]) -> Element<'a, Msg> {
    let choice = |value: &str, label: String| PredictChoice { value: value.into(), label };
    let mut choices = vec![choice("off", tr("Off").into())];
    if m.mtp {
        choices.push(choice("mtp", tr("Its own prediction layers").into()));
    }
    choices.push(choice("ngram", tr("Text already in the conversation").into()));
    for d in models.iter().filter(|d| d.learnable && d.name != m.name && !m.arch.is_empty() && d.arch == m.arch && d.size * 3 <= m.size) {
        choices.push(choice(&format!("draft:{}", d.name), trf("Draft with {model}", &[("model", &d.label)])));
    }
    let current = s.speculation.get(&m.name).cloned().unwrap_or_else(|| if m.mtp { "mtp".into() } else { "off".into() });
    let selected = choices.iter().find(|c| c.value == current).cloned().or_else(|| choices.first().cloned());
    let name = m.name.clone();
    let line = row![
        text(tr("Predict ahead")).size(13),
        pick_list(choices, selected, move |c| Msg::Predict(name.clone(), c)).padding([4, 8]).text_size(12).style(theme::select).menu_style(theme::menu),
        text(tr("The model checks several predicted words in one step and keeps the ones it agrees with, so it writes faster without changing what it writes.")).size(12).style(theme::muted).width(Fill),
    ]
    .spacing(10)
    .align_y(Alignment::Center);
    container(line).padding(iced::Padding { top: 0.0, right: 12.0, bottom: 0.0, left: 24.0 }).into()
}

/// How strongly a model leans toward what it learned from ratings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnStrength(pub f32);

const LEARN_STRENGTHS: [LearnStrength; 3] = [LearnStrength(1.0), LearnStrength(2.0), LearnStrength(crate::llama::learn::MAX_STRENGTH)];

impl std::fmt::Display for LearnStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(tr(if self.0 <= 1.0 {
            "Gentle"
        } else if self.0 <= 2.0 {
            "Moderate"
        } else {
            "Strong"
        }))
    }
}

/// Learning from ratings for one model, while the steering tool runs.
struct LearnRun {
    model: String,
    done: u64,
    total: u64,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

/// The sizes a repository offered when the download did not name one it has.
fn size_choices<'a>(choices: &'a Option<crate::llama::download::ChooseSize>) -> Element<'a, Msg> {
    let Some(c) = choices else { return space().into() };
    let mut sizes = row![].spacing(8);
    for size in &c.sizes {
        sizes = sizes.push(
            button(text(format!("{} · {}", size.label, gb(size.bytes))).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::DownloadSize(size.spec.clone())),
        );
    }
    column![text(c.message.clone()).size(12).style(theme::warn_text), sizes.wrap()].spacing(6).into()
}

fn custom_progress<'a>(download: &'a Option<Download>) -> Element<'a, Msg> {
    let Some(d) = download.as_ref() else { return space().into() };
    let Job::Custom(spec) = &d.job else { return space().into() };
    let (done, total) = d.progress.as_ref().map(|p| (p.done, p.total.max(1))).unwrap_or((0, 1));
    row![
        text(spec.clone()).size(12).font(fonts::mono()),
        progress_bar(0.0..=total as f32, done as f32).length(120).girth(6).style(theme::meter),
        text(trf("{done} of {total}", &[("done", &gb(done)), ("total", &gb(total))])).size(12).style(theme::muted),
        button(text(tr("Stop")).size(12)).padding([3, 10]).style(theme::secondary).on_press(Msg::CancelDownload),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

/// The running version, with a button that looks for a newer release when this build knows where releases are.
fn version<'a>() -> Element<'a, Msg> {
    let mut control = row![text(env!("CARGO_PKG_VERSION")).size(13).style(theme::muted)].spacing(12).align_y(Alignment::Center);
    if crate::update::repository().is_some() {
        control = control.push(button(text(tr("Check for updates")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::CheckUpdates));
    }
    field(tr("Version"), None, control.into())
}

/// How to remove this copy of Scoobert: its uninstaller, its folder when portable, or its AppImage file.
fn removal<'a>() -> Element<'a, Msg> {
    let (help, button_label, msg): (String, &str, Msg) = if crate::paths::uninstaller().is_some() {
        (
            tr("Removes Scoobert from this computer. You choose whether to also remove your conversations and downloaded models. Project notes always stay.").into(),
            tr("Uninstall Scoobert"),
            Msg::Uninstall,
        )
    } else if let Some(folder) = crate::paths::portable_folder() {
        (tr("This is a portable copy. To remove it, close Scoobert and delete its folder.").into(), tr("Show the folder"), Msg::ShowFile(folder))
    } else if let Some(appimage) = std::env::var_os("APPIMAGE").map(PathBuf::from) {
        (tr("To remove Scoobert, close it and delete its AppImage file.").into(), tr("Show the file"), Msg::ShowFile(appimage))
    } else {
        return space().into();
    };
    let control = button(text(button_label).size(13)).padding([6, 12]).style(theme::secondary).on_press(msg);
    column![field(tr("Remove Scoobert"), None, control.into()), text(help).size(12).style(theme::muted)].spacing(4).into()
}

fn icons_ok<'a>() -> Element<'a, Msg> {
    super::icons::tinted(Icon::Check, 16.0, |t| t.ok).into()
}

/// The page that edits one model's system prompt.
fn prompt_view(p: &PromptPage) -> Element<'_, Msg> {
    let back = button(row![icon(Icon::ArrowLeft, 14.0), text(tr("Back")).size(13)].spacing(4).align_y(Alignment::Center))
        .padding([4, 8])
        .style(theme::ghost)
        .on_press(Msg::ClosePrompt);
    let editor = text_editor(&p.content).on_action(Msg::PromptEdit).height(360).size(13).padding(12).font(fonts::mono()).style(theme::bare_editor);
    let changed = p.content.text().trim_end() != DEFAULT_PROMPT;
    column![
        back,
        text(trf("System prompt: {model}", &[("model", &p.label)])).size(20).font(fonts::ui_semibold()),
        text(tr("Every conversation on this model starts with these instructions. Scoobert fills in {notes_folder} with the notes folder's name and {shell_tool} with the name of its command tool. Removing the parts about tools, writing files, or notes makes the model use them worse.")).size(12).style(theme::muted),
        container(editor).width(Fill).style(theme::card),
        text(tr("A conversation that already started switches to the new prompt at its next message, and the model reads the whole conversation again then, which takes a while on a long one.")).size(12).style(theme::muted),
        row![
            button(text(tr("Save")).size(13)).padding([6, 16]).style(theme::primary).on_press(Msg::SavePrompt),
            button(text(tr("Reset to Scoobert's prompt")).size(13)).padding([6, 12]).style(theme::secondary).on_press_maybe(changed.then_some(Msg::ResetPrompt)),
        ]
        .spacing(8),
    ]
    .spacing(12)
    .into()
}

fn heading<'a>(label: &'static str) -> Element<'a, Msg> {
    text(label).size(18).font(fonts::ui_semibold()).into()
}

fn subheading<'a>(label: &'static str) -> Element<'a, Msg> {
    container(text(label).size(14).font(fonts::ui_semibold())).padding(iced::Padding { top: 8.0, ..Default::default() }).into()
}

fn field<'a>(label: &'static str, help: Option<&'static str>, control: Element<'a, Msg>) -> Element<'a, Msg> {
    let mut left = column![text(label).size(14)].spacing(2).width(Fill);
    if let Some(h) = help {
        left = left.push(text(h).size(12).style(theme::muted));
    }
    row![left, control].spacing(16).align_y(Alignment::Center).into()
}

fn switch_row<'a>(label: &'static str, help: &'static str, on: bool, msg: fn(bool) -> Msg) -> Element<'a, Msg> {
    field(label, Some(help), super::switch::switch(on, msg).into())
}
