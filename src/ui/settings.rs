//! The Settings dialog: general options, models on this computer, and hosted models with API keys.

use std::path::PathBuf;
use std::sync::Arc;

use iced::widget::{Column, button, checkbox, column, container, pick_list, progress_bar, row, rule, scrollable, space, text, text_input, toggler};
use iced::{Alignment, Element, Fill, Length, Task};

use super::Message;
use super::fonts;
use super::icons::{Icon, icon};
use super::setup::{self, Download, Job};
use super::theme;
use crate::agent::providers::{self, Provider};
use crate::agent::{Host, ModelOption};
use crate::llama::catalog::{CATALOG, memory_needed};
use crate::store::{Approvals, CustomProvider, HostedModel, State, ThemeChoice};
use crate::util::gb;

const KEEP_ALIVE: [KeepAlive; 4] = [KeepAlive(5), KeepAlive(30), KeepAlive(120), KeepAlive(0)];
const CONTEXT_SIZES: [u32; 6] = [8192, 16_384, 32_768, 65_536, 131_072, 262_144];

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
            0 => f.write_str("Until Scoobert closes"),
            m if m < 60 => write!(f, "{m} minutes"),
            m => write!(f, "{} hours", m / 60),
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
        write!(f, "{}K tokens", self.0 / 1024)
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
}

#[derive(Debug, Clone)]
pub enum Msg {
    Init,
    Section(Section),
    Theme(ThemeChoice),
    Approvals(Approvals),
    NotesFolder(String),
    CommitNotesFolder,
    ActivityLog(bool),
    RememberStep(bool),
    KeepAlive(KeepAlive),
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
                if let Some(p) = self.provider.clone() {
                    return (self.select_provider(p, ctx.host), Effect::None);
                }
            }
            Msg::Section(section) => {
                self.section = section;
                if section == Section::Hosted {
                    return self.update(Msg::Init, ctx);
                }
            }
            Msg::Theme(t) => {
                s.theme = t;
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
                        async { rfd::AsyncFileDialog::new().set_title("Choose the models folder").pick_folder().await.map(|h| h.path().to_path_buf()) },
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
                    self.error = Some("Give the provider a name and an address that starts with https:// or http://.".into());
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
            Msg::Close => return (Task::done(Message::CloseModal), Effect::None),
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
        let sidebar = column![
            text("Settings").size(18).font(fonts::ui_semibold()),
            space().height(8),
            nav("General", Section::General),
            nav("Models on this computer", Section::Local),
            nav("Hosted models", Section::Hosted),
        ]
        .spacing(4)
        .width(210)
        .padding(16);
        let content = match self.section {
            Section::General => self.general(ctx.state, ctx.isolation),
            Section::Local => self.local(ctx),
            Section::Hosted => self.hosted(ctx.state),
        };
        let close = button(icon(Icon::Close, 16.0)).padding(6).style(theme::ghost).on_press(Msg::Close);
        let body = column![row![space::horizontal(), close], scrollable(container(content).padding(iced::Padding { top: 0.0, right: 24.0, bottom: 24.0, left: 8.0 })).height(Fill).style(theme::scrollbar)];
        container(row![sidebar, rule::vertical(1).style(theme::divider), body.width(Fill)]).height(600).into()
    }

    fn general<'a>(&'a self, state: &'a State, isolation: &'a str) -> Element<'a, Msg> {
        let s = &state.settings;
        let mode_help = match s.approvals {
            Approvals::Ask => "Scoobert asks before it edits a file or runs a command. Edits to the notes folder never ask.".to_string(),
            Approvals::Project => format!("For leaving a task running. Edits inside the project run without asking, and edits elsewhere are refused. {isolation} Every command's memory is capped."),
            Approvals::Auto => "Everything runs without asking, including commands that download or install software.".to_string(),
        };
        column![
            heading("General"),
            column![
                field(
                    "When Scoobert makes changes",
                    None,
                    pick_list(Approvals::ALL, Some(s.approvals), Msg::Approvals).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu).into(),
                ),
                text(mode_help).size(12).style(theme::muted),
            ]
            .spacing(4),
            field(
                "Appearance",
                None,
                pick_list(ThemeChoice::ALL, Some(s.theme), Msg::Theme).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu).into(),
            ),
            field(
                "Notes folder",
                Some("The folder inside each project where Scoobert keeps its notes."),
                text_input("Notes", &self.notes_folder)
                    .on_input(Msg::NotesFolder)
                    .on_submit(Msg::CommitNotesFolder)
                    .size(13)
                    .padding([6, 10])
                    .width(220)
                    .style(theme::input)
                    .into(),
            ),
            switch_row("Log finished tasks in today's note", "Adds what changed to the Daily folder after a task that edited files.", s.activity_log, Msg::ActivityLog),
            switch_row(
                "Ask the model what to remember",
                "After a task, a local model lists up to three facts for the related note. This takes about a minute on a laptop CPU.",
                s.remember_step,
                Msg::RememberStep
            ),
        ]
        .spacing(18)
        .into()
    }

    fn local<'a>(&'a self, ctx: ViewCtx<'a>) -> Element<'a, Msg> {
        let s = &ctx.state.settings;
        let installed: Vec<&ModelOption> = ctx.models.iter().filter(|m| m.provider.is_empty()).collect();
        let mut list = Column::new().spacing(8);
        for m in &installed {
            let mut sizes: Vec<ContextChoice> = CONTEXT_SIZES.iter().map(|&c| ContextChoice(c)).collect();
            if !CONTEXT_SIZES.contains(&m.context) {
                sizes.push(ContextChoice(m.context));
            }
            let free = crate::sys::available_memory();
            let warn = if m.memory_needed > free {
                text(format!("Needs about {} of free memory; {} is free now.", gb(m.memory_needed), gb(free))).size(12).style(theme::warn_text)
            } else {
                text(format!("Needs about {} of free memory.", gb(m.memory_needed))).size(12).style(theme::muted)
            };
            let name = m.name.clone();
            list = list.push(
                container(
                    row![
                        column![
                            text(m.label.clone()).size(14).font(fonts::ui_semibold()),
                            text(format!("{}{}", gb(m.size), if m.vision { ", reads images" } else { "" })).size(12).style(theme::muted),
                            warn,
                        ]
                        .spacing(2)
                        .width(Fill),
                        text("Context").size(12).style(theme::muted),
                        pick_list(sizes, Some(ContextChoice(m.context)), move |c| Msg::Context(name.clone(), c))
                            .padding([4, 8])
                            .text_size(12)
                            .style(theme::select)
                            .menu_style(theme::menu),
                    ]
                    .spacing(10)
                    .align_y(Alignment::Center),
                )
                .padding(12)
                .style(theme::card),
            );
        }
        if installed.is_empty() {
            list = list.push(text("No models yet. Download one below.").size(13).style(theme::muted));
        }

        let mut catalog = Column::new().spacing(8);
        for (i, m) in CATALOG.iter().enumerate() {
            let have = installed.iter().any(|x| x.name == m.name);
            let status: Element<'a, Msg> = if have {
                text("Downloaded").size(12).style(theme::accent_text).into()
            } else if let Some(d) = ctx.download.as_ref().filter(|d| d.job == Job::Catalog(i)) {
                let (done, total) = d.progress.as_ref().map(|p| (p.done, p.total.max(1))).unwrap_or((0, 1));
                row![
                    progress_bar(0.0..=total as f32, done as f32).length(120).girth(6).style(theme::meter),
                    text(format!("{} of {}", gb(done), gb(total))).size(12).style(theme::muted),
                    button(text("Stop").size(12)).padding([3, 10]).style(theme::secondary).on_press(Msg::CancelDownload),
                ]
                .spacing(8)
                .align_y(Alignment::Center)
                .into()
            } else if ctx.queue.contains(&Job::Catalog(i)) {
                text("Waiting...").size(12).style(theme::muted).into()
            } else {
                button(row![icon(Icon::Download, 14.0), text(format!("Download {}", gb(m.download_bytes))).size(12)].spacing(6).align_y(Alignment::Center))
                    .padding([4, 10])
                    .style(theme::secondary)
                    .on_press(Msg::Download(i))
                    .into()
            };
            let fit = if setup::fits(i) {
                text(format!("Needs about {} of free memory.", gb(memory_needed(m)))).size(12).style(theme::muted)
            } else {
                text(format!("Needs about {} of free memory, more than this computer has.", gb(memory_needed(m)))).size(12).style(theme::warn_text)
            };
            catalog = catalog.push(
                container(
                    row![
                        column![
                            row![text(m.label).size(14).font(fonts::ui_semibold()), container(text(m.tier).size(11)).padding([1, 6]).style(theme::chip)]
                                .spacing(8)
                                .align_y(Alignment::Center),
                            text(m.summary).size(12).style(theme::muted),
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
            heading("Models on this computer"),
            list,
            subheading("Download"),
            catalog,
            subheading("Download another model"),
            text("Any GGUF model on Hugging Face, as owner/repository:quantization.").size(12).style(theme::muted),
            row![
                text_input("unsloth/Qwen3.5-9B-GGUF:Q4_K_M", &self.custom_spec)
                    .on_input(Msg::CustomSpec)
                    .on_submit(Msg::DownloadCustom)
                    .size(13)
                    .padding([6, 10])
                    .style(theme::input),
                button(text("Download").size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::DownloadCustom),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            custom_progress(ctx.download),
            subheading("Where models live"),
            row![
                text_input(&crate::paths::display(&crate::paths::get().default_models_dir()), &self.models_dir)
                    .on_input(Msg::ModelsDir)
                    .on_submit(Msg::CommitModelsDir)
                    .size(13)
                    .padding([6, 10])
                    .style(theme::input),
                button(text("Browse").size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::BrowseModelsDir),
                button(icon(Icon::External, 15.0)).padding(6).style(theme::ghost).on_press(Msg::RevealModels),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            field(
                "Unload the model after",
                Some("An idle model still holds its memory. Saved caches make the next load quick."),
                pick_list(&KEEP_ALIVE[..], Some(KeepAlive(s.keep_alive_minutes)), Msg::KeepAlive)
                    .padding([5, 10])
                    .text_size(13)
                    .style(theme::select)
                    .menu_style(theme::menu)
                    .into(),
            ),
            field(
                "llama-server program",
                Some("Leave empty to use the copy that comes with Scoobert."),
                text_input("Bundled", &self.server_path)
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

    fn hosted<'a>(&'a self, state: &'a State) -> Element<'a, Msg> {
        let s = &state.settings;
        let all = providers::all(&s.custom_providers);
        let choices: Vec<ProviderChoice> = all.iter().cloned().map(ProviderChoice).collect();
        let selected = self.provider.clone().map(ProviderChoice);
        let mut col = column![
            heading("Hosted models"),
            text("Hosted models run on the provider's servers, so Scoobert sends them your conversations and the files it reads. Keys are kept in the system credential store.")
                .size(13)
                .style(theme::muted),
            row![
                text("Provider").size(13).width(90),
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
                    text("Key saved").size(13),
                    space::horizontal(),
                    button(text("Remove key").size(13)).padding([5, 12]).style(theme::danger).on_press(Msg::RemoveKey),
                ]
                .spacing(8)
                .align_y(Alignment::Center)
                .into()
            } else {
                let mut r = row![
                    text_input("Paste an API key", &self.key_input)
                        .secure(true)
                        .on_input(Msg::KeyInput)
                        .on_submit(Msg::SaveKey)
                        .size(13)
                        .padding([6, 10])
                        .style(theme::input),
                    button(text("Save key").size(13)).padding([6, 12]).style(theme::primary).on_press_maybe((!self.key_input.trim().is_empty()).then_some(Msg::SaveKey)),
                ]
                .spacing(8)
                .align_y(Alignment::Center);
                if !p.key_url.is_empty() {
                    r = r.push(button(text("Get a key").size(13)).style(theme::link).on_press(Msg::OpenUrl(p.key_url.clone())));
                }
                r.into()
            };
            col = col.push(row![text("API key").size(13).width(90), key_row].spacing(10).align_y(Alignment::Center));
            if p.id.starts_with("custom-") {
                col = col.push(
                    row![
                        text(p.base_url.clone()).size(12).style(theme::muted).font(fonts::mono()),
                        space::horizontal(),
                        button(text("Remove this provider").size(12)).padding([4, 10]).style(theme::danger).on_press(Msg::RemoveCustom(p.id.clone())),
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
                        button(icon(Icon::Close, 14.0)).padding(4).style(theme::ghost).on_press(Msg::RemoveHosted(m.reference())),
                    ]
                    .align_y(Alignment::Center),
                );
            }
            col = col.push(subheading("In the model menu"));
            col = col.push(list);
        }
        col = col.push(subheading("Add a provider by address"));
        col = col.push(text("Any server that accepts OpenAI-style chat requests, such as a model server on another computer.").size(12).style(theme::muted));
        col = col.push(
            row![
                text_input("Name", &self.custom_name).on_input(Msg::CustomName).size(13).padding([6, 10]).width(160).style(theme::input),
                text_input("https://example.com/v1", &self.custom_url).on_input(Msg::CustomUrl).on_submit(Msg::AddCustom).size(13).padding([6, 10]).style(theme::input),
                button(text("Add").size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::AddCustom),
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
            return text("Loading the model list...").size(13).style(theme::muted).into();
        }
        if self.available.is_empty() {
            let hint = if self.has_key { "Refresh the list to see this provider's models." } else { "Save a key to see this provider's models." };
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
            let mut tags = vec![format!("{}K context", m.context / 1000)];
            if m.vision {
                tags.push("images".into());
            }
            if m.reasoning {
                tags.push("thinking".into());
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
                text_input("Search models", &self.filter).on_input(Msg::Filter).size(13).padding([6, 10]).style(theme::input),
                button(icon(Icon::Refresh, 14.0)).padding(6).style(theme::ghost).on_press(Msg::ListModels),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            text("Tick the models to show in the model menu.").size(12).style(theme::muted),
            container(scrollable(list.padding([4, 8])).style(theme::scrollbar)).height(260).style(theme::card),
        ]
        .spacing(8)
        .into()
    }
}

fn custom_progress<'a>(download: &'a Option<Download>) -> Element<'a, Msg> {
    let Some(d) = download.as_ref() else { return space().into() };
    let Job::Custom(spec) = &d.job else { return space().into() };
    let (done, total) = d.progress.as_ref().map(|p| (p.done, p.total.max(1))).unwrap_or((0, 1));
    row![
        text(spec.clone()).size(12).font(fonts::mono()),
        progress_bar(0.0..=total as f32, done as f32).length(120).girth(6).style(theme::meter),
        text(format!("{} of {}", gb(done), gb(total))).size(12).style(theme::muted),
        button(text("Stop").size(12)).padding([3, 10]).style(theme::secondary).on_press(Msg::CancelDownload),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn icons_ok<'a>() -> Element<'a, Msg> {
    super::icons::tinted(Icon::Check, 16.0, |t| t.ok).into()
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
    field(label, Some(help), toggler(on).on_toggle(msg).size(20).style(theme::switch).into())
}
