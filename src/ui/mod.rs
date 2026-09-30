//! The Scoobert window: projects and conversations on the left, the conversation in the middle, notes on the right.

pub mod chat;
pub mod fonts;
pub mod graph;
pub mod icons;
pub mod notes;
pub mod settings;
pub mod setup;
pub mod theme;

use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use iced::futures::{SinkExt, Stream};
use iced::keyboard::{self, Key, key::Named};
use iced::widget::scrollable::RelativeOffset;
use iced::widget::text_editor::{self, Binding};
use iced::widget::{
    Column, button, center, column, container, hover, image, mouse_area, opaque, operation, pick_list,
    progress_bar, row, rule, scrollable, space, stack, text, text_editor as editor, text_input, tooltip,
};
use iced::{Alignment, Element, Fill, Length, Size, Subscription, Task, Theme, system, window};

use crate::agent::conversation::{Conversation, Image, Summary};
use crate::agent::{Decision, Event, Host, ModelOption, Snapshot};
use crate::llama::{ServerStatus, SharedSettings};
use crate::store::{Approvals, State, ThemeChoice, Thinking};
use crate::util::{ago, clip, short_count};
use chat::Chat;
use icons::{Icon, icon};

const SIDEBAR_WIDTH: f32 = 290.0;
const COMPOSER_ID: &str = "composer";
const RENAME_ID: &str = "rename";

pub fn run() -> iced::Result {
    let size = State::load().window.map(|(w, h)| Size::new(w.max(760.0), h.max(480.0))).unwrap_or(Size::new(1320.0, 860.0));
    iced::application(App::new, App::update, App::view)
        .window(window::Settings {
            size,
            min_size: Some(Size::new(760.0, 480.0)),
            icon: window::icon::from_file_data(include_bytes!("../../assets/icon.png"), None).ok(),
            exit_on_close_request: false,
            // Matches the .desktop file, so Linux desktops show the right icon and group the window.
            #[cfg(target_os = "linux")]
            platform_specific: window::settings::PlatformSpecific { application_id: "scoobert".into(), ..Default::default() },
            ..window::Settings::default()
        })
        .centered()
        .exit_on_close_request(false)
        .settings(iced::Settings { default_text_size: 14.into(), ..iced::Settings::default() })
        .default_font(fonts::ui())
        .title(App::title)
        .theme(App::theme)
        .subscription(App::subscription)
        .run()
}

#[derive(Clone)]
pub struct HostHandle(pub Arc<Host>);

impl std::fmt::Debug for HostHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Host")
    }
}

/// A model menu entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelChoice {
    pub name: String,
    pub label: String,
}

impl std::fmt::Display for ModelChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label)
    }
}

#[derive(Debug, Clone)]
pub enum Confirm {
    DeleteConversation(PathBuf),
    RemoveProject(PathBuf),
}

#[derive(Debug, Clone)]
pub enum Message {
    HostReady(HostHandle),
    Host(Event),
    Mode(iced::theme::Mode),
    WindowOpened(window::Id),
    Scale(f32),
    Resized(Size),
    CloseRequested(window::Id),
    ShutdownDone(window::Id),
    /// Closes the app the same way the window's close button does.
    Quit,
    Tick,
    Escape,
    Noop,

    AddProject,
    ProjectPicked(Option<PathBuf>),
    SelectProject(PathBuf),
    SelectNoProject,
    RevealProject(PathBuf),
    Sessions(PathBuf, Vec<Summary>),
    NewConversation,
    OpenConversation(PathBuf),
    Opened(Result<Box<Snapshot>, String>),
    StartRename(PathBuf, String),
    RenameInput(String),
    CommitRename,
    AskConfirm(Confirm),
    Confirmed(Confirm),

    Composer(text_editor::Action),
    Send,
    /// Resumes an interrupted task with a hidden note that explains what happened.
    Continue,
    Stop,
    AttachImage,
    ImagesPicked(Vec<Image>),
    FileDropped(PathBuf),
    RemoveImage(usize),
    SetModel(ModelChoice),
    SetThinking(Thinking),
    SetApprovals(Approvals),
    Approve(u64, Decision),
    Toggle(String),
    Link(String),
    Copy(String),
    SaveAsNote(String),
    /// The answer to the offer to turn on web search before sending: true turns it on.
    WebOffer(bool),

    Notes(notes::Msg),
    Settings(settings::Msg),
    Setup(setup::Msg),
    OpenSettings(settings::Section),
    CloseModal,

    Toast(String),
    Update(Option<(String, String)>),
    OpenUrl(String),
}

pub struct App {
    state: State,
    shared: SharedSettings,
    host: Option<Arc<Host>>,
    system_dark: bool,
    light: Theme,
    dark: Theme,
    logo: image::Handle,
    scale: f32,
    window: Option<window::Id>,
    window_size: Option<Size>,
    server: ServerStatus,
    models: Vec<ModelOption>,
    sessions: Vec<Summary>,
    renaming: Option<(PathBuf, String)>,
    chat: Option<Chat>,
    composer: text_editor::Content,
    images: Vec<Image>,
    notes: notes::Pane,
    settings: Option<settings::Panel>,
    setup: setup::Setup,
    confirm: Option<Confirm>,
    toast: Option<(String, Instant)>,
    update_notice: Option<(String, String)>,
    /// A message that looks like it needs the web is waiting while the user decides about web search.
    web_offer: bool,
}

struct HostSeed(SharedSettings);

impl Hash for HostSeed {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "host".hash(state);
    }
}

fn host_worker(seed: &HostSeed) -> impl Stream<Item = Message> + use<> {
    let settings = seed.0.clone();
    iced::stream::channel(256, async move |mut output: iced::futures::channel::mpsc::Sender<Message>| {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let host = Host::new(settings, tx);
        let _ = output.send(Message::HostReady(HostHandle(host))).await;
        while let Some(event) = rx.recv().await {
            if output.send(Message::Host(event)).await.is_err() {
                break;
            }
        }
        std::future::pending::<()>().await;
    })
}

impl App {
    fn new() -> (App, Task<Message>) {
        let state = State::load();
        let shared = Arc::new(RwLock::new(state.settings.clone()));
        let logo = image::Handle::from_bytes(include_bytes!("../../assets/logo.png").as_slice());
        let notes = notes::Pane::new(state.notes_open);
        let mut app = App {
            state,
            shared,
            host: None,
            system_dark: true,
            light: theme::build(false),
            dark: theme::build(true),
            logo,
            scale: 1.0,
            window: None,
            window_size: None,
            server: ServerStatus::Stopped,
            models: Vec::new(),
            sessions: Vec::new(),
            renaming: None,
            chat: None,
            composer: text_editor::Content::new(),
            images: Vec::new(),
            notes,
            settings: None,
            setup: setup::Setup::default(),
            confirm: None,
            toast: None,
            update_notice: None,
            web_offer: false,
        };
        let mut tasks = vec![system::theme().map(Message::Mode)];
        if crate::update::due(app.state.update_checked_at) {
            app.state.update_checked_at = crate::util::now_millis();
            app.save();
            tasks.push(Task::perform(crate::update::check(), Message::Update));
        }
        (app, Task::batch(tasks))
    }

    fn title(&self) -> String {
        match (&self.chat, self.current_project()) {
            (Some(c), Some(p)) if !c.is_empty() => format!("{} - {} - Scoobert", clip(&c.title, 60), project_name(&p)),
            (_, Some(p)) => format!("{} - Scoobert", project_name(&p)),
            (Some(c), None) if !c.is_empty() => format!("{} - Scoobert", clip(&c.title, 60)),
            _ => "Scoobert".into(),
        }
    }

    fn theme(&self) -> Theme {
        let dark = match self.state.settings.theme {
            ThemeChoice::System => self.system_dark,
            ThemeChoice::Dark => true,
            ThemeChoice::Light => false,
        };
        if dark { self.dark.clone() } else { self.light.clone() }
    }

    fn subscription(&self) -> Subscription<Message> {
        let mut subs = vec![
            Subscription::run_with(HostSeed(self.shared.clone()), host_worker),
            system::theme_changes().map(Message::Mode),
            window::open_events().map(Message::WindowOpened),
            window::close_requests().map(Message::CloseRequested),
            window::resize_events().map(|(_, size)| Message::Resized(size)),
            window::events().filter_map(|(_, event)| match event {
                window::Event::FileDropped(path) => Some(Message::FileDropped(path)),
                _ => None,
            }),
            keyboard::listen().filter_map(|event| match event {
                keyboard::Event::KeyPressed { key: Key::Named(Named::Escape), .. } => Some(Message::Escape),
                _ => None,
            }),
            iced::time::every(Duration::from_secs(30)).map(|_| Message::Tick),
        ];
        if self.toast.is_some() || self.notes.dirty() {
            subs.push(iced::time::every(Duration::from_secs(1)).map(|_| Message::Tick));
        }
        Subscription::batch(subs)
    }

    fn save(&mut self) {
        *self.shared.write().unwrap() = self.state.settings.clone();
        if let Err(err) = self.state.save() {
            eprintln!("[state] {err:#}");
        }
    }

    fn current_project(&self) -> Option<PathBuf> {
        self.state.current_project.clone().filter(|p| self.state.project(p).is_some())
    }

    /// The folder conversations work in: the selected project, or the shared folder when none is selected.
    fn workspace(&self) -> PathBuf {
        self.current_project().unwrap_or_else(crate::paths::projects_root)
    }

    /// Remembers the conversation to reopen for its project, or for no project.
    fn remember_last(&mut self, cwd: &std::path::Path, file: PathBuf) {
        if crate::agent::is_general(cwd) {
            if self.state.general_last_session.as_ref() != Some(&file) {
                self.state.general_last_session = Some(file);
                self.save();
            }
        } else if let Some(p) = self.state.project_mut(cwd)
            && p.last_session.as_ref() != Some(&file)
        {
            p.last_session = Some(file);
            self.save();
        }
    }

    fn toast(&mut self, message: impl Into<String>) {
        self.toast = Some((message.into(), Instant::now()));
    }

    fn refresh_models(&mut self) {
        if let Some(host) = &self.host {
            self.models = host.models();
        }
    }

    fn load_sessions(&self) -> Task<Message> {
        let project = self.workspace();
        Task::perform(
            async move {
                let dir = project.clone();
                let list = tokio::task::spawn_blocking(move || Conversation::list(&dir)).await.unwrap_or_default();
                (project, list)
            },
            |(p, list)| Message::Sessions(p, list),
        )
    }

    fn open(&mut self, file: Option<PathBuf>) -> Task<Message> {
        let Some(host) = self.host.clone() else { return Task::none() };
        let project = self.workspace();
        if self.current_project().is_none() {
            let _ = std::fs::create_dir_all(&project);
        }
        if file.is_some() && self.chat.as_ref().map(|c| &c.file) == file.as_ref() {
            return Task::none();
        }
        Task::perform(
            async move { host.open(&project, file.as_deref()).map(Box::new).map_err(|e| format!("{e:#}")) },
            Message::Opened,
        )
    }

    /// Switches to a project, or to no project when path is None, and reopens its last conversation.
    fn select_project(&mut self, path: Option<PathBuf>) -> Task<Message> {
        self.state.current_project = path.clone();
        self.save();
        self.chat = None;
        self.sessions.clear();
        let last = match &path {
            Some(p) => self.state.project(p).and_then(|p| p.last_session.clone()),
            None => self.state.general_last_session.clone(),
        };
        let last = last.filter(|f| f.exists());
        self.notes.set_project(path.map(|p| p.join(&self.state.settings.notes_folder)));
        Task::batch([self.load_sessions(), self.open(last), self.notes.reload()])
    }

    fn model_choices(&self) -> Vec<ModelChoice> {
        self.models
            .iter()
            .filter(|m| m.usable)
            .map(|m| ModelChoice {
                name: m.name.clone(),
                label: if m.provider.is_empty() { m.label.clone() } else { format!("{} ({})", m.label, m.provider) },
            })
            .collect()
    }

    fn current_model(&self) -> Option<&ModelOption> {
        let name = self.chat.as_ref().map(|c| c.model.clone()).unwrap_or_else(|| self.state.settings.model.clone());
        self.models.iter().find(|m| m.name == name)
    }

    fn needs_setup(&self) -> bool {
        !self.state.setup_done && self.host.is_some() && self.models.is_empty()
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::HostReady(HostHandle(host)) => {
                self.host = Some(host);
                self.refresh_models();
                if !self.models.is_empty() && !self.state.setup_done {
                    self.state.setup_done = true;
                    self.save();
                }
                return self.select_project(self.current_project());
            }
            Message::Host(event) => return self.on_host_event(event),
            Message::Mode(mode) => self.system_dark = mode != iced::theme::Mode::Light,
            Message::WindowOpened(id) => {
                self.window = Some(id);
                return window::scale_factor(id).map(Message::Scale);
            }
            Message::Scale(s) => self.scale = s,
            Message::Resized(size) => self.window_size = Some(size),
            Message::CloseRequested(id) => {
                if let Some(size) = self.window_size {
                    self.state.window = Some((size.width, size.height));
                }
                let flush = self.notes.flush();
                self.save();
                let host = self.host.clone();
                return flush.chain(Task::perform(
                    async move {
                        if let Some(h) = host {
                            h.shutdown().await;
                        }
                    },
                    move |_| Message::ShutdownDone(id),
                ));
            }
            Message::ShutdownDone(id) => return window::close(id),
            Message::Quit => {
                if let Some(id) = self.window {
                    return self.update(Message::CloseRequested(id));
                }
                return iced::exit();
            }
            Message::Tick => {
                if self.toast.as_ref().is_some_and(|(_, at)| at.elapsed() > Duration::from_secs(5)) {
                    self.toast = None;
                }
                return self.notes.tick().map(Message::Notes);
            }
            Message::Escape => {
                if self.confirm.is_some() || self.settings.is_some() {
                    self.confirm = None;
                    self.settings = None;
                } else if self.renaming.is_some() {
                    self.renaming = None;
                } else if self.chat.as_ref().is_some_and(|c| c.running) {
                    return self.update(Message::Stop);
                }
            }
            Message::Noop => {}

            Message::AddProject => {
                return Task::perform(
                    async { rfd::AsyncFileDialog::new().set_title("Choose a project folder").pick_folder().await.map(|h| h.path().to_path_buf()) },
                    Message::ProjectPicked,
                );
            }
            Message::ProjectPicked(Some(path)) => {
                if self.state.project(&path).is_none() {
                    self.state.projects.push(crate::store::Project { path: path.clone(), last_session: None });
                    self.state.projects.sort_by_key(|p| project_name(&p.path).to_lowercase());
                }
                return self.select_project(Some(path));
            }
            Message::ProjectPicked(None) => {}
            Message::SelectProject(path) => {
                if self.current_project().as_ref() != Some(&path) {
                    return self.select_project(Some(path));
                }
            }
            Message::SelectNoProject => {
                if self.current_project().is_some() {
                    return self.select_project(None);
                }
            }
            Message::RevealProject(path) => {
                let _ = opener::reveal(&path);
            }
            Message::Sessions(project, list) => {
                if self.workspace() == project {
                    self.sessions = list;
                }
            }
            Message::NewConversation => return self.open(None),
            Message::OpenConversation(file) => return self.open(Some(file)),
            Message::Opened(Ok(snap)) => {
                let file = snap.file.clone();
                if let Some(host) = &self.host {
                    host.close_idle(&snap.id);
                }
                let project = snap.cwd.clone();
                if file.exists() {
                    self.remember_last(&project, file);
                }
                self.chat = Some(Chat::from_snapshot(*snap));
                return Task::batch([operation::focus(COMPOSER_ID), operation::snap_to(chat::TRANSCRIPT_ID, RelativeOffset::START)]);
            }
            Message::Opened(Err(err)) => self.toast(err),
            Message::StartRename(file, title) => {
                self.renaming = Some((file, title));
                return operation::focus(RENAME_ID);
            }
            Message::RenameInput(value) => {
                if let Some((_, t)) = &mut self.renaming {
                    *t = value;
                }
            }
            Message::CommitRename => {
                if let (Some((file, title)), Some(host)) = (self.renaming.take(), &self.host) {
                    let title = title.trim().to_string();
                    if !title.is_empty() {
                        match host.rename(&file, &title) {
                            Ok(()) => {
                                if let Some(c) = self.chat.as_mut().filter(|c| c.file == file) {
                                    c.title = title;
                                }
                            }
                            Err(e) => self.toast(format!("{e:#}")),
                        }
                    }
                    return self.load_sessions();
                }
            }
            Message::AskConfirm(c) => self.confirm = Some(c),
            Message::Confirmed(c) => {
                self.confirm = None;
                match c {
                    Confirm::DeleteConversation(file) => {
                        let Some(host) = &self.host else { return Task::none() };
                        if let Err(e) = host.delete(&file) {
                            self.toast(format!("{e:#}"));
                        }
                        let was_open = self.chat.as_ref().is_some_and(|c| c.file == file);
                        let mut tasks = vec![self.load_sessions()];
                        if was_open {
                            self.chat = None;
                            tasks.push(self.open(None));
                        }
                        return Task::batch(tasks);
                    }
                    Confirm::RemoveProject(path) => {
                        self.state.projects.retain(|p| p.path != path);
                        let was_current = self.state.current_project.as_ref() == Some(&path);
                        self.save();
                        if was_current {
                            return self.select_project(None);
                        }
                    }
                }
            }

            Message::Composer(action) => self.composer.perform(action),
            Message::Send => return self.send(true),
            Message::Continue => {
                let (Some(host), Some(chat)) = (self.host.clone(), self.chat.as_mut()) else { return Task::none() };
                match host.prompt(&chat.id, "Continue".into(), Vec::new(), true) {
                    Ok(()) => {
                        chat.pending = Some("Continue".into());
                        chat.running = true;
                        chat.interrupted = false;
                        chat.error = None;
                        return operation::snap_to(chat::TRANSCRIPT_ID, RelativeOffset::START);
                    }
                    Err(e) => chat.error = Some(format!("{e:#}")),
                }
            }
            Message::WebOffer(enable) => {
                self.web_offer = false;
                if enable {
                    self.state.settings.web_access = true;
                    self.save();
                }
                return self.send(false);
            }
            Message::Stop => {
                if let (Some(host), Some(chat)) = (&self.host, &mut self.chat) {
                    host.abort(&chat.id);
                    chat.approvals.clear();
                }
            }
            Message::AttachImage => {
                return Task::perform(
                    async {
                        let files = rfd::AsyncFileDialog::new()
                            .add_filter("Images", &["png", "jpg", "jpeg", "gif", "webp"])
                            .set_title("Attach images")
                            .pick_files()
                            .await
                            .unwrap_or_default();
                        let mut out = Vec::new();
                        for f in files {
                            let bytes = f.read().await;
                            let name = f.file_name().to_lowercase();
                            let mime = match name.rsplit('.').next() {
                                Some("jpg" | "jpeg") => "image/jpeg",
                                Some("gif") => "image/gif",
                                Some("webp") => "image/webp",
                                _ => "image/png",
                            };
                            use base64::Engine;
                            out.push(Image { mime: mime.into(), data: base64::engine::general_purpose::STANDARD.encode(bytes) });
                        }
                        out
                    },
                    Message::ImagesPicked,
                );
            }
            Message::ImagesPicked(images) => self.images.extend(images),
            Message::FileDropped(path) => {
                let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
                let mime = match ext.as_str() {
                    "png" => Some("image/png"),
                    "jpg" | "jpeg" => Some("image/jpeg"),
                    "gif" => Some("image/gif"),
                    "webp" => Some("image/webp"),
                    _ => None,
                };
                let vision = self.current_model().is_some_and(|m| m.vision);
                match (mime, std::fs::read(&path)) {
                    (Some(mime), Ok(bytes)) if vision => {
                        use base64::Engine;
                        self.images.push(Image { mime: mime.into(), data: base64::engine::general_purpose::STANDARD.encode(bytes) });
                    }
                    _ => {
                        // Other files go into the message as a path, which the model can read.
                        let shown = crate::paths::display(&path);
                        self.composer.perform(text_editor::Action::Move(text_editor::Motion::DocumentEnd));
                        let text = if self.composer.text().trim().is_empty() { shown } else { format!(" {shown}") };
                        self.composer.perform(text_editor::Action::Edit(text_editor::Edit::Paste(Arc::new(text))));
                    }
                }
            }
            Message::RemoveImage(i) => {
                if i < self.images.len() {
                    self.images.remove(i);
                }
            }
            Message::SetModel(choice) => {
                self.state.settings.model = choice.name.clone();
                self.save();
                if let (Some(host), Some(chat)) = (&self.host, &mut self.chat) {
                    match host.set_model(&chat.id, &choice.name) {
                        Ok(()) => chat.model = choice.name,
                        Err(e) => self.toast = Some((format!("{e:#}"), Instant::now())),
                    }
                }
            }
            Message::SetThinking(level) => {
                self.state.settings.thinking = level;
                self.save();
                if let (Some(host), Some(chat)) = (&self.host, &mut self.chat) {
                    let _ = host.set_thinking(&chat.id, level);
                    chat.thinking = level;
                }
            }
            Message::SetApprovals(a) => {
                self.state.settings.approvals = a;
                self.save();
            }
            Message::Approve(id, decision) => {
                if let Some(host) = &self.host {
                    host.answer(id, decision);
                }
                if let Some(chat) = &mut self.chat {
                    chat.approvals.retain(|(a, _)| *a != id);
                }
            }
            Message::Toggle(key) => {
                if let Some(chat) = &mut self.chat
                    && !chat.expanded.remove(&key)
                {
                    chat.expanded.insert(key);
                }
            }
            Message::Link(url) => return self.open_link(url),
            Message::Copy(text) => {
                self.toast("Copied.");
                return iced::clipboard::write(text);
            }
            Message::SaveAsNote(body) => {
                let title = body.lines().map(|l| l.trim_start_matches('#').trim()).find(|l| !l.is_empty()).map(|l| clip(l, 60)).unwrap_or_else(|| "Saved reply".into());
                let project = self.current_project();
                return self.notes.update(notes::Msg::CreateWith(title, body), project.as_deref());
            }

            Message::Notes(msg) => {
                let project = self.current_project();
                let task = self.notes.update(msg, project.as_deref());
                self.state.notes_open = self.notes.open;
                return task;
            }
            Message::Settings(msg) => {
                let Some(panel) = &mut self.settings else { return Task::none() };
                let mut ctx = settings::Ctx { state: &mut self.state, host: self.host.as_ref() };
                let (task, effect) = panel.update(msg, &mut ctx);
                return self.after_settings(task, effect);
            }
            Message::Setup(msg) => {
                let (task, effect) = self.setup.update(msg, &mut self.state, self.host.as_ref());
                return self.after_settings(task, effect);
            }
            Message::OpenSettings(section) => {
                self.settings = Some(settings::Panel::new(section, &self.state.settings));
                if section == settings::Section::Hosted {
                    return self.update(Message::Settings(settings::Msg::Init));
                }
            }
            Message::CloseModal => {
                self.settings = None;
                self.confirm = None;
            }

            Message::Toast(t) => self.toast(t),
            Message::Update(notice) => self.update_notice = notice,
            Message::OpenUrl(url) => {
                if url.starts_with("https://") {
                    let _ = opener::open(url);
                }
            }
        }
        Task::none()
    }

    /// Applies what a settings or setup change requires elsewhere in the app.
    fn after_settings(&mut self, task: Task<Message>, effect: settings::Effect) -> Task<Message> {
        match effect {
            settings::Effect::None => {}
            settings::Effect::Saved => self.save(),
            settings::Effect::ModelsChanged => {
                self.save();
                self.refresh_models();
            }
            settings::Effect::Toast(t) => {
                self.save();
                self.refresh_models();
                self.toast(t);
            }
        }
        task
    }

    fn open_link(&mut self, url: String) -> Task<Message> {
        if let Some(target) = url.strip_prefix("note:") {
            return self.notes.update(notes::Msg::OpenLink(target.to_string()), self.current_project().as_deref());
        }
        if url.starts_with("https://") || url.starts_with("http://") {
            let _ = opener::open(&url);
        }
        Task::none()
    }

    /// Sends the message. With web search off, a message that looks like it needs the web first offers to turn it on.
    fn send(&mut self, offer_web: bool) -> Task<Message> {
        let text = self.composer.text().trim().to_string();
        if text.is_empty() && self.images.is_empty() {
            return Task::none();
        }
        if offer_web && !self.state.settings.web_access && NEEDS_WEB.is_match(&text) {
            self.web_offer = true;
            return Task::none();
        }
        let Some(host) = self.host.clone() else { return Task::none() };
        let Some(chat) = &mut self.chat else {
            self.toast("Add a project folder first.");
            return Task::none();
        };
        if chat.running {
            return Task::none();
        }
        let images = std::mem::take(&mut self.images);
        match host.prompt(&chat.id, text.clone(), images, false) {
            Ok(()) => {
                chat.pending = Some(text);
                chat.running = true;
                chat.interrupted = false;
                chat.error = None;
                chat.notice = None;
                self.composer = text_editor::Content::new();
                operation::snap_to(chat::TRANSCRIPT_ID, RelativeOffset::START)
            }
            Err(e) => {
                chat.error = Some(format!("{e:#}"));
                Task::none()
            }
        }
    }

    fn on_host_event(&mut self, event: Event) -> Task<Message> {
        match &event {
            Event::Server(status) => {
                self.server = status.clone();
                return Task::none();
            }
            Event::NotesChanged { cwd } => {
                if self.current_project().as_ref() == Some(cwd) {
                    return self.notes.reload();
                }
                return Task::none();
            }
            Event::Error { conv: None, message } => {
                self.toast(message.clone());
                return Task::none();
            }
            Event::ProjectStarted { conv, path, file } => {
                // The conversation moved into its new project, so the sidebar and notes follow it there.
                if self.state.project(path).is_none() {
                    self.state.projects.push(crate::store::Project { path: path.clone(), last_session: Some(file.clone()) });
                    self.state.projects.sort_by_key(|p| project_name(&p.path).to_lowercase());
                }
                self.state.current_project = Some(path.clone());
                self.save();
                if let Some(chat) = self.chat.as_mut().filter(|c| &c.id == conv) {
                    chat.cwd = path.clone();
                    chat.file = file.clone();
                }
                self.toast(format!("Started the project {}", project_name(path)));
                self.notes.set_project(Some(path.join(&self.state.settings.notes_folder)));
                return Task::batch([self.load_sessions(), self.notes.reload()]);
            }
            Event::NotesSaved { notes, .. } => {
                self.toast(format!("Saved to notes: {}", notes.join(", ")));
                return self.notes.reload();
            }
            _ => {}
        }
        let conv = match &event {
            Event::Activity { conv, .. }
            | Event::Delta { conv, .. }
            | Event::Message { conv, .. }
            | Event::ToolStarted { conv, .. }
            | Event::ToolOutput { conv, .. }
            | Event::Approval { conv, .. }
            | Event::Compacted { conv, .. }
            | Event::Titled { conv, .. }
            | Event::Settled { conv, .. } => conv.clone(),
            Event::Error { conv: Some(conv), .. } => conv.clone(),
            _ => return Task::none(),
        };
        let settled = matches!(event, Event::Settled { .. });
        // The conversation file exists from the user's message on, so the sidebar and the resume point follow
        // it right away rather than when the task ends.
        let saved = matches!(event, Event::Message { message: crate::agent::conversation::Message::User(_), .. } | Event::Titled { .. });
        let Some(chat) = self.chat.as_mut().filter(|c| c.id == conv) else {
            return if settled { self.load_sessions() } else { Task::none() };
        };
        chat.apply(event);
        if settled || saved {
            let (project, file) = (chat.cwd.clone(), chat.file.clone());
            if file.exists() {
                self.remember_last(&project, file);
            }
            return self.load_sessions();
        }
        Task::none()
    }

    // ---- view ----

    fn view(&self) -> Element<'_, Message> {
        let theme = self.theme();
        let body: Element<'_, Message> = if self.needs_setup() {
            self.setup.view(&self.state).map(Message::Setup)
        } else {
            let mut main = row![self.sidebar(), rule::vertical(1).style(theme::divider), self.main_area(&theme)];
            if self.notes.open && self.current_project().is_some() {
                main = main.push(rule::vertical(1).style(theme::divider));
                main = main.push(self.notes.view(&theme));
            }
            main.height(Fill).into()
        };
        let mut layout = column![self.topbar(), rule::horizontal(1).style(theme::divider)];
        if let Some((version, url)) = &self.update_notice {
            layout = layout.push(
                container(
                    row![
                        text(format!("Scoobert {version} is available.")).size(13),
                        button(text("Open the release page").size(13)).style(theme::link).on_press(Message::OpenUrl(url.clone())),
                        space::horizontal(),
                        button(icon(Icon::Close, 14.0)).style(theme::ghost).on_press(Message::Update(None)),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                )
                .padding([6, 16])
                .style(theme::banner),
            );
        }
        layout = layout.push(body);
        let base = container(layout).width(Fill).height(Fill).style(theme::app);

        let mut layers = stack![base];
        if let Some(panel) = &self.settings {
            let isolation = self.host.as_ref().map(|h| h.isolation.describe()).unwrap_or_default();
            let ctx = settings::ViewCtx { state: &self.state, isolation, models: &self.models, download: &self.setup.download, queue: &self.setup.queue };
            layers = layers.push(modal(panel.view(ctx).map(Message::Settings), 760.0));
        }
        if let Some(c) = &self.confirm {
            layers = layers.push(modal(confirm_dialog(c), 420.0));
        }
        if let Some((t, _)) = &self.toast {
            layers = layers.push(
                container(container(text(t.clone()).size(13)).padding([8, 14]).max_width(560).style(theme::toast))
                    .width(Fill)
                    .height(Fill)
                    .align_x(Alignment::Center)
                    .align_y(Alignment::End)
                    .padding(24),
            );
        }
        layers.into()
    }

    fn logo(&self, target: f32) -> Element<'_, Message> {
        // Whole device pixels per source pixel keep the pixel art sharp at any display scale.
        let k = ((target * self.scale) / 48.0).floor().max(1.0);
        let (w, h) = (47.0 * k / self.scale, 48.0 * k / self.scale);
        image(self.logo.clone()).filter_method(image::FilterMethod::Nearest).width(w).height(h).into()
    }

    fn topbar(&self) -> Element<'_, Message> {
        let brand = container(row![self.logo(36.0), text("Scoobert").size(19).font(fonts::ui_semibold())].spacing(10).align_y(Alignment::Center))
            .width(SIDEBAR_WIDTH)
            .padding([0, 16]);
        let mut crumbs = row![].spacing(8).align_y(Alignment::Center);
        let place = self.current_project().map(|p| project_name(&p)).unwrap_or_else(|| "No project".into());
        crumbs = crumbs.push(text(place).size(14).font(fonts::ui_semibold()));
        if let Some(c) = self.chat.as_ref().filter(|c| !c.is_empty()) {
            crumbs = crumbs.push(text("/").size(14).style(theme::muted));
            crumbs = crumbs.push(text(clip(&c.title, 70)).size(14).style(theme::muted).wrapping(text::Wrapping::None));
        }
        let choices = self.model_choices();
        let selected = self.current_model().map(|m| ModelChoice {
            name: m.name.clone(),
            label: if m.provider.is_empty() { m.label.clone() } else { format!("{} ({})", m.label, m.provider) },
        });
        let model_pick = pick_list(choices, selected, Message::SetModel)
            .placeholder("No model")
            .width(Length::Shrink)
            .padding([5, 10])
            .text_size(13)
            .style(theme::select)
            .menu_style(theme::menu);
        let thinking = self.chat.as_ref().map(|c| c.thinking).unwrap_or(self.state.settings.thinking);
        let reasoning = self.current_model().is_none_or(|m| m.reasoning);
        let mut right = row![text("Model").size(13).style(theme::muted), model_pick].spacing(8).align_y(Alignment::Center);
        if reasoning {
            right = right.push(text("Thinking").size(13).style(theme::muted));
            right = right.push(
                pick_list(Thinking::ALL, Some(thinking), Message::SetThinking)
                    .padding([5, 10])
                    .text_size(13)
                    .style(theme::select)
                    .menu_style(theme::menu),
            );
        }
        let notes_style = if self.notes.open { theme::secondary } else { theme::ghost };
        right = right.push(
            button(row![icon(Icon::Panel, 16.0), text("Notes").size(13)].spacing(6).align_y(Alignment::Center))
                .padding([6, 12])
                .style(notes_style)
                // Notes belong to a project, so the pane has nothing to show without one.
                .on_press_maybe(self.current_project().is_some().then_some(Message::Notes(notes::Msg::TogglePane))),
        );
        row![brand, container(crumbs).width(Fill).clip(true), right.padding([0, 16])].height(56).align_y(Alignment::Center).into()
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let new_button = button(row![icon(Icon::Plus, 16.0), text("New conversation").size(14)].spacing(8).align_y(Alignment::Center))
            .width(Fill)
            .padding([8, 14])
            .style(theme::secondary)
            .on_press(Message::NewConversation);
        let header = row![
            text("Projects").size(13).style(theme::muted),
            space::horizontal(),
            tooltip(
                button(icon(Icon::FolderOpen, 16.0)).padding(4).style(theme::ghost).on_press(Message::AddProject),
                container(text("Add a project folder").size(12)).padding([4, 8]).style(theme::tooltip),
                tooltip::Position::Bottom,
            ),
        ]
        .align_y(Alignment::Center)
        .padding([0, 4]);

        let current = self.current_project();
        let mut list = Column::new().spacing(2);
        let general = current.is_none();
        list = list.push(
            button(
                row![icon(Icon::File, 15.0), text("No project").size(14).font(if general { fonts::ui_bold() } else { fonts::ui() })]
                    .spacing(8)
                    .align_y(Alignment::Center),
            )
            .width(Fill)
            .padding([6, 4])
            .style(theme::row_button)
            .on_press(Message::SelectNoProject),
        );
        if general {
            list = list.extend(self.sessions.iter().map(|s| self.session_row(s)));
            if self.sessions.is_empty() {
                list = list.push(container(text("Ask anything. Scoobert starts a project when a task needs one.").size(12).style(theme::muted)).padding([4, 12]));
            }
        }
        list = list.push(space().height(12));
        list = list.push(header);
        for p in &self.state.projects {
            let is_current = current.as_ref() == Some(&p.path);
            let name = button(text(p.name()).size(14).font(if is_current { fonts::ui_bold() } else { fonts::ui() }))
                .width(Fill)
                .padding([6, 4])
                .style(theme::row_button)
                .on_press(Message::SelectProject(p.path.clone()));
            let actions = row![
                space::horizontal(),
                button(icon(Icon::Folder, 14.0)).padding(4).style(theme::ghost).on_press(Message::RevealProject(p.path.clone())),
                button(icon(Icon::Close, 14.0)).padding(4).style(theme::ghost).on_press(Message::AskConfirm(Confirm::RemoveProject(p.path.clone()))),
            ]
            .align_y(Alignment::Center);
            list = list.push(hover(name, container(actions).height(Fill).align_y(Alignment::Center)));
            if is_current {
                for s in &self.sessions {
                    list = list.push(self.session_row(s));
                }
                if self.sessions.is_empty() {
                    list = list.push(container(text("No conversations yet").size(12).style(theme::muted)).padding([4, 12]));
                }
                list = list.push(space().height(8));
            }
        }
        if self.state.projects.is_empty() {
            list = list.push(container(text("Add an existing folder with the folder button, or let Scoobert start one.").size(12).style(theme::muted)).padding([4, 4]));
        }

        let (dot, status): (fn(&theme::Tokens) -> iced::Color, String) = self.status_line();
        let footer = row![
            container(space()).width(8).height(8).style(theme::dot(dot)),
            text(status).size(13).style(theme::muted).wrapping(text::Wrapping::None),
            space::horizontal(),
            button(icon(Icon::Gear, 18.0)).padding(4).style(theme::ghost).on_press(Message::OpenSettings(settings::Section::General)),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .padding([10, 16]);

        container(
            column![
                container(new_button).padding([14, 14]),
                scrollable(list.padding([0, 10])).height(Fill).style(theme::scrollbar),
                rule::horizontal(1).style(theme::divider),
                footer,
            ]
            .height(Fill),
        )
        .width(SIDEBAR_WIDTH)
        .height(Fill)
        .style(theme::sidebar)
        .into()
    }

    fn session_row<'a>(&'a self, s: &'a Summary) -> Element<'a, Message> {
        let selected = self.chat.as_ref().is_some_and(|c| c.file == s.file);
        if let Some((file, value)) = &self.renaming
            && *file == s.file
        {
            return text_input("Conversation name", value)
                .id(RENAME_ID)
                .on_input(Message::RenameInput)
                .on_submit(Message::CommitRename)
                .size(13)
                .padding([6, 8])
                .style(theme::input)
                .into();
        }
        // The title gets the space the time leaves and is cut there, so the two never overlap.
        let label = row![
            container(text(clip(&s.title, 30)).size(13).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            text(ago(s.modified)).size(12).style(theme::muted).wrapping(text::Wrapping::None),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let item = button(label).width(Fill).padding([6, 10]).style(theme::list_item(selected)).on_press(Message::OpenConversation(s.file.clone()));
        let bar = container(space()).width(2).height(Fill).style(if selected { theme::accent_bar } else { |_: &Theme| container::Style::default() });
        let base = row![bar, item].height(32);
        let actions = container(
            row![
                button(icon(Icon::Pencil, 14.0)).padding(4).style(theme::ghost).on_press(Message::StartRename(s.file.clone(), s.title.clone())),
                button(icon(Icon::Trash, 14.0)).padding(4).style(theme::ghost).on_press(Message::AskConfirm(Confirm::DeleteConversation(s.file.clone()))),
            ]
            .spacing(2),
        )
        .padding([0, 4])
        .style(theme::sidebar);
        hover(base, container(actions).width(Fill).height(Fill).align_x(Alignment::End).align_y(Alignment::Center)).into()
    }

    fn status_line(&self) -> (fn(&theme::Tokens) -> iced::Color, String) {
        if let Some(m) = self.current_model()
            && !m.provider.is_empty()
        {
            return (|t| t.ok, format!("Using {}", m.provider));
        }
        match &self.server {
            ServerStatus::Stopped => (|t| t.muted, "Model not loaded".into()),
            ServerStatus::Loading(m) => (|t| t.warn, format!("Loading {m}")),
            ServerStatus::Ready(m) => (|t| t.ok, format!("{m} loaded")),
            ServerStatus::Error(_) => (|t| t.danger, "The model stopped".into()),
        }
    }

    fn main_area(&self, theme: &Theme) -> Element<'_, Message> {
        let general = self.current_project().is_none();
        let intro = if general {
            "Ask anything. When a task needs its own files, Scoobert starts a project folder for it, or you can pick a project on the left."
        } else {
            "Scoobert reads and edits files in this project, runs commands, and keeps notes as it works."
        };
        let transcript: Element<'_, Message> = match &self.chat {
            Some(c) if !c.is_empty() => c.view(chat::markdown_settings(theme)),
            Some(_) => center(
                column![
                    self.logo(96.0),
                    text("What should Scoobert work on?").size(18).font(fonts::ui_semibold()),
                    text(intro).size(13).style(theme::muted),
                ]
                .spacing(10)
                .align_x(Alignment::Center),
            )
            .into(),
            None => center(text("Opening...").style(theme::muted)).into(),
        };
        let mut col = column![transcript];
        if let Some(e) = self.server_error() {
            col = col.push(container(container(text(e).size(13)).padding([8, 12]).width(Fill).style(theme::error_box)).padding([0, 24]).max_width(868));
        }
        col = col.push(container(self.composer_view()).padding(iced::Padding { top: 8.0, right: 24.0, bottom: 4.0, left: 24.0 }).center_x(Fill));
        col = col.push(container(self.footer()).padding(iced::Padding { top: 0.0, right: 24.0, bottom: 10.0, left: 24.0 }).center_x(Fill));
        col.width(Fill).height(Fill).into()
    }

    fn server_error(&self) -> Option<String> {
        let local = self.current_model().is_some_and(|m| m.provider.is_empty());
        match &self.server {
            ServerStatus::Error(e) if local && !self.chat.as_ref().is_some_and(|c| c.error.is_some()) => Some(clip(e, 600)),
            _ => None,
        }
    }

    fn composer_view(&self) -> Element<'_, Message> {
        let running = self.chat.as_ref().is_some_and(|c| c.running);
        let input = editor(&self.composer)
            .id(COMPOSER_ID)
            .placeholder("Ask Scoobert")
            .on_action(Message::Composer)
            .key_binding(|kp| {
                if !matches!(kp.status, text_editor::Status::Focused { .. }) {
                    return None;
                }
                match kp.key.as_ref() {
                    Key::Named(Named::Enter) if !kp.modifiers.shift() => Some(Binding::Custom(Message::Send)),
                    _ => Binding::from_key_press(kp),
                }
            })
            .size(15)
            .padding(0)
            .min_height(24)
            .max_height(220)
            .font(fonts::ui())
            .style(theme::bare_editor);
        let mut attachments = row![].spacing(6);
        for (i, _) in self.images.iter().enumerate() {
            attachments = attachments.push(
                container(
                    row![text(format!("Image {}", i + 1)).size(12), button(icon(Icon::Close, 12.0)).padding(2).style(theme::ghost).on_press(Message::RemoveImage(i))]
                        .spacing(4)
                        .align_y(Alignment::Center),
                )
                .padding([2, 8])
                .style(theme::chip),
            );
        }
        let vision = self.current_model().is_some_and(|m| m.vision);
        let action: Element<'_, Message> = if running {
            button(row![icons::tinted(Icon::Stop, 15.0, |_| iced::Color::WHITE), text("Stop").size(14).font(fonts::ui_semibold())].spacing(6).align_y(Alignment::Center))
                .padding([7, 18])
                .style(theme::stop)
                .on_press(Message::Stop)
                .into()
        } else {
            let can_send = !self.composer.text().trim().is_empty() || !self.images.is_empty();
            button(row![text("Send").size(13), icons::tinted(Icon::ArrowRight, 14.0, |t| t.accent_text)].spacing(6).align_y(Alignment::Center))
                .padding([6, 14])
                .style(theme::primary)
                .on_press_maybe(can_send.then_some(Message::Send))
                .into()
        };
        let mut tools = row![].spacing(8).align_y(Alignment::Center);
        if vision {
            tools = tools.push(
                tooltip(
                    button(icon(Icon::Image, 16.0)).padding(4).style(theme::ghost).on_press(Message::AttachImage),
                    container(text("Attach images").size(12)).padding([4, 8]).style(theme::tooltip),
                    tooltip::Position::Top,
                ),
            );
        }
        tools = tools.push(text("Enter to send, Shift+Enter for a new line").size(12).style(theme::muted));
        tools = tools.push(space::horizontal());
        tools = tools.push(action);
        let mut col = Column::new().spacing(10);
        if self.web_offer {
            col = col.push(
                container(
                    column![
                        text("This sounds like it needs the web, and web search is off.").size(13).font(fonts::ui_semibold()),
                        text("With it on, Scoobert searches DuckDuckGo and reads pages. Your searches leave your computer.").size(12).style(theme::muted),
                        row![
                            button(text("Turn on web search and send").size(13)).padding([5, 12]).style(theme::primary).on_press(Message::WebOffer(true)),
                            button(text("Send without it").size(13)).padding([5, 12]).style(theme::secondary).on_press(Message::WebOffer(false)),
                        ]
                        .spacing(8),
                    ]
                    .spacing(6),
                )
                .padding(10)
                .width(Fill)
                .style(theme::banner),
            );
        }
        if !self.images.is_empty() {
            col = col.push(attachments);
        }
        col = col.push(input).push(tools);
        container(col).padding([12, 16]).max_width(820).width(Fill).style(theme::composer(false)).into()
    }

    fn footer(&self) -> Element<'_, Message> {
        let context = self.current_model().map(|m| m.context).unwrap_or(0) as u64;
        let used = self.chat.as_ref().map(|c| c.context).unwrap_or(0);
        let mut r = row![].spacing(10).align_y(Alignment::Center);
        if context > 0 {
            r = r.push(progress_bar(0.0..=context as f32, used.min(context) as f32).length(100).girth(4).style(theme::meter));
            r = r.push(text(format!("{} of {} context", short_count(used), short_count(context))).size(12).style(theme::muted));
        }
        r = r.push(space::horizontal());
        r = r.push(
            pick_list(Approvals::ALL, Some(self.state.settings.approvals), Message::SetApprovals)
                .text_size(12)
                .padding([3, 8])
                .style(theme::quiet_select)
                .menu_style(theme::menu),
        );
        container(r).max_width(820).width(Fill).into()
    }
}

/// Requests that usually need the internet, which trigger the offer to turn on web search.
static NEEDS_WEB: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(concat!(
        r"(?i)\b(search|look)\s+(the\s+)?(web|internet|online)\b|\blook\s+(it|this|that|them|up)\b|\bgoogle\b|\bresearch\b",
        r"|\b(on|from)\s+the\s+(web|internet)\b|\bonline\b|\blatest\s+(version|release|news|docs)\b|\bnews\s+(about|on)\b",
        r"|\bhow\s+(do|are|did)\s+other\s+(people|projects|developers|apps)\b|\bwhat\s+do\s+people\s+(say|recommend|use)\b|\bbrowse\b|https?://",
    ))
    .unwrap()
});

fn project_name(p: &std::path::Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| crate::paths::display(p))
}

fn modal<'a>(content: Element<'a, Message>, width: f32) -> Element<'a, Message> {
    opaque(
        mouse_area(center(opaque(container(content).max_width(width).width(Fill).style(theme::modal))).padding(24).style(theme::backdrop))
            .on_press(Message::CloseModal),
    )
}

fn confirm_dialog<'a>(c: &Confirm) -> Element<'a, Message> {
    let (title, body, action) = match c {
        Confirm::DeleteConversation(_) => ("Delete this conversation?", "The conversation file moves to the trash, so you can restore it from there.", "Delete"),
        Confirm::RemoveProject(_) => ("Remove this project from the list?", "Scoobert forgets the folder but keeps its files, notes, and conversations.", "Remove"),
    };
    column![
        text(title).size(16).font(fonts::ui_semibold()),
        text(body).size(13).style(theme::muted),
        row![
            space::horizontal(),
            button(text("Cancel").size(13)).padding([6, 14]).style(theme::secondary).on_press(Message::CloseModal),
            button(text(action).size(13)).padding([6, 14]).style(theme::danger).on_press(Message::Confirmed(c.clone())),
        ]
        .spacing(8),
    ]
    .spacing(12)
    .padding(20)
    .into()
}
