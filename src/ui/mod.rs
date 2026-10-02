//! The Scoobert window: projects and conversations on the left, the conversation in the middle, notes on the right.

pub mod chat;
pub mod fonts;
pub mod graph;
pub mod icons;
pub mod lab;
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
use crate::i18n::{tr, trf};
use crate::llama::{ServerStatus, SharedSettings};
use crate::store::{Approvals, PlanFirst, State, ThemeChoice, Thinking};
use crate::update::Release;
use crate::util::{ago, clip, short_count};
use chat::Chat;
use icons::{Icon, icon};

const SIDEBAR_WIDTH: f32 = 290.0;
/// The chat keeps at least this width. A narrower window hides the sidebar first, then narrows the notes pane.
const CHAT_MIN: f32 = 440.0;
const NOTES_MIN: f32 = 300.0;
/// Below this window width the top bar drops its text labels.
const COMPACT_WIDTH: f32 = 1100.0;
/// The top bar's sidebar button and logo when the sidebar is hidden.
const BRAND_COMPACT: f32 = 104.0;
/// One of the minimize, maximize, and close buttons.
const CAPTION_BUTTON: f32 = 48.0;
const CAPTION_HEIGHT: f32 = 40.0;
/// The glyphs fill about a quarter of a button's width, as most programs draw them.
const CAPTION_GLYPH: f32 = 22.0;

/// Between the muted and the text color, so the window buttons are easy to find without standing out.
fn caption_glyph(t: &theme::Tokens) -> iced::Color {
    theme::mix(t.muted, t.text, 0.35)
}
/// The minimize, maximize, and close buttons.
const CAPTION_WIDTH: f32 = 3.0 * CAPTION_BUTTON;
/// Below this window width the top bar shows the project without the conversation title, and tightens the rest.
const LOCATION_WIDTH: f32 = 900.0;
const COMPOSER_ID: &str = "composer";
const RENAME_ID: &str = "rename";
const PROJECT_SEARCH_ID: &str = "project-search";

pub fn run() -> iced::Result {
    let state = State::load();
    // The default font follows the language, so the language is set before the window is built.
    crate::i18n::set(state.settings.language().code);
    let saved = state.window.map(|(w, h)| Size::new(w, h)).unwrap_or(Size::new(1310.0, 730.0));
    // Looking for an NVIDIA card runs a program, so it happens while the window opens.
    std::thread::spawn(crate::llama::cuda::detect);
    let (size, position) = place_window(saved);
    iced::application(App::new, App::update, App::view)
        .window(window::Settings {
            size,
            position,
            min_size: Some(Size::new(760.0, 480.0)),
            icon: window::icon::from_file_data(include_bytes!("../../assets/icon.png"), None).ok(),
            exit_on_close_request: false,
            // The top bar holds the window buttons, so the system title bar is not drawn.
            decorations: false,
            #[cfg(windows)]
            platform_specific: window::settings::PlatformSpecific { undecorated_shadow: true, ..Default::default() },
            // Matches the .desktop file, so Linux desktops show the right icon and group the window.
            #[cfg(target_os = "linux")]
            platform_specific: window::settings::PlatformSpecific { application_id: "scoobert".into(), ..Default::default() },
            ..window::Settings::default()
        })
        .exit_on_close_request(false)
        .settings(iced::Settings { default_text_size: 14.into(), ..iced::Settings::default() })
        .default_font(fonts::ui())
        .title(App::title)
        .theme(App::theme)
        .subscription(App::subscription)
        .run()
}

/// Shrinks the window to fit above the taskbar and centers it there. The size excludes the title bar and borders.
fn place_window(size: Size) -> (Size, window::Position) {
    let fit = |s: Size| Size::new(s.width.max(760.0), s.height.max(480.0));
    let Some((x, y, width, height)) = crate::sys::work_area() else { return (fit(size), window::Position::Centered) };
    let (frame_w, frame_h) = (16.0, 40.0);
    let size = fit(Size::new(size.width.min(width - frame_w), size.height.min(height - frame_h)));
    let corner = iced::Point::new(x + ((width - size.width - frame_w) / 2.0).max(0.0), y + ((height - size.height - frame_h) / 2.0).max(0.0));
    (size, window::Position::Specific(corner))
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
    /// Go back to the user message at `index`, with `files` files the model changed after it.
    Rewind { index: usize, text: String, files: usize },
    /// Install an update while a task is running.
    InstallUpdate,
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
    /// A press on an empty part of the top bar: a drag moves the window, and a second press soon after maximizes it.
    TitlePressed,
    WindowMenu,
    Minimize,
    ToggleMaximize,
    Maximized(bool),
    /// The system's id for the window, which is its handle on Windows.
    RawWindow(u64),
    /// The window moved or came to the front.
    WindowMoved,
    ResizeFrom(window::Direction),
    CloseHover(bool),
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
    /// Starts a conversation in a project, or in Chats when the project is None.
    NewConversationIn(Option<PathBuf>),
    /// Starts a conversation with no project.
    NewChat,
    SearchProjects(String),
    SearchIndex(Vec<(Option<PathBuf>, Summary)>),
    /// Opens a conversation from the search results, in its project or in Chats when the project is None.
    OpenIn(Option<PathBuf>, PathBuf),
    OpenConversation(PathBuf),
    Opened(Result<Box<Snapshot>, String>),
    StartRename(PathBuf, String),
    RenameInput(String),
    CommitRename,
    AskConfirm(Confirm),
    Confirmed(Confirm),
    AskRewind(usize, String),
    RewindFiles(bool),
    Rewound(Result<(Box<Snapshot>, Vec<String>), String>, String),

    Composer(text_editor::Action),
    Send,
    /// Gives the queued messages to the model while it thinks, instead of after its current step.
    SendNow,
    /// Resumes an interrupted task with a hidden note that explains what happened.
    Continue,
    Stop,
    AttachImage,
    ImagesPicked(Vec<Image>),
    FileDropped(PathBuf),
    RemoveImage(usize),
    SetModel(ModelChoice),
    SetThinking(Thinking),
    SetLanguage(&'static crate::i18n::Language),
    LanguageMenu(bool),
    ApprovalsMenu(bool),
    ToggleSidebar,
    CloseDrawer,
    SetApprovals(Approvals),
    SetPlanFirst(PlanFirst),
    Approve(u64, Decision),
    Toggle(String),
    Link(String),
    Copy(String),
    /// Asks the model to update the project's notes from the open conversation.
    UpdateNotes,
    /// The answer to the offer to turn on web search before sending: true turns it on.
    WebOffer(bool),
    /// Turns on loading models from disk and continues, or dismisses the offer.
    DiskOffer(bool),

    Notes(notes::Msg),
    Settings(settings::Msg),
    Setup(setup::Msg),
    OpenSettings(settings::Section),
    CloseModal,

    Toast(String),
    Update(Option<Release>),
    CheckUpdates,
    Checked(Result<Option<Release>, String>),
    InstallUpdate,
    UpdateProgress(u64, u64),
    UpdateReady(Result<PathBuf, String>),
    UpdateLaunched(Result<(), String>),
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
    maximized: bool,
    raw_window: Option<u64>,
    /// When the top bar was last pressed, to tell a double click from two drags.
    title_pressed: Option<Instant>,
    close_hover: bool,
    server: ServerStatus,
    models: Vec<ModelOption>,
    sessions: Vec<Summary>,
    search: String,
    /// Every conversation in Chats and the projects, loaded when a search starts.
    search_index: Option<Vec<(Option<PathBuf>, Summary)>>,
    renaming: Option<(PathBuf, String)>,
    chat: Option<Chat>,
    composer: text_editor::Content,
    images: Vec<Image>,
    notes: notes::Pane,
    settings: Option<settings::Panel>,
    setup: setup::Setup,
    confirm: Option<Confirm>,
    toast: Option<(String, Instant)>,
    update_notice: Option<Release>,
    /// Bytes downloaded and the total while an update downloads.
    update_progress: Option<(u64, u64)>,
    /// A message that looks like it needs the web is waiting while the user decides about web search.
    web_offer: bool,
    /// The conversation whose model did not fit in memory, which shows the offer to load it from disk.
    disk_offer: Option<String>,
    language_menu: bool,
    approvals_menu: bool,
    /// The sidebar open over the chat, in a window too narrow to show it beside the chat.
    sidebar_drawer: bool,
    /// The interface font when the window was built, which text without its own font keeps until a restart.
    start_font: iced::Font,
    font_noted: bool,
    /// The model whose saved prompts are being built ahead of time.
    preparing: Option<(String, u64)>,
    /// The conversation whose model and cached prompt were last loaded because the user started typing.
    warmed: Option<String>,
    /// Whether a rewind also puts back the files the model changed.
    rewind_files: bool,
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
        crate::i18n::set(state.settings.language().code);
        let shared = Arc::new(RwLock::new(state.settings.clone()));
        let logo = image::Handle::from_bytes(include_bytes!("../../assets/logo.png").as_slice());
        let notes = notes::Pane::new(!state.notes_closed);
        let setup = setup::Setup::new(&state.settings.models_dir());
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
            maximized: false,
            raw_window: None,
            title_pressed: None,
            close_hover: false,
            server: ServerStatus::Stopped,
            models: Vec::new(),
            sessions: Vec::new(),
            search: String::new(),
            search_index: None,
            renaming: None,
            chat: None,
            composer: text_editor::Content::new(),
            images: Vec::new(),
            notes,
            settings: None,
            setup,
            confirm: None,
            toast: None,
            update_notice: None,
            update_progress: None,
            web_offer: false,
            disk_offer: None,
            language_menu: false,
            approvals_menu: false,
            sidebar_drawer: false,
            start_font: fonts::ui(),
            font_noted: false,
            rewind_files: true,
            warmed: None,
            preparing: None,
        };
        let mut tasks = vec![system::theme().map(Message::Mode)];
        if crate::update::due(app.state.update_checked_at) {
            app.state.update_checked_at = crate::util::now_millis();
            app.save();
            tasks.push(Task::perform(crate::update::check(), |r| Message::Update(r.ok().flatten())));
        }
        (app, Task::batch(tasks))
    }

    /// The window's name in the taskbar and the window switcher. The top bar already shows the project and title.
    fn title(&self) -> String {
        "Scoobert".into()
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
                window::Event::Moved(_) | window::Event::Focused => Some(Message::WindowMoved),
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

    /// Text without its own font uses the one chosen at startup, so a language that needs another font asks for a
    /// restart, once.
    fn note_font_change(&mut self) {
        if fonts::ui() != self.start_font && !self.font_noted {
            self.font_noted = true;
            self.toast = Some((tr("Restart Scoobert to use this language's font everywhere.").into(), Instant::now()));
        }
    }

    /// Whether the sidebar sits beside the chat, and the notes pane's width, for the window's current width.
    fn panes(&self) -> (bool, f32) {
        let width = self.window_size.map_or(f32::MAX, |s| s.width);
        let docked = !self.state.sidebar_closed && self.sidebar_fits();
        let side = if docked { SIDEBAR_WIDTH + 1.0 } else { 0.0 };
        let notes = if self.notes.open { self.notes.width().min(width - side - 1.0 - CHAT_MIN).max(NOTES_MIN) } else { 0.0 };
        (docked, notes)
    }

    /// Whether the window leaves the chat its minimum width with the sidebar and the smallest notes pane beside it.
    fn sidebar_fits(&self) -> bool {
        let width = self.window_size.map_or(f32::MAX, |s| s.width);
        let notes = if self.notes.open { 1.0 + NOTES_MIN } else { 0.0 };
        width - SIDEBAR_WIDTH - 1.0 - notes >= CHAT_MIN
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

    /// Loads the default model and builds its saved prompts for Chats and the open project: once after setup, once
    /// after each update changes the prompt, and at every start when the user turned that on. A model that does not
    /// fit in free memory is left for later.
    fn prepare_default_model(&mut self) {
        let Some(host) = &self.host else { return };
        if !self.state.setup_done || self.setup.download.is_some() || !self.setup.queue.is_empty() {
            return;
        }
        let version = env!("CARGO_PKG_VERSION");
        if self.state.prepared_version == version && !self.state.settings.preload_model {
            return;
        }
        let mut places = vec![crate::paths::projects_root()];
        places.extend(self.current_project());
        if host.preload(&self.state.settings.model, places) {
            self.state.prepared_version = version.to_string();
            self.save();
        }
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
        let Some(host) = self.host.clone() else { return };
        self.models = host.models();
        // A fresh install names a model it may not have downloaded, so the largest downloaded model that fits the
        // computer's memory becomes the default, or the smallest when none fits.
        if self.models.iter().any(|m| m.usable && m.name == self.state.settings.model) {
            return;
        }
        let total = crate::sys::total_memory();
        let local = || self.models.iter().filter(|m| m.usable && m.provider.is_empty());
        let pick = local()
            .filter(|m| m.memory_needed <= total)
            .max_by_key(|m| m.memory_needed)
            .or_else(|| local().min_by_key(|m| m.memory_needed))
            .or_else(|| self.models.iter().find(|m| m.usable));
        let Some(name) = pick.map(|m| m.name.clone()) else { return };
        self.state.settings.model = name.clone();
        self.save();
        // A conversation with no messages yet takes the new default too.
        if let Some(chat) = self.chat.as_mut()
            && chat.is_empty()
            && !self.models.iter().any(|m| m.usable && m.name == chat.model)
            && host.set_model(&chat.id, &name).is_ok()
        {
            chat.model = name;
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
        let last = match &path {
            Some(p) => self.state.project(p).and_then(|p| p.last_session.clone()),
            None => self.state.general_last_session.clone(),
        };
        self.switch_to(path, last.filter(|f| f.exists()))
    }

    /// Switches to a project, or to no project when path is None, and opens `file` there or a new conversation.
    fn switch_to(&mut self, path: Option<PathBuf>, file: Option<PathBuf>) -> Task<Message> {
        self.search.clear();
        self.search_index = None;
        self.state.current_project = path.clone();
        self.save();
        self.chat = None;
        self.sessions.clear();
        self.notes.set_project(path.map(|p| p.join(&self.state.settings.notes_folder)));
        Task::batch([self.load_sessions(), self.open(file), self.notes.reload()])
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
        // Picking something in the sidebar drawer closes it.
        if matches!(
            message,
            Message::OpenConversation(_)
                | Message::OpenIn(..)
                | Message::NewChat
                | Message::NewConversationIn(_)
                | Message::SelectProject(_)
                | Message::SelectNoProject
                | Message::OpenSettings(_)
        ) {
            self.sidebar_drawer = false;
        }
        match message {
            Message::HostReady(HostHandle(host)) => {
                self.host = Some(host);
                self.refresh_models();
                if !self.models.is_empty() && !self.state.setup_done {
                    self.state.setup_done = true;
                    self.save();
                }
                self.prepare_default_model();
                return self.select_project(self.current_project());
            }
            Message::Host(event) => return self.on_host_event(event),
            Message::Mode(mode) => self.system_dark = mode != iced::theme::Mode::Light,
            Message::WindowOpened(id) => {
                self.window = Some(id);
                return Task::batch([window::scale_factor(id).map(Message::Scale), window::raw_id::<Message>(id).map(Message::RawWindow)]);
            }
            Message::RawWindow(raw) => {
                self.raw_window = Some(raw);
                crate::sys::round_corners(raw, self.maximized);
            }
            Message::WindowMoved => {
                if let Some(raw) = self.raw_window {
                    crate::sys::follow_window(raw);
                }
            }
            Message::Scale(s) => self.scale = s,
            Message::Resized(size) => {
                self.window_size = Some(size);
                if let Some(id) = self.window {
                    return window::is_maximized(id).map(Message::Maximized);
                }
            }
            // Every resize ends here, so a clipped window's rounded shape follows its new size.
            Message::Maximized(on) => {
                self.maximized = on;
                if let Some(raw) = self.raw_window {
                    crate::sys::round_corners(raw, on);
                }
            }
            Message::TitlePressed => {
                let Some(id) = self.window else { return Task::none() };
                let double = self.title_pressed.is_some_and(|t| t.elapsed() < Duration::from_millis(400));
                self.title_pressed = (!double).then(Instant::now);
                return if double { window::toggle_maximize(id) } else { window::drag(id) };
            }
            Message::WindowMenu => return self.window.map(window::show_system_menu).unwrap_or_else(Task::none),
            Message::Minimize => return self.window.map(|id| window::minimize(id, true)).unwrap_or_else(Task::none),
            Message::ToggleMaximize => return self.window.map(window::toggle_maximize).unwrap_or_else(Task::none),
            Message::ResizeFrom(direction) => return self.window.map(|id| window::drag_resize(id, direction)).unwrap_or_else(Task::none),
            Message::CloseHover(on) => self.close_hover = on,
            Message::CloseRequested(id) => {
                if let Some(size) = self.window_size {
                    self.state.window = Some((size.width, size.height));
                }
                let flush = self.notes.flush();
                self.save();
                let host = self.host.clone();
                // The window goes away at once while the model server stops behind it.
                return window::set_mode(id, window::Mode::Hidden).chain(flush).chain(Task::perform(
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
                } else if !self.search.is_empty() {
                    self.search.clear();
                    self.search_index = None;
                } else if self.chat.as_ref().is_some_and(|c| c.running) {
                    return self.update(Message::Stop);
                }
            }
            Message::Noop => {}

            Message::AddProject => {
                return Task::perform(
                    async { rfd::AsyncFileDialog::new().set_title(tr("Choose a project folder")).pick_folder().await.map(|h| h.path().to_path_buf()) },
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
                self.search.clear();
                self.search_index = None;
            }
            Message::SelectNoProject => {
                if self.current_project().is_some() {
                    return self.select_project(None);
                }
                self.search.clear();
                self.search_index = None;
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
            Message::NewConversationIn(place) => {
                if self.current_project() == place {
                    return self.open(None);
                }
                return self.switch_to(place, None);
            }
            Message::SearchProjects(query) => {
                let starting = self.search.trim().is_empty() && !query.trim().is_empty();
                self.search = query;
                if self.search.trim().is_empty() {
                    self.search_index = None;
                } else if starting {
                    return self.load_search_index();
                }
            }
            Message::SearchIndex(index) => {
                if !self.search.trim().is_empty() {
                    self.search_index = Some(index);
                }
            }
            Message::OpenIn(place, file) => {
                if self.current_project() == place {
                    self.search.clear();
                    self.search_index = None;
                    return self.open(Some(file));
                }
                return self.switch_to(place, Some(file));
            }
            Message::NewChat => {
                if self.current_project().is_none() {
                    return self.open(None);
                }
                return self.switch_to(None, None);
            }
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
            Message::AskRewind(index, text) => {
                let files = match (&self.host, &self.chat) {
                    (Some(host), Some(chat)) => host.files_changed_after(&chat.id, index),
                    _ => 0,
                };
                self.rewind_files = true;
                self.confirm = Some(Confirm::Rewind { index, text, files });
            }
            Message::RewindFiles(on) => self.rewind_files = on,
            Message::Rewound(Ok((snap, problems)), text) => {
                self.chat = Some(Chat::from_snapshot(*snap));
                self.composer = text_editor::Content::with_text(&text);
                self.toast(if problems.is_empty() { tr("Rewound. Edit the message and send it again.").to_string() } else { problems.join(" ") });
                return Task::batch([self.load_sessions(), operation::focus(COMPOSER_ID)]);
            }
            Message::Rewound(Err(e), _) => self.toast(e),
            Message::Confirmed(c) => {
                self.confirm = None;
                match c {
                    Confirm::Rewind { index, text, .. } => {
                        let (Some(host), Some(chat)) = (self.host.clone(), self.chat.as_ref()) else { return Task::none() };
                        let (id, restore) = (chat.id.clone(), self.rewind_files);
                        return Task::perform(
                            async move { host.rewind(&id, index, restore).await.map(|(s, p)| (Box::new(s), p)).map_err(|e| format!("{e:#}")) },
                            move |r| Message::Rewound(r, text.clone()),
                        );
                    }
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
                    Confirm::InstallUpdate => return self.start_update(),
                }
            }

            Message::Composer(action) => {
                // Typing is the sign a message is coming, so the model loads then rather than when a conversation opens.
                if matches!(action, text_editor::Action::Edit(_))
                    && let (Some(host), Some(chat)) = (&self.host, &self.chat)
                    && self.warmed.as_deref() != Some(chat.id.as_str())
                {
                    host.warm(&chat.id);
                    self.warmed = Some(chat.id.clone());
                }
                self.composer.perform(action);
            }
            Message::Send => return self.send(true),
            Message::SendNow => {
                if let (Some(host), Some(chat)) = (&self.host, &self.chat)
                    && !host.deliver_now(&chat.id)
                {
                    self.toast(tr("Scoobert is writing a tool call, so the message waits until that step finishes."));
                }
            }
            Message::Continue => {
                let (Some(host), Some(chat)) = (self.host.clone(), self.chat.as_mut()) else { return Task::none() };
                match host.prompt(&chat.id, tr("Continue").into(), Vec::new(), true) {
                    Ok(()) => {
                        chat.pending = Some(tr("Continue").into());
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
            Message::DiskOffer(enable) => {
                self.disk_offer = None;
                if enable {
                    self.state.settings.models_from_disk = true;
                    self.save();
                    // The message that failed is still unanswered, so Continue answers it rather than adding another.
                    return self.update(Message::Continue);
                }
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
                            .add_filter(tr("Images"), &["png", "jpg", "jpeg", "gif", "webp"])
                            .set_title(tr("Attach images"))
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
                self.warmed = None;
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
                self.approvals_menu = false;
                self.state.settings.approvals = a;
                self.save();
            }
            Message::ApprovalsMenu(open) => self.approvals_menu = open,
            Message::SetPlanFirst(plan) => {
                self.state.settings.plan_first = plan;
                self.save();
            }
            Message::SetLanguage(language) => {
                crate::i18n::set(language.code);
                self.state.settings.language = language.code.to_string();
                self.language_menu = false;
                self.save();
                self.note_font_change();
            }
            Message::LanguageMenu(open) => self.language_menu = open,
            Message::ToggleSidebar => {
                if self.panes().0 {
                    self.state.sidebar_closed = true;
                    self.save();
                } else if self.state.sidebar_closed && self.sidebar_fits() {
                    self.state.sidebar_closed = false;
                    self.save();
                } else {
                    self.sidebar_drawer = !self.sidebar_drawer;
                }
            }
            Message::CloseDrawer => self.sidebar_drawer = false,
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
                self.toast(tr("Copied."));
                return iced::clipboard::write(text);
            }
            Message::UpdateNotes => {
                let (Some(host), Some(chat)) = (self.host.clone(), self.chat.as_mut()) else { return Task::none() };
                match host.update_notes(&chat.id) {
                    Ok(()) => {
                        chat.pending = Some(tr("Update the notes from this conversation.").into());
                        chat.running = true;
                        chat.interrupted = false;
                        chat.error = None;
                        // The notes pane shows the pages as the model changes them.
                        if !self.notes.open {
                            return self.notes.update(notes::Msg::TogglePane, self.current_project().as_deref());
                        }
                    }
                    Err(e) => chat.error = Some(format!("{e:#}")),
                }
            }

            Message::Notes(msg) => {
                let project = self.current_project();
                let task = self.notes.update(msg, project.as_deref());
                self.state.notes_closed = !self.notes.open;
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
            Message::CheckUpdates => {
                self.state.update_checked_at = crate::util::now_millis();
                self.save();
                return Task::perform(crate::update::check(), |r| Message::Checked(r.map_err(|e| format!("{e:#}"))));
            }
            Message::Checked(Ok(Some(release))) => {
                // The banner offers the update, and the settings panel would cover it.
                self.settings = None;
                self.update_notice = Some(release);
            }
            Message::Checked(Ok(None)) => self.toast(trf("Scoobert {version} is the newest version.", &[("version", &env!("CARGO_PKG_VERSION"))])),
            Message::Checked(Err(e)) => self.toast(e),
            Message::InstallUpdate => {
                if self.chat.as_ref().is_some_and(|c| c.running) {
                    self.confirm = Some(Confirm::InstallUpdate);
                } else {
                    return self.start_update();
                }
            }
            Message::UpdateProgress(done, total) => self.update_progress = Some((done, total)),
            Message::UpdateReady(Ok(file)) => {
                let Some(kind) = crate::update::install_kind() else { return Task::none() };
                if let Some((_, total)) = self.update_progress {
                    self.update_progress = Some((total, total));
                }
                if let Some(size) = self.window_size {
                    self.state.window = Some((size.width, size.height));
                }
                let flush = self.notes.flush();
                self.save();
                let host = self.host.clone();
                // The model server and running tasks stop first so the installer can replace their files.
                return flush.chain(Task::perform(
                    async move {
                        if let Some(h) = host {
                            h.shutdown().await;
                        }
                        crate::update::install(&file, &kind).map_err(|e| format!("{e:#}"))
                    },
                    Message::UpdateLaunched,
                ));
            }
            Message::UpdateReady(Err(e)) => {
                self.update_progress = None;
                self.toast(e);
            }
            Message::UpdateLaunched(Ok(())) => return self.window.map(window::close).unwrap_or_else(iced::exit),
            Message::UpdateLaunched(Err(e)) => {
                self.update_progress = None;
                self.toast(trf("Could not install the update. {error}", &[("error", &e)]));
            }
            Message::OpenUrl(url) => {
                if url.starts_with("https://") {
                    let _ = opener::open(url);
                }
            }
        }
        Task::none()
    }

    /// Downloads the release the banner offers and reports its progress to the banner.
    fn start_update(&mut self) -> Task<Message> {
        let (Some(asset), Some(kind)) = (self.update_notice.as_ref().and_then(|r| r.asset.clone()), crate::update::install_kind()) else {
            return Task::none();
        };
        self.update_progress = Some((0, asset.size));
        let stream = iced::stream::channel(16, async move |mut out: iced::futures::channel::mpsc::Sender<Message>| {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(u64, u64)>();
            let job = tokio::spawn(async move {
                crate::update::download(&asset, &kind, |done, total| {
                    let _ = tx.send((done, total));
                })
                .await
            });
            let mut job = std::pin::pin!(job);
            let mut shown = u64::MAX;
            let result = loop {
                tokio::select! {
                    Some((done, total)) = rx.recv() => {
                        let percent = done * 100 / total.max(1);
                        if percent != shown {
                            shown = percent;
                            let _ = out.send(Message::UpdateProgress(done, total)).await;
                        }
                    }
                    r = &mut job => break r,
                }
            };
            let result = match result {
                Ok(r) => r.map_err(|e| format!("{e:#}")),
                Err(e) => Err(e.to_string()),
            };
            let _ = out.send(Message::UpdateReady(result)).await;
        });
        Task::run(stream, std::convert::identity)
    }

    /// Applies what a settings or setup change requires elsewhere in the app.
    fn after_settings(&mut self, task: Task<Message>, effect: settings::Effect) -> Task<Message> {
        match effect {
            settings::Effect::None => {}
            settings::Effect::Saved => {
                self.save();
                self.note_font_change();
            }
            settings::Effect::ModelsChanged => {
                self.save();
                self.refresh_models();
                self.prepare_default_model();
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
        self.disk_offer = None;
        // While a task runs, the message waits for its current step, like a note passed in rather than a Stop.
        if let (Some(host), Some(chat)) = (self.host.clone(), self.chat.as_mut())
            && chat.running
        {
            let images = std::mem::take(&mut self.images);
            match host.queue(&chat.id, text.clone(), images) {
                Ok(()) => {
                    chat.queued.push(text);
                    self.composer = text_editor::Content::new();
                }
                Err(e) => chat.error = Some(format!("{e:#}")),
            }
            return Task::none();
        }
        if offer_web && !self.state.settings.web_access && NEEDS_WEB.is_match(&text) {
            self.web_offer = true;
            return Task::none();
        }
        let Some(host) = self.host.clone() else { return Task::none() };
        let Some(chat) = &mut self.chat else {
            self.toast(tr("Add a project folder first."));
            return Task::none();
        };
        if chat.running {
            return Task::none();
        }
        let images = std::mem::take(&mut self.images);
        let sent = match self.state.settings.plan_first {
            PlanFirst::Discuss | PlanFirst::Build if plans_first(chat) => host.plan(&chat.id, text.clone(), images, self.state.settings.plan_first == PlanFirst::Build),
            _ => host.prompt(&chat.id, text.clone(), images, false),
        };
        match sent {
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
            Event::Server(ServerStatus::GpuFailed(model)) => {
                // The model loads on the processor now, and later loads skip the card for it.
                if !self.state.settings.gpu_failed.contains(model) {
                    self.state.settings.gpu_failed.push(model.clone());
                    self.save();
                }
                self.toast(trf("The graphics card could not load {model}, so it runs on the processor.", &[("model", model)]));
                return Task::none();
            }
            Event::Server(ServerStatus::CudaFailed(model)) => {
                self.toast(trf("NVIDIA support could not load {model}, so Scoobert uses the card's default support until it restarts.", &[("model", model)]));
                return Task::none();
            }
            Event::Server(status) => {
                self.server = status.clone();
                return Task::none();
            }
            Event::Preparing(model) => {
                self.preparing = model.clone();
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
                self.toast(trf("Started the project {name}", &[("name", &project_name(path))]));
                self.notes.set_project(Some(path.join(&self.state.settings.notes_folder)));
                return Task::batch([self.load_sessions(), self.notes.reload()]);
            }
            Event::NotesSaved { notes, .. } => {
                self.toast(trf("Saved to notes: {notes}", &[("notes", &notes.join(", "))]));
                return self.notes.reload();
            }
            Event::DiskOffer { conv } => {
                self.disk_offer = Some(conv.clone());
                return Task::none();
            }
            _ => {}
        }
        let conv = match &event {
            Event::Activity { conv, .. }
            | Event::Delta { conv, .. }
            | Event::Replacing { conv }
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
        let stopped = matches!(event, Event::Settled { interrupted: true, .. });
        let Some(chat) = self.chat.as_mut().filter(|c| c.id == conv) else {
            return if settled { self.load_sessions() } else { Task::none() };
        };
        chat.apply(event);
        // Messages the task ended without reading go back to the message box after Stop, and otherwise start the
        // next task, since the task may have finished just as they were sent.
        let left = if settled { self.host.as_ref().map(|h| h.take_queued(&conv)).unwrap_or_default() } else { Vec::new() };
        if settled {
            chat.queued.clear();
        }
        if !left.is_empty() {
            let text = left.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("\n\n");
            let images: Vec<_> = left.into_iter().flat_map(|(_, i)| i).collect();
            if stopped {
                let draft = self.composer.text();
                let draft = draft.trim();
                self.composer = text_editor::Content::with_text(&if draft.is_empty() { text } else { format!("{text}\n\n{draft}") });
                self.images.extend(images);
            } else if let Some(host) = self.host.clone() {
                match host.prompt(&conv, text.clone(), images, false) {
                    Ok(()) => {
                        chat.pending = Some(text);
                        chat.running = true;
                    }
                    Err(e) => chat.error = Some(format!("{e:#}")),
                }
            }
        }
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
            let (docked, notes_width) = self.panes();
            let mut main = row![];
            if docked {
                main = main.push(self.sidebar()).push(rule::vertical(1).style(theme::divider));
            }
            main = main.push(self.main_area(&theme));
            if self.notes.open {
                main = main.push(rule::vertical(1).style(theme::divider));
                main = main.push(self.notes.view(&theme, notes_width));
            }
            main.height(Fill).into()
        };
        let mut layout = column![];
        if let Some(release) = &self.update_notice {
            let mut bar = row![text(trf("Scoobert {version} is available.", &[("version", &release.version)])).size(13)].spacing(8).align_y(Alignment::Center);
            match self.update_progress {
                Some((done, total)) if total > 0 && done >= total => bar = bar.push(text(tr("Installing. Scoobert restarts by itself.")).size(13).style(theme::muted)),
                Some((done, total)) if total > 0 => {
                    bar = bar.push(text(trf("Downloading, {percent}%", &[("percent", &(done * 100 / total))])).size(13).style(theme::muted))
                }
                Some(_) => bar = bar.push(text(tr("Downloading")).size(13).style(theme::muted)),
                None => {
                    // Only a copy that can replace itself gets an asset.
                    if release.asset.is_some() {
                        bar = bar.push(button(text(tr("Download and install")).size(13)).padding([4, 12]).style(theme::primary).on_press(Message::InstallUpdate));
                    }
                    bar = bar.push(button(text(tr("Open the release page")).size(13)).style(theme::link).on_press(Message::OpenUrl(release.page.clone())));
                    bar = bar.push(space::horizontal());
                    bar = bar.push(button(icon(Icon::Close, 14.0)).style(theme::ghost).on_press(Message::Update(None)));
                }
            }
            layout = layout.push(container(bar).width(Fill).padding([6, 16]).style(theme::banner));
        }
        layout = layout.push(body);
        let base = container(layout).width(Fill).height(Fill).style(theme::app);

        let mut layers = stack![base];
        if self.sidebar_drawer && !self.panes().0 && !self.needs_setup() {
            layers = layers.push(mouse_area(container(space()).width(Fill).height(Fill).style(theme::backdrop)).on_press(Message::CloseDrawer));
            layers = layers.push(container(container(self.sidebar()).height(Fill).style(theme::drawer)).height(Fill));
        }
        if self.language_menu {
            let mut list = Column::new().spacing(2);
            for language in crate::i18n::LANGUAGES {
                let mark: Element<'_, Message> =
                    if language == crate::i18n::current() { icons::tinted(Icon::Check, 14.0, |t| t.accent_ink).into() } else { space().width(14).into() };
                list = list.push(
                    button(row![text(language.name).size(13).width(Fill), mark].align_y(Alignment::Center))
                        .width(Fill)
                        .padding([6, 10])
                        .style(theme::row_button)
                        .on_press(Message::SetLanguage(language)),
                );
            }
            // A click anywhere else closes the menu.
            layers = layers.push(mouse_area(container(space()).width(Fill).height(Fill)).on_press(Message::LanguageMenu(false)));
            // The right padding lines the menu up under the language button, left of Notes and the window controls.
            let compact = self.window_size.is_some_and(|s| s.width < COMPACT_WIDTH);
            layers = layers.push(
                container(container(list).width(220).padding(4).style(theme::popover))
                    .width(Fill)
                    .align_x(Alignment::End)
                    .padding(iced::Padding { top: 4.0, right: if compact { 202.0 } else { 250.0 }, ..iced::Padding::ZERO }),
            );
        }
        if let Some(panel) = &self.settings {
            let isolation = self.host.as_ref().map(|h| h.isolation.describe()).unwrap_or_default();
            let ctx = settings::ViewCtx { state: &self.state, isolation, models: &self.models, download: &self.setup.download, queue: &self.setup.queue };
            layers = layers.push(modal(panel.view(ctx).map(Message::Settings), 760.0));
        }
        if let Some(c) = &self.confirm {
            layers = layers.push(modal(confirm_dialog(c, self.rewind_files), 460.0));
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
        // Dialogs cover the area below the top bar, so the window can still be moved and closed.
        let title_bar = mouse_area(self.topbar()).on_press(Message::TitlePressed).on_right_press(Message::WindowMenu);
        let framed = container(column![title_bar, rule::horizontal(1).style(theme::divider), layers]).width(Fill).height(Fill).style(theme::app);
        // macOS cannot start a resize from the app, and resizes a borderless window from its edges itself.
        if self.maximized || cfg!(target_os = "macos") {
            return framed.into();
        }
        stack![framed, resize_edges()].into()
    }

    fn logo(&self, target: f32) -> Element<'_, Message> {
        // Whole device pixels per source pixel keep the pixel art sharp at any display scale.
        let k = ((target * self.scale) / 48.0).floor().max(1.0);
        let (w, h) = (47.0 * k / self.scale, 48.0 * k / self.scale);
        image(self.logo.clone()).filter_method(image::FilterMethod::Nearest).width(w).height(h).into()
    }

    fn topbar(&self) -> Element<'_, Message> {
        let docked = self.panes().0;
        let compact = self.window_size.is_some_and(|s| s.width < COMPACT_WIDTH);
        let narrow = self.window_size.is_some_and(|s| s.width < LOCATION_WIDTH);
        let toggle = tooltip(
            button(icon(Icon::Sidebar, 16.0)).padding([6, 8]).style(if self.sidebar_drawer { theme::secondary } else { theme::ghost }).on_press(Message::ToggleSidebar),
            container(text(tr("Show or hide the sidebar")).size(12)).padding([4, 8]).style(theme::tooltip),
            tooltip::Position::Bottom,
        );
        // Beside the sidebar the brand spans its width. Without it, the name goes so the conversation keeps the room.
        let brand: Element<'_, Message> = if docked {
            container(row![self.logo(36.0), text("Scoobert").size(19).font(fonts::ui_semibold()), space::horizontal(), toggle].spacing(10).align_y(Alignment::Center))
                .width(SIDEBAR_WIDTH)
                .padding([0, 12])
                .into()
        } else {
            container(row![self.logo(36.0), toggle].spacing(10).align_y(Alignment::Center)).width(BRAND_COMPACT).padding([0, 12]).into()
        };
        let choices = self.model_choices();
        let selected = self.current_model().map(|m| ModelChoice {
            name: m.name.clone(),
            label: if m.provider.is_empty() { m.label.clone() } else { format!("{} ({})", m.label, m.provider) },
        });
        // A long model name is cut in a narrow window rather than crowding out the buttons after it.
        let model_pick = pick_list(choices, selected, Message::SetModel)
            .placeholder(tr("No model"))
            .width(if narrow { Length::Fixed(130.0) } else if compact { Length::Fixed(180.0) } else { Length::Shrink })
            .padding([5, 10])
            .text_size(13)
            .style(theme::select)
            .menu_style(theme::menu);
        let thinking = self.chat.as_ref().map(|c| c.thinking).unwrap_or(self.state.settings.thinking);
        let reasoning = self.current_model().is_none_or(|m| m.reasoning);
        let mut right = row![].spacing(8).align_y(Alignment::Center);
        if !compact {
            right = right.push(text(tr("Model")).size(13).style(theme::muted));
        }
        right = right.push(model_pick);
        if reasoning {
            if !compact {
                right = right.push(text(tr("Thinking")).size(13).style(theme::muted));
            }
            right = right.push(
                pick_list(Thinking::ALL, Some(thinking), Message::SetThinking)
                    .padding([5, 10])
                    .text_size(13)
                    .style(theme::select)
                    .menu_style(theme::menu),
            );
        }
        // The button shows a two-letter code so the top bar keeps room for the conversation title.
        let code = crate::i18n::current().code.split('-').next().unwrap_or_default().to_uppercase();
        let mut language = row![icon(Icon::Globe, 16.0)].spacing(6).align_y(Alignment::Center);
        if !narrow {
            language = language.push(text(code).size(13));
        }
        right = right.push(
            button(language)
                .padding([6, 10])
                .style(if self.language_menu { theme::secondary } else { theme::ghost })
                .on_press(Message::LanguageMenu(!self.language_menu)),
        );
        let notes_style = if self.notes.open { theme::secondary } else { theme::ghost };
        let mut notes_label = row![icon(Icon::Panel, 16.0)].spacing(6).align_y(Alignment::Center);
        if !compact {
            notes_label = notes_label.push(text(tr("Notes")).size(13));
        }
        right = right.push(button(notes_label).padding([6, 12]).style(notes_style).on_press(Message::Notes(notes::Msg::TogglePane)));
        let caption = |i: Icon, m: Message| {
            button(center(icons::tinted(i, CAPTION_GLYPH, caption_glyph))).width(CAPTION_BUTTON).height(CAPTION_HEIGHT).padding(0).style(theme::caption).on_press(m)
        };
        let close_glyph =
            if self.close_hover { icons::tinted(Icon::Close, CAPTION_GLYPH, |_| iced::Color::WHITE) } else { icons::tinted(Icon::Close, CAPTION_GLYPH, caption_glyph) };
        let close = mouse_area(button(center(close_glyph)).width(CAPTION_BUTTON).height(CAPTION_HEIGHT).padding(0).style(theme::caption_close).on_press(Message::Quit))
            .on_enter(Message::CloseHover(true))
            .on_exit(Message::CloseHover(false));
        let controls = row![
            caption(Icon::Minimize, Message::Minimize),
            caption(if self.maximized { Icon::Restore } else { Icon::Maximize }, Message::ToggleMaximize),
            close,
        ];
        // The window buttons sit in a layer of their own, so a top bar too full for the window can never push them out.
        let bar = row![brand, self.location(), right.padding([0, if narrow { 8 } else { 16 }]), space().width(CAPTION_WIDTH)].height(56).align_y(Alignment::Center);
        stack![bar, container(controls).width(Fill).height(Fill).align_x(Alignment::End).align_y(Alignment::Start)].height(56).into()
    }

    /// Where the open conversation lives: the project as a label that opens its folder, then the conversation title.
    fn location(&self) -> Element<'_, Message> {
        let narrow = self.window_size.is_some_and(|s| s.width < LOCATION_WIDTH);
        let name = self.current_project().map(|p| project_name(&p)).unwrap_or_else(|| tr("Chats").into());
        // A narrow window keeps the project label and leaves out the title, so a long name is shortened to fit.
        let shown = if narrow { clip(&name, 14) } else { name.clone() };
        let place: Element<'_, Message> = match self.current_project() {
            Some(p) => tooltip(
                button(row![icon(Icon::Folder, 14.0), text(shown).size(13).wrapping(text::Wrapping::None)].spacing(6).align_y(Alignment::Center))
                    .padding([4, 10])
                    .style(theme::place)
                    .on_press(Message::RevealProject(p)),
                container(text(tr("Show the project folder")).size(12)).padding([4, 8]).style(theme::tooltip),
                tooltip::Position::Bottom,
            )
            .into(),
            None => button(text(name.clone()).size(13)).padding([4, 10]).style(theme::place).into(),
        };
        let mut crumbs = row![place].spacing(8).align_y(Alignment::Center);
        if let Some(c) = self.chat.as_ref().filter(|c| !c.is_empty() && !narrow) {
            crumbs = crumbs.push(icons::tinted(Icon::ChevronRight, 14.0, |t| t.muted));
            // A long project name leaves less room, and the title is shortened with an ellipsis before it reaches the model menu.
            let room = 46usize.saturating_sub(name.chars().count()).max(12);
            crumbs = crumbs.push(text(clip(&c.title, room)).size(13).font(fonts::for_text(&c.title)).style(theme::muted).wrapping(text::Wrapping::None));
        }
        // Lines the label up with the messages, which sit 24 in from the chat column and center once it is wider than they are.
        let (docked, notes_width) = self.panes();
        let (side, brand) = if docked { (SIDEBAR_WIDTH + 1.0, SIDEBAR_WIDTH) } else { (0.0, BRAND_COMPACT) };
        let notes = if self.notes.open { 1.0 + notes_width } else { 0.0 };
        let column = self.window_size.map_or(0.0, |s| s.width) - side - notes;
        let left = (side + ((column - chat::MAX_WIDTH) / 2.0).max(0.0) + 24.0 - brand).max(8.0);
        let right = if narrow { 8.0 } else { 32.0 };
        container(crumbs).width(Fill).clip(true).padding(iced::Padding { left, right, ..Default::default() }).into()
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let new_button = button(row![icon(Icon::Plus, 16.0), text(tr("New chat")).size(14)].spacing(8).align_y(Alignment::Center))
            .width(Fill)
            .padding([8, 14])
            .style(theme::secondary)
            .on_press(Message::NewChat);
        let header = row![
            text(tr("Projects")).size(13).style(theme::muted),
            space::horizontal(),
            tooltip(
                button(icon(Icon::FolderOpen, 16.0)).padding(4).style(theme::ghost).on_press(Message::AddProject),
                container(text(tr("Add a project folder")).size(12)).padding([4, 8]).style(theme::tooltip),
                tooltip::Position::Bottom,
            ),
        ]
        .align_y(Alignment::Center)
        .padding([0, 4]);

        let search = text_input(tr("Search projects"), &self.search)
            .id(PROJECT_SEARCH_ID)
            .on_input(Message::SearchProjects)
            .size(13)
            .padding([6, 10])
            .style(theme::input);
        let current = self.current_project();
        let mut list = Column::new().spacing(2).push(header);
        if !self.search.trim().is_empty() {
            list = list.push(self.search_results());
            return self.sidebar_frame(new_button.into(), search.into(), list);
        }
        // Chats is listed first, as a project without a folder.
        let general = current.is_none();
        let chats = button(text(tr("Chats")).size(14).font(if general { fonts::ui_bold() } else { fonts::ui() }))
            .width(Fill)
            .padding([6, 4])
            .style(theme::row_button)
            .on_press(Message::SelectNoProject);
        let chats_actions = row![
            space::horizontal(),
            row_actions(row![tooltip(
                button(icon(Icon::Plus, 14.0)).padding([4, 3]).style(theme::ghost).on_press(Message::NewConversationIn(None)),
                container(text(tr("New chat")).size(12)).padding([4, 8]).style(theme::tooltip),
                tooltip::Position::Bottom,
            )]),
        ]
        .height(Fill)
        .align_y(Alignment::Center);
        list = list.push(hover(chats, container(chats_actions).height(Fill).align_y(Alignment::Center)));
        if general {
            list = list.extend(self.sessions.iter().map(|s| self.session_row(s)));
            list = list.push(space().height(8));
        }
        for p in &self.state.projects {
            let is_current = current.as_ref() == Some(&p.path);
            let name = button(text(p.name()).size(14).font(if is_current { fonts::ui_bold() } else { fonts::ui() }))
                .width(Fill)
                .padding([6, 4])
                .style(theme::row_button)
                .on_press(Message::SelectProject(p.path.clone()));
            let actions = row![
                space::horizontal(),
                row_actions(row![
                    tooltip(
                        button(icon(Icon::Plus, 14.0)).padding([4, 3]).style(theme::ghost).on_press(Message::NewConversationIn(Some(p.path.clone()))),
                        container(text(tr("New conversation")).size(12)).padding([4, 8]).style(theme::tooltip),
                        tooltip::Position::Bottom,
                    ),
                    button(icon(Icon::Folder, 14.0)).padding([4, 3]).style(theme::ghost).on_press(Message::RevealProject(p.path.clone())),
                    button(icon(Icon::Close, 14.0)).padding([4, 3]).style(theme::ghost).on_press(Message::AskConfirm(Confirm::RemoveProject(p.path.clone()))),
                ]),
            ]
            .height(Fill)
            .align_y(Alignment::Center);
            list = list.push(hover(name, container(actions).height(Fill).align_y(Alignment::Center)));
            if is_current {
                for s in &self.sessions {
                    list = list.push(self.session_row(s));
                }
                list = list.push(space().height(8));
            }
        }
        if self.state.projects.is_empty() {
            list = list.push(container(text(tr("Add an existing folder with the folder button, or let Scoobert start one.")).size(12).style(theme::muted)).padding([4, 4]));
        }
        self.sidebar_frame(new_button.into(), search.into(), list)
    }

    fn sidebar_frame<'a>(&'a self, new_button: Element<'a, Message>, search: Element<'a, Message>, list: Column<'a, Message>) -> Element<'a, Message> {
        let (dot, status): (fn(&theme::Tokens) -> iced::Color, String) = self.status_line();
        // The status is cut to fit beside the gear, since a model name can be long.
        let status: Element<'_, Message> = container(text(clip(&status, 32)).size(13).style(theme::muted).wrapping(text::Wrapping::None)).width(Fill).clip(true).into();
        let status: Element<'_, Message> = match &self.preparing {
            Some(_) => tooltip(
                status,
                container(text(tr("Building a saved prompt so new conversations start quickly. This happens once per model.")).size(12).width(260)).padding([4, 8]).style(theme::tooltip),
                tooltip::Position::Top,
            )
            .into(),
            None => status,
        };
        let percent: Element<'_, Message> = match &self.preparing {
            Some((_, pct)) => text(format!("{pct}%")).size(13).style(theme::muted).into(),
            None => space().into(),
        };
        let footer = row![
            container(space()).width(8).height(8).style(theme::dot(dot)),
            status,
            percent,
            button(icon(Icon::Gear, 18.0)).padding(4).style(theme::ghost).on_press(Message::OpenSettings(settings::Section::General)),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .padding([10, 16]);

        container(
            column![
                container(column![new_button, search].spacing(10)).padding([14, 14]),
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

    /// Lists every conversation in Chats and the projects, newest first, for the sidebar search.
    fn load_search_index(&self) -> Task<Message> {
        let mut places: Vec<Option<PathBuf>> = vec![None];
        places.extend(self.state.projects.iter().map(|p| Some(p.path.clone())));
        let root = crate::paths::projects_root();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    let mut all: Vec<(Option<PathBuf>, Summary)> = places
                        .into_iter()
                        .flat_map(|place| {
                            let dir = place.clone().unwrap_or_else(|| root.clone());
                            Conversation::list(&dir).into_iter().map(move |s| (place.clone(), s))
                        })
                        .collect();
                    all.sort_by(|a, b| b.1.modified.cmp(&a.1.modified));
                    all
                })
                .await
                .unwrap_or_default()
            },
            Message::SearchIndex,
        )
    }

    /// Projects whose names match the search, then conversations whose titles match it.
    fn search_results(&self) -> Element<'_, Message> {
        let query = self.search.trim().to_lowercase();
        let mut col = Column::new().spacing(2);
        let mut found = false;
        let place_row = |label: String, msg: Message| button(text(label).size(14)).width(Fill).padding([6, 4]).style(theme::row_button).on_press(msg);
        if "chats".contains(&query) || tr("Chats").to_lowercase().contains(&query) {
            col = col.push(place_row(tr("Chats").into(), Message::SelectNoProject));
            found = true;
        }
        for p in self.state.projects.iter().filter(|p| p.name().to_lowercase().contains(&query)) {
            col = col.push(place_row(p.name(), Message::SelectProject(p.path.clone())));
            found = true;
        }
        let Some(index) = &self.search_index else {
            return col.push(container(text(tr("Searching conversations")).size(12).style(theme::muted)).padding([6, 4])).into();
        };
        let hits: Vec<&(Option<PathBuf>, Summary)> = index.iter().filter(|(_, s)| s.title.to_lowercase().contains(&query)).take(50).collect();
        if !hits.is_empty() {
            col = col.push(container(text(tr("Conversations")).size(13).style(theme::muted)).padding(iced::Padding { top: 10.0, right: 4.0, bottom: 2.0, left: 4.0 }));
            found = true;
        }
        for (place, s) in hits {
            let project = place.as_deref().map(project_name).unwrap_or_else(|| tr("Chats").into());
            let label = column![
                container(text(clip(&s.title, 34)).size(13).font(fonts::for_text(&s.title)).wrapping(text::Wrapping::None)).width(Fill).clip(true),
                row![text(project).size(12).style(theme::muted), space::horizontal(), text(ago(s.modified)).size(12).style(theme::muted)],
            ]
            .spacing(2);
            col = col.push(button(label).width(Fill).padding([6, 10]).style(theme::list_item(false)).on_press(Message::OpenIn(place.clone(), s.file.clone())));
        }
        if !found {
            col = col.push(container(text(tr("No projects or conversations match.")).size(12).style(theme::muted)).padding([6, 4]));
        }
        col.into()
    }

    fn session_row<'a>(&'a self, s: &'a Summary) -> Element<'a, Message> {
        let selected = self.chat.as_ref().is_some_and(|c| c.file == s.file);
        if let Some((file, value)) = &self.renaming
            && *file == s.file
        {
            return text_input(tr("Conversation name"), value)
                .id(RENAME_ID)
                .on_input(Message::RenameInput)
                .on_submit(Message::CommitRename)
                .size(13)
                .padding([6, 8])
                .style(theme::input)
                .into();
        }
        // The title gets the space the time leaves and is cut there, so the two never overlap.
        let when = ago(s.modified);
        let label = row![
            container(text(clip(&s.title, 30)).size(13).font(fonts::for_text(&s.title)).wrapping(text::Wrapping::None)).width(Fill).clip(true),
            text(when.clone()).size(12).style(theme::muted).wrapping(text::Wrapping::None),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let item = button(label).width(Fill).padding([6, 10]).style(theme::list_item(selected)).on_press(Message::OpenConversation(s.file.clone()));
        let base = container(item).height(32);
        // The buttons hide the whole time and fade over the end of the title, as on a project row, instead of
        // leaving half of the time showing through the fade.
        const FADE: f32 = 20.0;
        let solid = (estimated_width(&when, 12.0) + 10.0 + 8.0).max(46.0);
        let actions = container(
            row![
                button(icon(Icon::Pencil, 14.0)).padding([4, 3]).style(theme::ghost).on_press(Message::StartRename(s.file.clone(), s.title.clone())),
                button(icon(Icon::Trash, 14.0)).padding([4, 3]).style(theme::ghost).on_press(Message::AskConfirm(Confirm::DeleteConversation(s.file.clone()))),
            ]
            .align_y(Alignment::Center),
        )
        .width(FADE + solid)
        .height(Fill)
        .align_x(Alignment::End)
        .align_y(Alignment::Center)
        .padding(iced::Padding { left: FADE, right: 6.0, ..iced::Padding::ZERO })
        .style(theme::conversation_actions(selected, FADE / (FADE + solid)));
        hover(base, container(actions).width(Fill).height(Fill).align_x(Alignment::End)).into()
    }

    fn status_line(&self) -> (fn(&theme::Tokens) -> iced::Color, String) {
        if let Some(m) = self.current_model()
            && !m.provider.is_empty()
        {
            return (|t| t.ok, trf("Using {provider}", &[("provider", &m.provider)]));
        }
        if let Some((m, _)) = &self.preparing {
            return (|t| t.warn, trf("Preparing {model}", &[("model", m)]));
        }
        match &self.server {
            ServerStatus::Stopped => (|t| t.muted, tr("Model not loaded").into()),
            ServerStatus::Loading(m) | ServerStatus::GpuFailed(m) | ServerStatus::CudaFailed(m) => (|t| t.warn, trf("Loading {model}", &[("model", m)])),
            ServerStatus::Ready(m) => (|t| t.ok, trf("{model} loaded", &[("model", m)])),
            ServerStatus::Error(_) => (|t| t.danger, tr("The model stopped").into()),
        }
    }

    fn main_area(&self, theme: &Theme) -> Element<'_, Message> {
        let general = self.current_project().is_none();
        let intro = if general {
            tr("Ask anything. When a task needs its own files, Scoobert starts a project folder for it, or you can pick a project on the left.")
        } else {
            tr("Scoobert reads and edits files in this project, runs commands, and keeps notes as it works.")
        };
        let transcript: Element<'_, Message> = match &self.chat {
            Some(c) if !c.is_empty() => c.view(chat::markdown_settings(theme)),
            Some(_) => center(
                column![
                    self.logo(96.0),
                    text(tr("What should Scoobert work on?")).size(18).font(fonts::ui_semibold()),
                    text(intro).size(13).style(theme::muted),
                ]
                .spacing(10)
                .align_x(Alignment::Center),
            )
            .into(),
            None => center(text(tr("Opening...")).style(theme::muted)).into(),
        };
        let mut col = column![transcript];
        if let Some(e) = self.server_error() {
            col = col.push(container(container(text(e).size(13)).padding([8, 12]).width(Fill).style(theme::error_box)).padding([0, 24]).max_width(868));
        }
        col = col.push(container(self.composer_view()).padding(iced::Padding { top: 8.0, right: 24.0, bottom: 4.0, left: 24.0 }).center_x(Fill));
        col = col.push(container(self.footer()).padding(iced::Padding { top: 0.0, right: 24.0, bottom: 10.0, left: 24.0 }).center_x(Fill));
        let pane: Element<'_, Message> = col.width(Fill).height(Fill).into();
        if !self.approvals_menu {
            return pane;
        }
        let mut list = Column::new().spacing(2);
        for a in Approvals::ALL {
            let mark: Element<'_, Message> =
                if a == self.state.settings.approvals { icons::tinted(Icon::Check, 14.0, |t| t.accent_ink).into() } else { space().width(14).into() };
            list = list.push(
                button(row![text(a.to_string()).size(13).width(Fill), mark].spacing(8).align_y(Alignment::Center))
                    .width(Fill)
                    .padding([6, 10])
                    .style(theme::row_button)
                    .on_press(Message::SetApprovals(a)),
            );
        }
        // The menu opens above the footer's right end, laid out like the footer so the two line up at any width.
        let menu = container(container(container(list).width(260).padding(4).style(theme::popover)).max_width(820).width(Fill).align_x(Alignment::End))
            .center_x(Fill)
            .height(Fill)
            .align_y(Alignment::End)
            .padding(iced::Padding { top: 0.0, right: 24.0, bottom: 38.0, left: 24.0 });
        stack![pane, mouse_area(container(space()).width(Fill).height(Fill)).on_press(Message::ApprovalsMenu(false)), menu].into()
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
            .placeholder(tr("Ask Scoobert"))
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
                    row![text(trf("Image {number}", &[("number", &(i + 1))])).size(12), button(icon(Icon::Close, 12.0)).padding(2).style(theme::ghost).on_press(Message::RemoveImage(i))]
                        .spacing(4)
                        .align_y(Alignment::Center),
                )
                .padding([2, 8])
                .style(theme::chip),
            );
        }
        let vision = self.current_model().is_some_and(|m| m.vision);
        let action: Element<'_, Message> = if running {
            let stop = button(row![icons::tinted(Icon::Stop, 15.0, |_| iced::Color::WHITE), text(tr("Stop")).size(14).font(fonts::ui_semibold())].spacing(6).align_y(Alignment::Center))
                .padding([7, 18])
                .style(theme::stop)
                .on_press(Message::Stop);
            // A message typed while Scoobert works waits for the current step instead of stopping it.
            if self.composer.text().trim().is_empty() && self.images.is_empty() {
                stop.into()
            } else {
                let send = tooltip(
                    button(row![text(tr("Send")).size(13), icons::tinted(Icon::ArrowRight, 14.0, |t| t.accent_text)].spacing(6).align_y(Alignment::Center))
                        .padding([6, 14])
                        .style(theme::primary)
                        .on_press(Message::Send),
                    container(text(tr("Sends when the current step finishes")).size(12)).padding([4, 8]).style(theme::tooltip),
                    tooltip::Position::Top,
                );
                row![send, stop].spacing(8).align_y(Alignment::Center).into()
            }
        } else {
            let can_send = !self.composer.text().trim().is_empty() || !self.images.is_empty();
            let arrow = if can_send { icons::tinted(Icon::ArrowRight, 14.0, |t| t.accent_text) } else { icons::tinted(Icon::ArrowRight, 14.0, |t| t.muted) };
            button(row![text(tr("Send")).size(13), arrow].spacing(6).align_y(Alignment::Center))
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
                    container(text(tr("Attach images")).size(12)).padding([4, 8]).style(theme::tooltip),
                    tooltip::Position::Top,
                ),
            );
        }
        tools = tools.push(text(tr("Enter to send, Shift+Enter for a new line")).size(12).style(theme::muted));
        tools = tools.push(space::horizontal());
        // Update notes covers the whole conversation, so it sits beside Send rather than on one reply.
        if !running && self.chat.as_ref().is_some_and(|c| !crate::agent::is_general(&c.cwd) && c.has_reply()) {
            tools = tools.push(tooltip(
                button(row![icon(Icon::File, 14.0), text(tr("Update notes")).size(13)].spacing(6).align_y(Alignment::Center))
                    .padding([6, 12])
                    .style(theme::secondary)
                    .on_press(Message::UpdateNotes),
                container(text(tr("Adds the features this conversation finished, and the ones still to do, to the project's notes.")).size(12).width(280))
                    .padding([4, 8])
                    .style(theme::tooltip),
                tooltip::Position::Top,
            ));
        }
        tools = tools.push(action);
        let mut col = Column::new().spacing(10);
        if self.web_offer {
            col = col.push(
                container(
                    column![
                        text(tr("This sounds like it needs the web, and web search is off.")).size(13).font(fonts::ui_semibold()),
                        text(tr("With it on, Scoobert searches DuckDuckGo and reads pages. Your searches leave your computer.")).size(12).style(theme::muted),
                        row![
                            button(text(tr("Turn on web search and send")).size(13)).padding([5, 12]).style(theme::primary).on_press(Message::WebOffer(true)),
                            button(text(tr("Send without it")).size(13)).padding([5, 12]).style(theme::secondary).on_press(Message::WebOffer(false)),
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
        if !running
            && let Some(chat) = &self.chat
            && self.disk_offer.as_deref() == Some(chat.id.as_str())
        {
            col = col.push(
                container(
                    column![
                        text(tr("The model does not fit in free memory.")).size(13).font(fonts::ui_semibold()),
                        text(tr("Scoobert can run it with the part that does not fit read from disk as it goes. Replies come much more slowly.")).size(12).style(theme::muted),
                        row![
                            button(text(tr("Run it from disk and continue")).size(13)).padding([5, 12]).style(theme::primary).on_press(Message::DiskOffer(true)),
                            button(text(tr("Not now")).size(13)).padding([5, 12]).style(theme::secondary).on_press(Message::DiskOffer(false)),
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
        // Shown before any typing, because a task here could not run beside the one that holds the local model.
        if !running
            && let (Some(host), Some(chat)) = (&self.host, &self.chat)
            && let Some(other) = host.local_task_elsewhere(&chat.id).filter(|o| o.running)
        {
            let body = if other.same_model {
                tr("Both conversations use the same model, so a message you send here waits until that task finishes.")
            } else {
                tr("That task uses another model. Stop it before you send a message here.")
            };
            let place = (!crate::agent::is_general(&other.cwd)).then(|| other.cwd.clone());
            col = col.push(
                container(
                    row![
                        column![
                            text(trf("Scoobert is working in “{title}”.", &[("title", &clip(&other.title, 60))])).size(13).font(fonts::ui_semibold()),
                            text(body).size(12).style(theme::muted),
                        ]
                        .spacing(4)
                        .width(Fill),
                        button(text(tr("Go to it")).size(13)).padding([5, 12]).style(theme::secondary).on_press(Message::OpenIn(place, other.file.clone())),
                    ]
                    .spacing(12)
                    .align_y(Alignment::Center),
                )
                .padding(10)
                .width(Fill)
                .style(theme::banner),
            );
        }
        if !self.images.is_empty() {
            col = col.push(attachments);
        }
        col = col.push(input);
        if !running && self.chat.as_ref().is_some_and(plans_first) {
            let mut choices = row![text(tr("Before it builds:")).size(12).style(theme::muted)].spacing(4).align_y(Alignment::Center);
            for plan in PlanFirst::ALL {
                let style = if plan == self.state.settings.plan_first { theme::secondary } else { theme::ghost };
                choices = choices.push(button(text(plan.to_string()).size(12)).padding([3, 8]).style(style).on_press(Message::SetPlanFirst(plan)));
            }
            col = col.push(tooltip(
                choices,
                container(text(tr("With a plan, Scoobert first writes the features it will build into the project's notes, so later steps and conversations can follow it.")).size(12).width(300))
                    .padding([4, 8])
                    .style(theme::tooltip),
                tooltip::Position::Top,
            ));
        }
        col = col.push(tools);
        container(col).padding([12, 16]).max_width(820).width(Fill).style(theme::composer(false)).into()
    }

    fn footer(&self) -> Element<'_, Message> {
        let context = self.current_model().map(|m| m.context).unwrap_or(0) as u64;
        let used = self.chat.as_ref().map(|c| c.context).unwrap_or(0);
        let mut r = row![].spacing(10).align_y(Alignment::Center);
        if context > 0 {
            r = r.push(progress_bar(0.0..=context as f32, used.min(context) as f32).length(100).girth(4).style(theme::meter));
            r = r.push(text(trf("{used} of {total} context", &[("used", &short_count(used)), ("total", &short_count(context))])).size(12).style(theme::muted));
        }
        r = r.push(space::horizontal());
        // A button sized to the current choice, since a pick list keeps the width of its longest option.
        r = r.push(
            button(
                row![text(self.state.settings.approvals.to_string()).size(12), icons::tinted(Icon::ChevronDown, 12.0, |t| t.muted)]
                    .spacing(6)
                    .align_y(Alignment::Center),
            )
            .padding([3, 8])
            .style(theme::quiet_button)
            .on_press(Message::ApprovalsMenu(!self.approvals_menu)),
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

/// Whether the next message can plan first: the first message of a conversation in a project.
fn plans_first(chat: &chat::Chat) -> bool {
    chat.is_empty() && !crate::agent::is_general(&chat.cwd)
}

/// A rough width for short UI text, wide enough for the UI font, with CJK characters counted at full width.
fn estimated_width(s: &str, size: f32) -> f32 {
    s.chars().map(|c| if c.is_ascii() { 0.58 } else { 1.0 }).sum::<f32>() * size
}

/// The buttons a hovered sidebar row shows, on a fade that hides the end of a long name under them.
fn row_actions<'a>(buttons: iced::widget::Row<'a, Message>) -> Element<'a, Message> {
    container(buttons.align_y(Alignment::Center))
        .height(Fill)
        .align_y(Alignment::Center)
        .padding(iced::Padding { left: 20.0, right: 2.0, ..iced::Padding::ZERO })
        .style(theme::row_actions)
        .into()
}

fn modal<'a>(content: Element<'a, Message>, width: f32) -> Element<'a, Message> {
    opaque(
        mouse_area(center(opaque(container(content).max_width(width).width(Fill).style(theme::modal))).padding(24).style(theme::backdrop))
            .on_press(Message::CloseModal),
    )
}

/// Thin strips along the window's edges and corners that resize it, since the window has no system frame.
fn resize_edges<'a>() -> Element<'a, Message> {
    use iced::mouse::Interaction;
    use window::Direction;
    const EDGE: f32 = 5.0;
    let edge = |width: Length, height: Length, direction: Direction, cursor: Interaction| -> Element<'a, Message> {
        mouse_area(space().width(width).height(height)).on_press(Message::ResizeFrom(direction)).interaction(cursor).into()
    };
    let (fixed, fill) = (Length::Fixed(EDGE), Length::Fill);
    column![
        row![
            edge(fixed, fixed, Direction::NorthWest, Interaction::ResizingDiagonallyDown),
            edge(fill, fixed, Direction::North, Interaction::ResizingVertically),
            edge(fixed, fixed, Direction::NorthEast, Interaction::ResizingDiagonallyUp),
        ],
        row![
            edge(fixed, fill, Direction::West, Interaction::ResizingHorizontally),
            space().width(Fill).height(Fill),
            edge(fixed, fill, Direction::East, Interaction::ResizingHorizontally),
        ]
        .height(Fill),
        row![
            edge(fixed, fixed, Direction::SouthWest, Interaction::ResizingDiagonallyUp),
            edge(fill, fixed, Direction::South, Interaction::ResizingVertically),
            edge(fixed, fixed, Direction::SouthEast, Interaction::ResizingDiagonallyDown),
        ],
    ]
    .into()
}

fn confirm_dialog<'a>(c: &Confirm, rewind_files: bool) -> Element<'a, Message> {
    if let Confirm::Rewind { files, .. } = c {
        let mut col = column![
            text(tr("Rewind to this message?")).size(16).font(fonts::ui_semibold()),
            text(tr("Scoobert stops what it is doing and forgets everything after this message. The message goes back into the box so you can change it and send it again.")).size(13).style(theme::muted),
        ]
        .spacing(12)
        .padding(20);
        if *files > 0 {
            let label = if *files == 1 {
                tr("Also undo the change to 1 file made after this message").to_string()
            } else {
                trf("Also undo the changes to {files} files made after this message", &[("files", files)])
            };
            col = col.push(iced::widget::checkbox(rewind_files).label(label).on_toggle(Message::RewindFiles).text_size(13).style(theme::check));
            col = col.push(text(tr("Commands Scoobert ran cannot be undone.")).size(12).style(theme::muted));
        }
        col = col.push(
            row![
                space::horizontal(),
                button(text(tr("Cancel")).size(13)).padding([6, 14]).style(theme::secondary).on_press(Message::CloseModal),
                button(text(tr("Rewind")).size(13)).padding([6, 14]).style(theme::primary).on_press(Message::Confirmed(c.clone())),
            ]
            .spacing(8),
        );
        return col.into();
    }
    let (title, body, action) = match c {
        Confirm::DeleteConversation(_) => (
            tr("Delete this conversation?"),
            tr("The conversation file moves to the trash, so you can restore it from there."),
            tr("Delete"),
        ),
        Confirm::RemoveProject(_) => (
            tr("Remove this project from the list?"),
            tr("Scoobert forgets the folder but keeps its files, notes, and conversations."),
            tr("Remove"),
        ),
        Confirm::InstallUpdate => (
            tr("Update while Scoobert is working?"),
            tr("Scoobert downloads the update, then stops the task and restarts to install it. Select Continue afterward to pick the task up again."),
            tr("Update"),
        ),
        Confirm::Rewind { .. } => unreachable!("handled above"),
    };
    column![
        text(title).size(16).font(fonts::ui_semibold()),
        text(body).size(13).style(theme::muted),
        row![
            space::horizontal(),
            button(text(tr("Cancel")).size(13)).padding([6, 14]).style(theme::secondary).on_press(Message::CloseModal),
            button(text(action).size(13)).padding([6, 14]).style(theme::danger).on_press(Message::Confirmed(c.clone())),
        ]
        .spacing(8),
    ]
    .spacing(12)
    .padding(20)
    .into()
}
