//! The notes pane: the project's notes folder as a file list, a search, a link graph, and an editor.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use iced::widget::{Column, button, canvas, column, container, markdown, row, rule, scrollable, space, text, text_editor, text_input};
use iced::{Alignment, Element, Fill, Length, Task, Theme};

use super::Message;
use super::fonts;
use super::graph::GraphView;
use super::icons::{Icon, icon};
use super::theme;
use crate::i18n::tr;
use crate::notes::links::{WIKILINK, note_name, parse_link, resolve_link};
use crate::notes::{Backlink, Entry, Graph, SearchHit, Vault};

const SAVE_DELAY: Duration = Duration::from_millis(800);
const SEARCH_ID: &str = "note-search";
const NOTE_RENAME_ID: &str = "note-rename";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Files,
    Search,
    Graph,
}

struct Editor {
    path: String,
    content: text_editor::Content,
    saved: String,
    edited: Option<Instant>,
    preview: bool,
    md: markdown::Content,
    /// The note's front matter, shown as properties above the formatted note.
    props: Vec<(String, String)>,
    rename: Option<String>,
    backlinks: Vec<Backlink>,
}

pub struct Pane {
    pub open: bool,
    wide: bool,
    tab: Tab,
    root: Option<PathBuf>,
    files: Vec<Entry>,
    collapsed: HashSet<String>,
    query: String,
    results: Vec<SearchHit>,
    graph: Option<GraphView>,
    editor: Option<Editor>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Msg {
    TogglePane,
    ToggleWide,
    Tab(Tab),
    Loaded(PathBuf, Vec<Entry>),
    ToggleDir(String),
    Open(String),
    OpenLink(String),
    Opened(PathBuf, String, Result<(String, Vec<Backlink>), String>),
    Edit(text_editor::Action),
    Save,
    Saved(Result<(), String>),
    Back,
    TogglePreview,
    New,
    Daily,
    Created(Result<String, String>),
    Query(String),
    Results(Vec<SearchHit>),
    GraphLoaded(PathBuf, Graph),
    Delete,
    Deleted(Result<(), String>),
    RenameStart,
    RenameInput(String),
    RenameCommit,
    Renamed(Result<String, String>),
    Reveal,
    Preview(String),
}

fn wrap(task: Task<Msg>) -> Task<Message> {
    task.map(Message::Notes)
}

fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> impl std::future::Future<Output = T> {
    async move { tokio::task::spawn_blocking(f).await.expect("the notes task finishes") }
}

/// The note's front matter as properties, and its body formatted with links the reader can follow.
fn render(text: &str) -> (Vec<(String, String)>, markdown::Content) {
    let mut props = Vec::new();
    let mut body = text;
    if let Some(rest) = text.strip_prefix("---").and_then(|r| r.strip_prefix('\n').or_else(|| r.strip_prefix("\r\n")))
        && let Some(end) = rest.find("\n---")
    {
        for line in rest[..end].lines() {
            if let Some((key, value)) = line.split_once(':') {
                let value = value.trim().trim_matches('"');
                if !matches!(value, "" | "[]") {
                    props.push((key.trim().to_string(), value.to_string()));
                }
            }
        }
        body = rest[end + 4..].trim_start_matches(['-', '\r', '\n']);
    }
    (props, markdown::Content::parse(&link_wikilinks(body)))
}

/// Shows [[wikilinks]] as the text a reader sees, for previews that are not formatted.
fn plain_links(text: &str) -> String {
    WIKILINK
        .replace_all(text, |c: &regex::Captures| {
            let link = parse_link(&c[2]);
            link.alias.unwrap_or_else(|| note_name(&link.target))
        })
        .into_owned()
}

/// Turns [[wikilinks]] into Markdown links the preview can follow.
fn link_wikilinks(text: &str) -> String {
    WIKILINK
        .replace_all(text, |c: &regex::Captures| {
            let link = parse_link(&c[2]);
            let label = link.alias.clone().unwrap_or_else(|| link.target.clone());
            format!("[{label}](<note:{}>)", link.target)
        })
        .into_owned()
}

impl Pane {
    pub fn new(open: bool) -> Pane {
        Pane {
            open,
            wide: false,
            tab: Tab::Files,
            root: None,
            files: Vec::new(),
            collapsed: HashSet::from(["Daily".to_string()]),
            query: String::new(),
            results: Vec::new(),
            graph: None,
            editor: None,
            error: None,
        }
    }

    pub fn set_project(&mut self, root: Option<PathBuf>) {
        if self.root != root {
            self.root = root;
            self.files.clear();
            self.results.clear();
            self.graph = None;
            self.editor = None;
        }
    }

    /// The width the pane asks for. A narrow window gives it less.
    pub fn width(&self) -> f32 {
        if self.wide { 720.0 } else { 420.0 }
    }

    pub fn dirty(&self) -> bool {
        self.editor.as_ref().is_some_and(|e| e.edited.is_some())
    }

    fn vault(&self) -> Option<Vault> {
        self.root.clone().map(Vault::new)
    }

    pub fn reload(&self) -> Task<Message> {
        let Some(vault) = self.vault() else { return Task::none() };
        let mut tasks = vec![wrap(Task::perform(
            blocking(move || {
                let files = vault.files();
                (vault.root, files)
            }),
            |(root, files)| Msg::Loaded(root, files),
        ))];
        if self.tab == Tab::Graph {
            tasks.push(self.load_graph());
        }
        if let Some(e) = self.editor.as_ref().filter(|e| e.edited.is_none()) {
            tasks.push(self.read(e.path.clone()));
        }
        Task::batch(tasks)
    }

    fn load_graph(&self) -> Task<Message> {
        let Some(vault) = self.vault() else { return Task::none() };
        wrap(Task::perform(
            blocking(move || {
                let g = vault.graph();
                (vault.root, g)
            }),
            |(root, g)| Msg::GraphLoaded(root, g),
        ))
    }

    fn read(&self, path: String) -> Task<Message> {
        let Some(vault) = self.vault() else { return Task::none() };
        wrap(Task::perform(
            blocking(move || {
                let result = vault.read(&path).map(|t| (t, vault.backlinks(&path))).map_err(|e| format!("{e:#}"));
                (vault.root, path, result)
            }),
            |(root, path, r)| Msg::Opened(root, path, r),
        ))
    }

    /// Saves unsaved edits right away, before the app closes.
    pub fn flush(&mut self) -> Task<Message> {
        if let (Some(vault), Some(e)) = (self.vault(), self.editor.as_mut())
            && e.edited.is_some()
        {
            let text = e.content.text();
            e.edited = None;
            e.saved = text.clone();
            let _ = vault.write(&e.path, &text);
        }
        Task::none()
    }

    pub fn tick(&mut self) -> Task<Msg> {
        if self.editor.as_ref().and_then(|e| e.edited).is_some_and(|t| t.elapsed() >= SAVE_DELAY) {
            return Task::done(Msg::Save);
        }
        Task::none()
    }

    pub fn update(&mut self, msg: Msg, project: Option<&Path>) -> Task<Message> {
        match msg {
            Msg::TogglePane => {
                self.open = !self.open;
                if self.open && self.files.is_empty() {
                    return self.reload();
                }
            }
            Msg::ToggleWide => self.wide = !self.wide,
            Msg::Tab(tab) => {
                self.tab = tab;
                match tab {
                    Tab::Graph => return self.load_graph(),
                    Tab::Search => return iced::widget::operation::focus(SEARCH_ID),
                    Tab::Files => {}
                }
            }
            Msg::Loaded(root, files) => {
                if self.root.as_ref() == Some(&root) {
                    self.files = files;
                }
            }
            Msg::ToggleDir(dir) => {
                if !self.collapsed.remove(&dir) {
                    self.collapsed.insert(dir);
                }
            }
            Msg::Open(path) => {
                let save = self.flush();
                self.tab = Tab::Files;
                return Task::batch([save, self.read(path)]);
            }
            Msg::OpenLink(target) => {
                let paths: Vec<String> = self.files.iter().filter(|f| !f.is_dir).map(|f| f.path.clone()).collect();
                match resolve_link(&target, &paths) {
                    Some(p) => {
                        self.open = true;
                        return self.update(Msg::Open(p.clone()), project);
                    }
                    None => {
                        let Some(vault) = self.vault() else { return Task::none() };
                        let title = target.clone();
                        return wrap(Task::perform(
                            blocking(move || vault.create("", &title, &format!("# {title}\n\n")).map_err(|e| format!("{e:#}"))),
                            Msg::Created,
                        ));
                    }
                }
            }
            Msg::Opened(root, path, result) => {
                if self.root.as_ref() != Some(&root) {
                    return Task::none();
                }
                match result {
                    Ok((body, backlinks)) => {
                        let same = self.editor.as_ref().is_some_and(|e| e.path == path);
                        if same && let Some(e) = self.editor.as_mut() {
                            if e.saved != body {
                                e.content = text_editor::Content::with_text(&body);
                                (e.props, e.md) = render(&body);
                                e.saved = body;
                            }
                            e.backlinks = backlinks;
                        } else {
                            self.editor = Some(Editor {
                                path,
                                content: text_editor::Content::with_text(&body),
                                // A note with only a title opens for writing. Anything longer opens for reading.
                                preview: body.lines().filter(|l| !l.trim().is_empty()).count() > 1,
                                md: render(&body).1,
                                props: render(&body).0,
                                saved: body,
                                edited: None,
                                rename: None,
                                backlinks,
                            });
                        }
                        self.error = None;
                    }
                    Err(e) => self.error = Some(e),
                }
            }
            Msg::Edit(action) => {
                if let Some(e) = self.editor.as_mut() {
                    let edit = action.is_edit();
                    e.content.perform(action);
                    if edit {
                        e.edited = Some(Instant::now());
                    }
                }
            }
            Msg::Save => {
                let (Some(vault), Some(e)) = (self.vault(), self.editor.as_mut()) else { return Task::none() };
                if e.edited.is_none() {
                    return Task::none();
                }
                let text = e.content.text();
                e.edited = None;
                e.saved = text.clone();
                (e.props, e.md) = render(&text);
                let path = e.path.clone();
                return wrap(Task::perform(blocking(move || vault.write(&path, &text).map(|_| ()).map_err(|e| format!("{e:#}"))), Msg::Saved));
            }
            Msg::Saved(Ok(())) => return self.reload_list(),
            Msg::Saved(Err(e)) => self.error = Some(e),
            Msg::Back => {
                let save = self.flush();
                self.editor = None;
                return Task::batch([save, self.reload_list()]);
            }
            Msg::TogglePreview => {
                if let Some(e) = self.editor.as_mut() {
                    e.preview = !e.preview;
                    (e.props, e.md) = render(&e.content.text());
                }
            }
            Msg::New => {
                let Some(vault) = self.vault() else { return Task::none() };
                self.open = true;
                return wrap(Task::perform(
                    blocking(move || vault.create("", "Untitled", "# Untitled\n\n").map_err(|e| format!("{e:#}"))),
                    Msg::Created,
                ));
            }
            Msg::Daily => {
                let Some(vault) = self.vault() else { return Task::none() };
                self.open = true;
                return wrap(Task::perform(
                    blocking(move || vault.daily(chrono::Local::now().date_naive()).map_err(|e| format!("{e:#}"))),
                    Msg::Created,
                ));
            }
            Msg::Created(Ok(path)) => {
                let save = self.flush();
                self.tab = Tab::Files;
                return Task::batch([save, self.reload_list(), self.read(path)]);
            }
            Msg::Created(Err(e)) => self.error = Some(e),
            Msg::Query(q) => {
                self.query = q.clone();
                let Some(vault) = self.vault() else { return Task::none() };
                return wrap(Task::perform(blocking(move || vault.search(&q)), Msg::Results));
            }
            Msg::Results(r) => self.results = r,
            Msg::GraphLoaded(root, g) => {
                if self.root.as_ref() == Some(&root) {
                    self.graph = Some(GraphView::new(g));
                }
            }
            Msg::Delete => {
                let (Some(vault), Some(e)) = (self.vault(), self.editor.take()) else { return Task::none() };
                let path = e.path;
                return wrap(Task::perform(blocking(move || vault.delete(&path).map_err(|e| format!("{e:#}"))), Msg::Deleted));
            }
            Msg::Deleted(r) => {
                if let Err(e) = r {
                    self.error = Some(e);
                }
                return self.reload_list();
            }
            Msg::RenameStart => {
                if let Some(e) = self.editor.as_mut() {
                    e.rename = Some(note_name(&e.path));
                    return iced::widget::operation::focus(NOTE_RENAME_ID);
                }
            }
            Msg::RenameInput(v) => {
                if let Some(e) = self.editor.as_mut() {
                    e.rename = Some(v);
                }
            }
            Msg::RenameCommit => {
                let save = self.flush();
                let (Some(vault), Some(e)) = (self.vault(), self.editor.as_mut()) else { return Task::none() };
                let Some(name) = e.rename.take() else { return Task::none() };
                let name = crate::notes::sanitize_name(&name);
                if name == note_name(&e.path) {
                    return Task::none();
                }
                let from = e.path.clone();
                let dir = from.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
                let to = format!("{dir}{name}.md");
                return Task::batch([
                    save,
                    wrap(Task::perform(blocking(move || vault.rename(&from, &to).map_err(|e| format!("{e:#}"))), Msg::Renamed)),
                ]);
            }
            Msg::Renamed(Ok(path)) => {
                if let Some(e) = self.editor.as_mut() {
                    e.path = path;
                }
                return self.reload_list();
            }
            Msg::Renamed(Err(e)) => self.error = Some(e),
            Msg::Reveal => {
                if let Some(root) = &self.root {
                    let target = self.editor.as_ref().map(|e| root.join(&e.path)).unwrap_or_else(|| root.clone());
                    if !root.exists() {
                        let _ = std::fs::create_dir_all(root);
                    }
                    let _ = opener::reveal(target);
                }
            }
            Msg::Preview(url) => {
                if let Some(target) = url.strip_prefix("note:") {
                    return self.update(Msg::OpenLink(target.to_string()), project);
                }
                if url.starts_with("https://") || url.starts_with("http://") {
                    let _ = opener::open(url);
                }
            }
        }
        Task::none()
    }

    fn reload_list(&self) -> Task<Message> {
        let Some(vault) = self.vault() else { return Task::none() };
        let graph = if self.tab == Tab::Graph { self.load_graph() } else { Task::none() };
        Task::batch([
            wrap(Task::perform(
                blocking(move || {
                    let files = vault.files();
                    (vault.root, files)
                }),
                |(root, files)| Msg::Loaded(root, files),
            )),
            graph,
        ])
    }

    pub fn view(&self, theme: &Theme, width: f32) -> Element<'_, Message> {
        if self.root.is_none() {
            let body = column![
                text(tr("Notes")).size(14).font(fonts::ui_semibold()),
                text(tr("Notes belong to a project. Pick a project in the sidebar or ask Scoobert to start one, and its notes appear here.")).size(13).style(theme::muted),
            ]
            .spacing(10)
            .padding(20);
            return container(body).width(width).height(Fill).style(theme::app).into();
        }
        // A pane narrowed by a small window shows the tabs as icons, so the note buttons beside them still fit.
        let labels = width >= 400.0;
        let tab = |label: &'static str, i: Icon, t: Tab| {
            let active = self.tab == t;
            let underline = container(space()).height(2).width(Fill).style(if active { theme::accent_bar } else { |_: &Theme| container::Style::default() });
            let mut content = row![icon(i, 15.0)].spacing(6).align_y(Alignment::Center);
            if labels {
                content = content.push(text(label).size(13));
            }
            let button = button(content).padding([8, 10]).style(theme::tab(active)).on_press(Message::Notes(Msg::Tab(t)));
            let button: Element<'_, Message> = if labels {
                button.into()
            } else {
                iced::widget::tooltip(button, container(text(label).size(12)).padding([4, 8]).style(theme::tooltip), iced::widget::tooltip::Position::Bottom).into()
            };
            column![button, underline].width(Length::Shrink)
        };
        let action = |i: Icon, tip: &'static str, m: Msg| {
            iced::widget::tooltip(
                button(icon(i, 16.0)).padding(5).style(theme::ghost).on_press(Message::Notes(m)),
                container(text(tip).size(12)).padding([4, 8]).style(theme::tooltip),
                iced::widget::tooltip::Position::Bottom,
            )
        };
        let header = row![
            tab(tr("Files"), Icon::Folder, Tab::Files),
            tab(tr("Search"), Icon::Search, Tab::Search),
            tab(tr("Graph"), Icon::Graph, Tab::Graph),
            space::horizontal(),
            action(Icon::Plus, tr("New note"), Msg::New),
            action(Icon::Calendar, tr("Today's note"), Msg::Daily),
            action(if self.wide { Icon::Shrink } else { Icon::Expand }, if self.wide { tr("Narrower") } else { tr("Wider") }, Msg::ToggleWide),
            action(Icon::External, tr("Show the notes folder"), Msg::Reveal),
        ]
        .spacing(2)
        .align_y(Alignment::End)
        .padding([4, 10]);

        let body: Element<'_, Message> = match self.tab {
            Tab::Files => match &self.editor {
                Some(e) => self.editor_view(e, theme),
                None => self.tree(),
            },
            Tab::Search => self.search_view(),
            Tab::Graph => match &self.graph {
                Some(g) if !g.is_empty() => canvas(g).width(Fill).height(Fill).into(),
                Some(_) => container(text(tr("Notes that link to each other with [[wikilinks]] appear here.")).size(13).style(theme::muted)).padding(20).into(),
                None => container(text(tr("Loading...")).size(13).style(theme::muted)).padding(20).into(),
            },
        };
        let mut col = column![header, rule::horizontal(1).style(theme::divider)];
        if let Some(e) = &self.error {
            col = col.push(container(text(e.clone()).size(12)).padding([6, 10]).width(Fill).style(theme::error_box));
        }
        col = col.push(body);
        container(col.height(Fill)).width(width).height(Fill).style(theme::app).into()
    }

    fn tree(&self) -> Element<'_, Message> {
        let notes: Vec<&Entry> = self.files.iter().filter(|f| f.is_dir || f.path.to_lowercase().ends_with(".md")).collect();
        if notes.iter().all(|f| f.is_dir) {
            return container(
                column![
                    text(tr("No notes yet.")).size(14),
                    text(tr("Scoobert reads these notes before it works and adds to them when it finishes a task. You can write them too."))
                        .size(13)
                        .style(theme::muted),
                    button(text(tr("New note")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Message::Notes(Msg::New)),
                ]
                .spacing(10),
            )
            .padding(20)
            .into();
        }
        let mut col = Column::new().spacing(1).padding([8, 8]);
        for f in notes {
            let hidden = self.collapsed.iter().any(|c| f.path.starts_with(&format!("{c}/")));
            if hidden {
                continue;
            }
            let depth = f.path.matches('/').count() as f32;
            let name = f.path.rsplit('/').next().unwrap_or(&f.path);
            let item: Element<'_, Message> = if f.is_dir {
                let open = !self.collapsed.contains(&f.path);
                button(row![icon(if open { Icon::ChevronDown } else { Icon::ChevronRight }, 14.0), text(name.to_string()).size(14)].spacing(4).align_y(Alignment::Center))
                    .width(Fill)
                    .padding([5, 8])
                    .style(theme::row_button)
                    .on_press(Message::Notes(Msg::ToggleDir(f.path.clone())))
                    .into()
            } else {
                button(text(note_name(name)).size(14))
                    .width(Fill)
                    .padding([5, 8])
                    .style(theme::list_item(false))
                    .on_press(Message::Notes(Msg::Open(f.path.clone())))
                    .into()
            };
            col = col.push(row![space().width(depth * 16.0), item]);
        }
        scrollable(col).height(Fill).style(theme::scrollbar).into()
    }

    fn search_view(&self) -> Element<'_, Message> {
        let input = text_input(tr("Search notes"), &self.query)
            .id(SEARCH_ID)
            .on_input(|q| Message::Notes(Msg::Query(q)))
            .padding([6, 10])
            .size(13)
            .style(theme::input);
        let mut list = Column::new().spacing(6);
        for hit in &self.results {
            let mut c = column![text(note_name(&hit.path)).size(14).font(fonts::ui_semibold())].spacing(2);
            for (line, snippet) in &hit.lines {
                c = c.push(text(format!("{line}: {snippet}")).size(12).style(theme::muted));
            }
            list = list.push(button(c).width(Fill).padding([6, 8]).style(theme::row_button).on_press(Message::Notes(Msg::Open(hit.path.clone()))));
        }
        if self.results.is_empty() && !self.query.trim().is_empty() {
            list = list.push(text(tr("No matches.")).size(13).style(theme::muted));
        }
        column![input, scrollable(list).height(Fill).style(theme::scrollbar)].spacing(10).padding(12).into()
    }

    fn editor_view<'a>(&'a self, e: &'a Editor, theme: &Theme) -> Element<'a, Message> {
        let title: Element<'a, Message> = match &e.rename {
            Some(v) => text_input(tr("Note name"), v)
                .id(NOTE_RENAME_ID)
                .on_input(|v| Message::Notes(Msg::RenameInput(v)))
                .on_submit(Message::Notes(Msg::RenameCommit))
                .size(14)
                .padding([4, 8])
                .style(theme::input)
                .into(),
            None => button(text(note_name(&e.path)).size(15).font(fonts::ui_semibold()))
                .padding([4, 6])
                .style(theme::row_button)
                .on_press(Message::Notes(Msg::RenameStart))
                .into(),
        };
        let status = if e.edited.is_some() { tr("Saving...") } else { "" };
        let bar = row![
            button(icon(Icon::ArrowLeft, 16.0)).padding(5).style(theme::ghost).on_press(Message::Notes(Msg::Back)),
            container(title).width(Fill),
            text(status).size(12).style(theme::muted),
            button(row![icon(if e.preview { Icon::Pencil } else { Icon::Check }, 15.0), text(if e.preview { tr("Edit") } else { tr("Done") }).size(13)].spacing(6).align_y(Alignment::Center))
                .padding([4, 10])
                .style(theme::secondary)
                .on_press(Message::Notes(Msg::TogglePreview)),
            button(icon(Icon::Trash, 16.0)).padding(5).style(theme::ghost).on_press(Message::Notes(Msg::Delete)),
        ]
        .spacing(4)
        .align_y(Alignment::Center)
        .padding([6, 8]);
        let body: Element<'a, Message> = if e.preview {
            // Note headings stay modest, since the note's name is already in the bar above.
            let mut settings = super::chat::markdown_settings(theme);
            settings.text_size = 14.into();
            settings.h1_size = 21.into();
            settings.h2_size = 17.into();
            settings.h3_size = 15.into();
            settings.h4_size = 14.into();
            let mut page = Column::new().spacing(14);
            if !e.props.is_empty() {
                let mut props = Column::new().spacing(3);
                for (key, value) in &e.props {
                    props = props.push(row![text(key.clone()).size(12).style(theme::muted).width(96), text(value.clone()).size(12)].spacing(8));
                }
                page = page.push(container(props).padding([8, 10]).width(Fill).style(theme::chip));
            }
            page = page.push(markdown::view_with(e.md.items(), settings, &PreviewViewer));
            scrollable(container(page).padding(16)).height(Fill).style(theme::scrollbar).into()
        } else {
            container(
                text_editor(&e.content)
                    .on_action(|a| Message::Notes(Msg::Edit(a)))
                    .height(Fill)
                    .size(14)
                    .padding(12)
                    .font(fonts::mono())
                    .style(theme::bare_editor),
            )
            .height(Fill)
            .into()
        };
        let mut col = column![bar, rule::horizontal(1).style(theme::divider), body];
        if !e.backlinks.is_empty() {
            let mut links = Column::new().spacing(2).push(text(tr("Linked from")).size(12).style(theme::muted));
            for b in &e.backlinks {
                links = links.push(
                    button(column![text(note_name(&b.path)).size(13), text(plain_links(&b.context)).size(12).style(theme::muted)])
                        .width(Fill)
                        .padding([4, 6])
                        .style(theme::row_button)
                        .on_press(Message::Notes(Msg::Open(b.path.clone()))),
                );
            }
            col = col.push(rule::horizontal(1).style(theme::divider));
            col = col.push(container(scrollable(links).style(theme::scrollbar)).max_height(180).padding([8, 12]));
        }
        col.height(Fill).into()
    }
}

struct PreviewViewer;

impl<'a> markdown::Viewer<'a, Message> for PreviewViewer {
    fn on_link_click(url: markdown::Uri) -> Message {
        Message::Notes(Msg::Preview(url))
    }

    fn code_block(&self, settings: markdown::Settings, language: Option<&'a str>, code: &'a str, lines: &'a [markdown::Text]) -> Element<'a, Message> {
        <super::chat::Viewer as markdown::Viewer<'a, Message>>::code_block(&super::chat::Viewer, settings, language, code, lines)
    }
}
