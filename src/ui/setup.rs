//! The first-run screen and the model downloads it shares with Settings.

use std::sync::Arc;

use iced::futures::SinkExt;
use iced::widget::text::Wrapping;
use iced::widget::{Column, button, checkbox, column, container, pick_list, progress_bar, row, scrollable, space, text};
use iced::{Alignment, Element, Fill, Task};
use tokio_util::sync::CancellationToken;

use super::Message;
use super::fonts;
use super::icons::{self, Icon};
use super::settings::{Effect, Section};
use super::theme;
use crate::agent::Host;
use crate::i18n::{tr, trf};
use crate::llama::catalog::{CATALOG, memory_needed};
use crate::llama::download::{ChooseSize, Progress};
use crate::store::State;
use crate::util::gb;

/// Memory left for the system and other apps when choosing which models fit.
const HEADROOM: u64 = 6_000_000_000;

/// A model to download: one from the catalog, or any repository given as "owner/repository:quantization".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Job {
    Catalog(usize),
    Custom(String),
}

pub struct Download {
    pub job: Job,
    pub progress: Option<Progress>,
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone)]
pub enum DownloadEvent {
    Progress(Progress),
    /// The downloaded model's name.
    Done(Result<String, String>),
    /// The repository has several sizes and the download did not name one it has.
    Choose(ChooseSize),
}

pub struct Setup {
    pub selected: Vec<bool>,
    pub queue: Vec<Job>,
    pub download: Option<Download>,
    pub error: Option<String>,
    /// The sizes of the last custom download, for the user to pick one.
    pub choices: Option<ChooseSize>,
    /// Free space on the drive of the models folder, measured when the folder changes and after each download.
    free_space: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Toggle(usize, bool),
    Start,
    StartOne(usize),
    StartCustom(String),
    Event(DownloadEvent),
    Cancel,
    UseHosted,
    Skip,
    Language(&'static crate::i18n::Language),
    BrowseFolder,
    FolderPicked(Option<std::path::PathBuf>),
}

/// The 9B, plus the best larger model that fits this computer's memory.
fn default_selection() -> Vec<bool> {
    let usable = crate::sys::total_memory().saturating_sub(HEADROOM);
    let best = (1..CATALOG.len()).rev().find(|&i| CATALOG[i].pick && memory_needed(&CATALOG[i]) <= usable);
    (0..CATALOG.len()).map(|i| i == 0 || Some(i) == best).collect()
}

pub fn fits(i: usize) -> bool {
    memory_needed(&CATALOG[i]) <= crate::sys::total_memory().saturating_sub(HEADROOM)
}

impl Setup {
    pub fn new(models_dir: &std::path::Path) -> Self {
        Setup { selected: default_selection(), queue: Vec::new(), download: None, error: None, choices: None, free_space: crate::sys::free_space(models_dir) }
    }

    pub fn update(&mut self, msg: Msg, state: &mut State, host: Option<&Arc<Host>>) -> (Task<Message>, Effect) {
        match msg {
            Msg::Toggle(i, on) => {
                if let Some(s) = self.selected.get_mut(i) {
                    *s = on;
                }
            }
            Msg::Start => {
                self.queue = (0..CATALOG.len()).filter(|&i| self.selected[i]).map(Job::Catalog).collect();
                return (self.next(host), Effect::None);
            }
            Msg::StartOne(i) => return (self.enqueue(Job::Catalog(i), host), Effect::None),
            Msg::StartCustom(spec) => return (self.enqueue(Job::Custom(spec.trim().to_string()), host), Effect::None),
            Msg::Event(DownloadEvent::Progress(p)) => {
                if let Some(d) = &mut self.download {
                    d.progress = Some(p);
                }
            }
            Msg::Event(DownloadEvent::Choose(choices)) => {
                self.download = None;
                self.choices = Some(choices);
                return (self.next(host), Effect::None);
            }
            Msg::Event(DownloadEvent::Done(result)) => {
                let finished = self.download.take();
                self.free_space = crate::sys::free_space(&state.settings.models_dir());
                match result {
                    Ok(name) => {
                        // The first model that arrives becomes the default, and a larger catalog model replaces the 9B.
                        let current_missing = host.is_none_or(|h| !h.llama.models().iter().any(|m| m.name == state.settings.model));
                        let larger = finished.is_some_and(|d| matches!(d.job, Job::Catalog(i) if i > 0 && CATALOG[i].pick));
                        if current_missing || larger {
                            state.settings.model = name;
                        }
                        state.setup_done = true;
                        let next = self.next(host);
                        return (next, Effect::ModelsChanged);
                    }
                    Err(e) => {
                        self.queue.clear();
                        if e != "Stopped." {
                            self.error = Some(e.clone());
                            return (Task::none(), Effect::Toast(e));
                        }
                    }
                }
            }
            Msg::Cancel => {
                self.queue.clear();
                if let Some(d) = &self.download {
                    d.cancel.cancel();
                }
            }
            Msg::UseHosted => {
                state.setup_done = true;
                return (Task::done(Message::OpenSettings(Section::Hosted)), Effect::Saved);
            }
            Msg::Skip => {
                state.setup_done = true;
                return (Task::none(), Effect::Saved);
            }
            Msg::Language(language) => {
                crate::i18n::set(language.code);
                state.settings.language = language.code.to_string();
                return (Task::none(), Effect::Saved);
            }
            Msg::BrowseFolder => {
                return (
                    Task::perform(
                        async { rfd::AsyncFileDialog::new().set_title(tr("Choose the models folder")).pick_folder().await.map(|h| h.path().to_path_buf()) },
                        |p| Message::Setup(Msg::FolderPicked(p)),
                    ),
                    Effect::None,
                );
            }
            Msg::FolderPicked(Some(dir)) => {
                state.settings.models_dir = dir.to_string_lossy().into_owned();
                self.free_space = crate::sys::free_space(&dir);
                // The folder may already hold models, which the model list then shows.
                return (Task::none(), Effect::ModelsChanged);
            }
            Msg::FolderPicked(None) => {}
        }
        (Task::none(), Effect::None)
    }

    fn enqueue(&mut self, job: Job, host: Option<&Arc<Host>>) -> Task<Message> {
        let active = self.download.as_ref().is_some_and(|d| d.job == job);
        if !active && !self.queue.contains(&job) {
            self.queue.push(job);
        }
        self.next(host)
    }

    fn next(&mut self, host: Option<&Arc<Host>>) -> Task<Message> {
        let Some(host) = host.cloned() else { return Task::none() };
        if self.download.is_some() || self.queue.is_empty() {
            return Task::none();
        }
        let job = self.queue.remove(0);
        let cancel = CancellationToken::new();
        self.download = Some(Download { job: job.clone(), progress: None, cancel: cancel.clone() });
        self.error = None;
        self.choices = None;
        let (spec, projector, revision) = match &job {
            Job::Catalog(i) => (CATALOG[*i].spec.to_string(), CATALOG[*i].projector, CATALOG[*i].revision),
            Job::Custom(spec) => (spec.clone(), true, "main"),
        };
        let stream = iced::stream::channel(64, async move |mut out: iced::futures::channel::mpsc::Sender<DownloadEvent>| {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Progress>();
            let dir = host.llama.models_dir();
            let http = host.http().clone();
            let job = tokio::spawn(async move {
                crate::llama::download::download(&http, &dir, &spec, projector, revision, &cancel, |p| {
                    let _ = tx.send(p);
                })
                .await
            });
            let mut job = std::pin::pin!(job);
            let result = loop {
                tokio::select! {
                    Some(p) = rx.recv() => { let _ = out.send(DownloadEvent::Progress(p)).await; }
                    r = &mut job => break r,
                }
            };
            let event = match result {
                Ok(Ok(folder)) => DownloadEvent::Done(Ok(folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())),
                Ok(Err(e)) => match e.downcast_ref::<ChooseSize>() {
                    Some(choices) => DownloadEvent::Choose(choices.clone()),
                    None => DownloadEvent::Done(Err(format!("{e:#}"))),
                },
                Err(e) => DownloadEvent::Done(Err(e.to_string())),
            };
            let _ = out.send(event).await;
        });
        Task::run(stream, |e| Message::Setup(Msg::Event(e)))
    }

    pub fn view(&self, state: &State) -> Element<'_, Msg> {
        let total_ram = crate::sys::total_memory();
        let mut cards = Column::new().spacing(10);
        for (i, m) in CATALOG.iter().enumerate() {
            let fits = fits(i);
            let need = memory_needed(m);
            let title = row![
                text(tr(m.label)).size(15).font(fonts::ui_semibold()),
                container(text(tr(m.tier)).size(11)).padding([1, 6]).style(theme::chip),
                space::horizontal(),
                text(trf("{size} download", &[("size", &gb(m.download_bytes))])).size(12).style(theme::muted),
            ]
            .spacing(8)
            .align_y(Alignment::Center);
            let mut info = column![title, text(tr(m.summary)).size(13).style(theme::muted)].spacing(6);
            info = info.push(if fits {
                text(trf("Needs about {memory} of free memory.", &[("memory", &gb(need))])).size(12).style(theme::muted)
            } else {
                text(trf(
                    "Needs about {memory} of free memory, more than this computer can spare. You can still run it by turning on loading models from disk in Settings, which uses storage as extra memory and runs much slower.",
                    &[("memory", &gb(need))],
                ))
                .size(12)
                .style(theme::warn_text)
            });
            if let Some(d) = self.download.as_ref().filter(|d| d.job == Job::Catalog(i)) {
                let (done, total, status) = d.progress.as_ref().map(|p| (p.done, p.total.max(1), p.status.clone())).unwrap_or((0, 1, tr("Starting...").into()));
                info = info.push(progress_bar(0.0..=total as f32, done as f32).girth(6).style(theme::meter));
                info = info.push(text(trf("{status}: {done} of {total}", &[("status", &status), ("done", &gb(done)), ("total", &gb(total))])).size(12).style(theme::muted));
            } else if self.queue.contains(&Job::Catalog(i)) {
                info = info.push(text(tr("Waiting...")).size(12).style(theme::muted));
            }
            let pick = checkbox(self.selected[i]).on_toggle_maybe(self.download.is_none().then_some(move |on| Msg::Toggle(i, on))).style(theme::check);
            let card = container(row![pick, info.width(Fill)].spacing(12)).padding(14).width(Fill);
            cards = cards.push(card.style(if self.selected[i] { theme::selected_card } else { theme::card }));
        }
        let busy = self.download.is_some();
        let any = self.selected.iter().any(|&s| s);
        let total: u64 = CATALOG.iter().zip(&self.selected).filter(|(_, s)| **s).map(|(m, _)| m.download_bytes).sum();
        let short = self.free_space.is_some_and(|free| total > free);
        let mut storage = column![
            row![
                text(tr("Where models are stored")).size(14),
                container(text(crate::paths::display(&state.settings.models_dir())).size(13).font(fonts::mono()).style(theme::muted).wrapping(Wrapping::None))
                    .width(Fill)
                    .clip(true),
                button(text(tr("Browse")).size(13)).padding([6, 12]).style(theme::secondary).on_press_maybe((!busy).then_some(Msg::BrowseFolder)),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        ]
        .spacing(4);
        if let Some(free) = self.free_space {
            storage = storage.push(if short {
                text(trf(
                    "The selected models need {size}, and this drive has {free} free. Choose another folder or fewer models.",
                    &[("size", &gb(total)), ("free", &gb(free))],
                ))
                .size(12)
                .style(theme::warn_text)
            } else {
                text(trf("{free} free on this drive.", &[("free", &gb(free))])).size(12).style(theme::muted)
            });
        }
        let mut actions = row![].spacing(10).align_y(Alignment::Center);
        actions = actions.push(if busy {
            button(text(tr("Stop the download")).size(14)).padding([8, 16]).style(theme::secondary).on_press(Msg::Cancel)
        } else {
            button(text(trf("Download {size}", &[("size", &gb(total))])).size(14))
                .padding([8, 16])
                .style(theme::primary)
                .on_press_maybe((any && !short).then_some(Msg::Start))
        });
        actions = actions.push(button(text(tr("Use a hosted model with an API key")).size(14)).padding([8, 16]).style(theme::secondary).on_press(Msg::UseHosted));
        actions = actions.push(space::horizontal());
        actions = actions.push(button(text(tr("Skip for now")).size(13)).style(theme::link).on_press(Msg::Skip));
        let languages: Vec<&'static crate::i18n::Language> = crate::i18n::LANGUAGES.iter().collect();
        let language = row![
            icons::tinted(Icon::Globe, 18.0, |t| t.muted),
            text(tr("Language")).size(14),
            pick_list(languages, Some(crate::i18n::current()), Msg::Language).padding([6, 12]).text_size(14).style(theme::select).menu_style(theme::menu),
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        let mut col = column![
            language,
            text(tr("Choose the models to download")).size(22).font(fonts::ui_semibold()),
            text(trf(
                "Scoobert runs Qwen models on this computer, so your code stays here. This computer has {memory} of memory. You can add or remove models later in Settings.",
                &[("memory", &gb(total_ram))]
            ))
            .size(14)
            .style(theme::muted),
            cards,
            storage,
            actions,
        ]
        .spacing(16)
        .max_width(760);
        if let Some(e) = &self.error {
            col = col.push(container(text(e.clone()).size(13)).padding([8, 12]).width(Fill).style(theme::error_box));
        }
        scrollable(container(col).padding(32).center_x(Fill)).height(Fill).style(theme::scrollbar).into()
    }
}
