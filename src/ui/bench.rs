//! The benchmark card in the model lab: runs the built-in coding tests on the model and lists every run so far.

use std::sync::Arc;

use iced::futures::SinkExt;
use iced::widget::{Column, button, column, container, pick_list, progress_bar, row, text};
use iced::{Alignment, Element, Fill, Task};
use tokio_util::sync::CancellationToken;

use super::Message;
use super::fonts;
use super::lab::Msg as LabMsg;
use super::settings::Msg as SettingsMsg;
use super::theme;
use crate::agent::Host;
use crate::i18n::{tr, trf};
use crate::llama::LocalModel;
use crate::llama::bench::{self, Event, Options, Run, Set, TaskResult};

#[derive(Debug, Clone)]
pub enum Msg {
    Loaded(Vec<Run>, bool),
    Set(SetChoice),
    Tries(TriesChoice),
    Thinking(bool),
    Start,
    Waiting,
    Event(Event),
    Done(Result<Run, String>),
    Stop,
}

fn wrap(m: Msg) -> Message {
    Message::Settings(SettingsMsg::Lab(LabMsg::Bench(m)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetChoice(pub Set);

impl std::fmt::Display for SetChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = bench::task_count(self.0);
        f.write_str(&match self.0 {
            Set::Normal => trf("Normal, {count} tasks", &[("count", &count)]),
            Set::Hard => trf("Hard, {count} tasks", &[("count", &count)]),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriesChoice(pub u32);

impl std::fmt::Display for TriesChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&if self.0 == 1 { tr("1 try").to_string() } else { trf("{count} tries", &[("count", &self.0)]) })
    }
}

struct Running {
    cancel: CancellationToken,
    status: String,
    done: usize,
    total: usize,
    results: Vec<TaskResult>,
}

pub struct Bench {
    set: Set,
    tries: u32,
    thinking: bool,
    history: Vec<Run>,
    /// Whether Python 3 is on this computer, once checked.
    python: Option<bool>,
    running: Option<Running>,
    error: Option<String>,
}

impl Bench {
    pub fn new() -> (Bench, Task<Message>) {
        let bench = Bench { set: Set::Normal, tries: 1, thinking: false, history: Vec::new(), python: None, running: None, error: None };
        let load = Task::perform(async { tokio::task::spawn_blocking(|| (bench::history(), bench::python().is_some())).await.unwrap_or_default() }, |(h, p)| wrap(Msg::Loaded(h, p)));
        (bench, load)
    }

    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    pub fn stop(&self) {
        if let Some(r) = &self.running {
            r.cancel.cancel();
        }
    }

    pub fn update(&mut self, msg: Msg, host: Option<&Arc<Host>>, model: &LocalModel) -> Task<Message> {
        match msg {
            Msg::Loaded(history, python) => {
                self.history = history;
                self.python = Some(python);
            }
            Msg::Set(c) => self.set = c.0,
            Msg::Tries(c) => self.tries = c.0,
            Msg::Thinking(on) => self.thinking = on,
            Msg::Start => return self.start(host, model),
            Msg::Waiting => {
                if let Some(r) = &mut self.running {
                    r.status = tr("Waiting for a conversation's task to finish...").into();
                }
            }
            Msg::Event(e) => {
                if let Some(r) = &mut self.running {
                    match e {
                        Event::Loading => r.status = tr("Loading the model...").into(),
                        Event::Task { index, count, name, attempt } => {
                            r.total = count;
                            r.status = if attempt == 1 {
                                trf("Task {number} of {count}: {name}", &[("number", &(index + 1)), ("count", &count), ("name", &name)])
                            } else {
                                trf("Task {number} of {count}: {name}, try {attempt}", &[("number", &(index + 1)), ("count", &count), ("name", &name), ("attempt", &attempt)])
                            };
                        }
                        Event::Finished(result) => {
                            r.done += 1;
                            r.results.push(result);
                        }
                    }
                }
            }
            Msg::Done(result) => {
                self.running = None;
                match result {
                    Ok(run) => self.history.insert(0, run),
                    Err(e) if e == tr("Stopped.") => {}
                    Err(e) => self.error = Some(e),
                }
            }
            Msg::Stop => self.stop(),
        }
        Task::none()
    }

    fn start(&mut self, host: Option<&Arc<Host>>, model: &LocalModel) -> Task<Message> {
        let Some(host) = host.cloned() else { return Task::none() };
        self.error = None;
        let cancel = CancellationToken::new();
        let total = bench::task_count(self.set);
        self.running = Some(Running { cancel: cancel.clone(), status: tr("Starting...").into(), done: 0, total, results: Vec::new() });
        let opts = Options { set: self.set, tries: self.tries, thinking: self.thinking };
        let model = model.clone();
        let stream = iced::stream::channel(64, async move |mut out: iced::futures::channel::mpsc::Sender<Msg>| {
            if host.model_in_use() {
                let _ = out.send(Msg::Waiting).await;
            }
            let hold = tokio::select! {
                h = host.hold_model() => h,
                _ = cancel.cancelled() => {
                    let _ = out.send(Msg::Done(Err(tr("Stopped.").into()))).await;
                    return;
                }
            };
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
            let work = tokio::spawn(bench::run(host.llama.clone(), model, opts, cancel, move |e| {
                let _ = tx.send(e);
            }));
            while let Some(e) = rx.recv().await {
                let _ = out.send(Msg::Event(e)).await;
            }
            let result = match work.await {
                Ok(Ok(run)) => Ok(run),
                Ok(Err(e)) if bench::stopped(&e) => Err(tr("Stopped.").to_string()),
                Ok(Err(e)) => Err(format!("{e:#}")),
                Err(e) => Err(e.to_string()),
            };
            drop(hold);
            let _ = out.send(Msg::Done(result)).await;
        });
        Task::run(stream, wrap)
    }

    pub fn view(&self) -> Element<'_, Msg> {
        let sets = [Set::Normal, Set::Hard].map(SetChoice);
        let tries: Vec<TriesChoice> = (1..=5).map(TriesChoice).collect();
        let controls = row![
            text(tr("Test")).size(13),
            pick_list(sets, Some(SetChoice(self.set)), Msg::Set).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu),
            text(tr("Tries per task")).size(13),
            pick_list(tries, Some(TriesChoice(self.tries)), Msg::Tries).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu),
            text(tr("Thinking")).size(13),
            super::switch::switch(self.thinking, Msg::Thinking),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .wrap();
        let mut col = column![
            text(tr("Benchmark")).size(15).font(fonts::ui_semibold()),
            text(tr("Gives the model a set of Python coding tasks with the settings it has now, such as the graphics card and Predict ahead, and runs hidden tests on each answer. With more than one try, a model whose code fails sees the test output and writes the code again, which shows how well it fixes its own mistakes.")).size(12).style(theme::muted),
            text(tr("The tests run the code each model writes, with Python, on this computer.")).size(12).style(theme::muted),
            controls,
        ]
        .spacing(10);
        if self.python == Some(false) {
            col = col.push(text(tr("Python 3 is not installed, so the benchmark measures only speed. Install Python 3 to check the answers.")).size(12).style(theme::warn_text));
        }
        if let Some(r) = &self.running {
            col = col.push(
                row![
                    column![progress_bar(0.0..=r.total.max(1) as f32, r.done as f32).girth(6).style(theme::meter), text(r.status.clone()).size(12).style(theme::muted)].spacing(4).width(Fill),
                    button(text(tr("Stop")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::Stop),
                ]
                .spacing(10)
                .align_y(Alignment::Center),
            );
            let mut list = Column::new().spacing(2);
            for t in &r.results {
                list = list.push(text(task_line(t)).size(12).font(fonts::mono()));
            }
            col = col.push(list);
        } else {
            col = col.push(button(text(tr("Run the benchmark")).size(13)).padding([6, 16]).style(theme::primary).on_press(Msg::Start));
        }
        if let Some(e) = &self.error {
            col = col.push(container(text(e.clone()).size(13)).padding([8, 12]).width(Fill).style(theme::error_box));
        }
        if !self.history.is_empty() {
            let mut runs = Column::new().spacing(8);
            for run in self.history.iter().take(30) {
                runs = runs.push(run_view(run));
            }
            col = col.push(text(tr("Results on this computer")).size(13).font(fonts::ui_semibold())).push(runs);
        }
        container(col).padding(12).width(Fill).style(theme::card).into()
    }
}

fn task_line(t: &TaskResult) -> String {
    let mark = match t.passed {
        Some(true) => tr("passed"),
        Some(false) => tr("failed"),
        None => tr("not checked"),
    };
    let tries = if t.tries == 1 { tr("1 try").to_string() } else { trf("{count} tries", &[("count", &t.tries)]) };
    format!("{:<16} {mark}, {tries}, {:.0} s", t.name, t.seconds)
}

fn run_view(run: &Run) -> Element<'_, Msg> {
    let when = chrono::DateTime::from_timestamp_millis(run.time).map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default();
    let set = SetChoice(run.set).to_string();
    let score = if run.checked() {
        trf("{passed} of {count} passed, {first} on the first try", &[("passed", &run.passed()), ("count", &run.tasks.len()), ("first", &run.first_try())])
    } else {
        tr("answers not checked").to_string()
    };
    let speed = trf("{minutes} minutes, {speed} tokens a second", &[("minutes", &format!("{:.1}", run.seconds() / 60.0)), ("speed", &format!("{:.1}", run.tokens_per_second()))]);
    let mut setup = vec![if run.gpu { tr("graphics card") } else { tr("processor") }.to_string()];
    if run.extras.iter().any(|a| a == "draft-mtp") {
        setup.push(tr("its own prediction layers").into());
    } else if run.extras.iter().any(|a| a == "ngram-simple") {
        setup.push(tr("text already in the conversation").into());
    } else if run.extras.iter().any(|a| a == "draft-simple") {
        setup.push(tr("a draft model").into());
    }
    if run.extras.iter().any(|a| a == "q8_0") {
        setup.push(tr("compact context memory").into());
    }
    if run.thinking {
        setup.push(tr("thinking").into());
    }
    if run.tries > 1 {
        setup.push(trf("up to {count} tries", &[("count", &run.tries)]));
    }
    column![
        text(format!("{}, {set}, {when}", run.model)).size(13).font(fonts::ui_semibold()),
        text(format!("{score}. {speed}.")).size(12),
        text(setup.join(", ")).size(12).style(theme::muted),
    ]
    .spacing(2)
    .into()
}
