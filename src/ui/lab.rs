//! The model lab page in Settings: a model's details, and the forms that make a variant of it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use iced::futures::SinkExt;
use iced::widget::{Column, button, column, container, pick_list, progress_bar, row, slider, space, text, text_input};
use iced::{Alignment, Element, Fill, Task};

use super::Message;
use super::fonts;
use super::icons::{Icon, icon};
use super::settings::{Effect, Msg as SettingsMsg};
use super::theme;
use crate::agent::Host;
use crate::i18n::{tr, trf};
use crate::llama::LocalModel;
use crate::llama::lab::{self, CONVERSIONS, Details, Job, LayerEdit, Part, Progress};
use crate::util::gb;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Steer,
    Adapter,
    Strength,
    Layers,
    Convert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartChoice(pub Part);

impl std::fmt::Display for PartChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(tr(match self.0 {
            Part::Attention => "Attention",
            Part::FeedForward => "Feed-forward",
            Part::Both => "Attention and feed-forward",
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatChoice(pub &'static str);

impl std::fmt::Display for FormatChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// One entry in a layer menu: a single layer, or a whole group on a model whose layers come in groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerChoice {
    first: u32,
    last: u32,
}

impl std::fmt::Display for LayerChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.first == self.last {
            write!(f, "{}", self.first)
        } else {
            f.write_str(&trf("{first} to {last}", &[("first", &self.first), ("last", &self.last)]))
        }
    }
}

/// Single layers from `start`, for steering.
fn single_layers(l: lab::Layers, start: u32) -> Vec<LayerChoice> {
    (start..l.main).map(|i| LayerChoice { first: i, last: i }).collect()
}

/// The whole groups of layers, or single layers when the model has no groups.
fn layer_groups(l: lab::Layers) -> Vec<LayerChoice> {
    let g = l.group.max(1);
    (0..l.main / g).map(|i| LayerChoice { first: i * g, last: i * g + g - 1 }).collect()
}

#[derive(Debug, Clone)]
pub enum Msg {
    Loaded(Result<Box<Details>, String>),
    Back,
    Tab(Tab),
    ToggleTensors,
    Name(String),
    Toward(String),
    Away(String),
    SteerStrength(f32),
    SteerFirst(LayerChoice),
    SteerLast(LayerChoice),
    Source(String),
    BrowseAdapter,
    AdapterPicked(Option<std::path::PathBuf>),
    AdapterStrength(f32),
    Part(PartChoice),
    Pick(LayerChoice),
    Unpick(LayerChoice),
    Percent(f32),
    Edit(LayerEdit),
    Target(FormatChoice),
    Start,
    Progress(Progress),
    Finished(Result<String, String>),
    Cancel,
    VariantStrength(usize, f32),
    SaveStrengths,
    Delete,
}

struct Running {
    progress: Option<Progress>,
    cancel: Arc<AtomicBool>,
}

pub struct Lab {
    model: LocalModel,
    details: Option<Result<Box<Details>, String>>,
    tab: Tab,
    show_tensors: bool,
    name: String,
    toward: String,
    away: String,
    steer_strength: f32,
    steer_first: u32,
    steer_last: u32,
    source: String,
    adapter_strength: f32,
    part: Part,
    /// The layers or groups of layers the strength and layer tabs change, in model order.
    picked: Vec<LayerChoice>,
    percent: f32,
    edit: LayerEdit,
    target: &'static str,
    running: Option<Running>,
    error: Option<String>,
    made: Option<String>,
    /// Strengths of the variant's steering vectors, then of its adapters, as the sliders show them.
    strengths: Vec<f32>,
    confirm_delete: bool,
}

impl Lab {
    /// Opens the lab for `model` and starts reading its file.
    pub fn open(model: LocalModel) -> (Lab, Task<Message>) {
        let lab = Lab {
            name: trf("{model} edit", &[("model", &model.name)]),
            model: model.clone(),
            details: None,
            tab: Tab::Steer,
            show_tensors: false,
            toward: String::new(),
            away: String::new(),
            steer_strength: 2.0,
            steer_first: 0,
            steer_last: 0,
            source: String::new(),
            adapter_strength: 1.0,
            part: Part::Attention,
            picked: Vec::new(),
            percent: 120.0,
            edit: LayerEdit::Repeat,
            target: "Q4_K_M",
            running: None,
            error: None,
            made: None,
            strengths: Vec::new(),
            confirm_delete: false,
        };
        let task = Task::perform(
            async move { tokio::task::spawn_blocking(move || lab::details(&model)).await.map_err(|e| e.to_string()).and_then(|r| r.map(Box::new).map_err(|e| format!("{e:#}"))) },
            |r| Message::Settings(SettingsMsg::Lab(Msg::Loaded(r))),
        );
        (lab, task)
    }

    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// Returns whether the page closed, the task to run, and what the settings panel should do.
    pub fn update(&mut self, msg: Msg, host: Option<&Arc<Host>>) -> (bool, Task<Message>, Effect) {
        match msg {
            Msg::Loaded(r) => {
                if let Ok(d) = &r {
                    let l = d.layers;
                    // Steering works best on the middle layers, and the first layer has no vector.
                    self.steer_first = (l.main / 4).max(1);
                    self.steer_last = (l.main * 3 / 4).max(1);
                    if let Some(v) = &d.variant {
                        self.strengths = v.steering.iter().map(|s| s.strength).chain(v.adapters.iter().map(|a| a.strength)).collect();
                    }
                }
                self.details = Some(r);
            }
            Msg::Back => {
                if let Some(r) = &self.running {
                    r.cancel.store(true, Ordering::SeqCst);
                }
                return (true, Task::none(), Effect::None);
            }
            Msg::Tab(t) => {
                self.tab = t;
                self.error = None;
            }
            Msg::ToggleTensors => self.show_tensors = !self.show_tensors,
            Msg::Name(v) => self.name = v,
            Msg::Toward(v) => self.toward = v,
            Msg::Away(v) => self.away = v,
            Msg::SteerStrength(v) => self.steer_strength = v,
            Msg::SteerFirst(c) => {
                self.steer_first = c.first;
                self.steer_last = self.steer_last.max(c.last);
            }
            Msg::SteerLast(c) => {
                self.steer_last = c.last;
                self.steer_first = self.steer_first.min(c.first);
            }
            Msg::Source(v) => self.source = v,
            Msg::BrowseAdapter => {
                return (
                    false,
                    Task::perform(
                        async { rfd::AsyncFileDialog::new().set_title(tr("Choose an adapter")).add_filter("GGUF", &["gguf"]).pick_file().await.map(|h| h.path().to_path_buf()) },
                        |p| Message::Settings(SettingsMsg::Lab(Msg::AdapterPicked(p))),
                    ),
                    Effect::None,
                );
            }
            Msg::AdapterPicked(Some(p)) => self.source = p.to_string_lossy().into_owned(),
            Msg::AdapterPicked(None) => {}
            Msg::AdapterStrength(v) => self.adapter_strength = v,
            Msg::Part(p) => self.part = p.0,
            Msg::Pick(c) => {
                if !self.picked.contains(&c) {
                    self.picked.push(c);
                    self.picked.sort_by_key(|c| c.first);
                }
            }
            Msg::Unpick(c) => self.picked.retain(|p| *p != c),
            Msg::Percent(v) => self.percent = v,
            Msg::Edit(e) => self.edit = e,
            Msg::Target(t) => self.target = t.0,
            Msg::Start => return (false, self.start(host), Effect::None),
            Msg::Progress(p) => {
                if let Some(r) = &mut self.running {
                    r.progress = Some(p);
                }
            }
            Msg::Finished(result) => {
                self.running = None;
                match result {
                    Ok(name) => {
                        self.made = Some(name);
                        return (false, Task::none(), Effect::ModelsChanged);
                    }
                    Err(e) if e == "Cancelled" || e == tr("Stopped.") => {}
                    Err(e) => self.error = Some(e),
                }
            }
            Msg::Cancel => {
                if let Some(r) = &self.running {
                    r.cancel.store(true, Ordering::SeqCst);
                }
            }
            Msg::VariantStrength(i, v) => {
                if let Some(s) = self.strengths.get_mut(i) {
                    *s = v;
                }
            }
            Msg::SaveStrengths => {
                let (Some(folder), Some(Ok(d))) = (&self.model.variant, &self.details) else { return (false, Task::none(), Effect::None) };
                let steering = d.variant.as_ref().map(|v| v.steering.len()).unwrap_or(0);
                let (s, a) = self.strengths.split_at(steering.min(self.strengths.len()));
                match lab::set_strengths(folder, s, a) {
                    Ok(()) => return (false, Task::none(), Effect::ModelsChanged),
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            }
            Msg::Delete => {
                if !self.confirm_delete {
                    self.confirm_delete = true;
                    return (false, Task::none(), Effect::None);
                }
                let Some(folder) = self.model.variant.clone() else { return (false, Task::none(), Effect::None) };
                if host.is_some_and(|h| h.llama.loaded_model().as_deref() == Some(self.model.name.as_str()) && h.model_in_use()) {
                    self.error = Some(tr("A conversation is using this model. Wait for its task to finish, then try again.").into());
                    return (false, Task::none(), Effect::None);
                }
                let llama = host.map(|h| h.llama.clone());
                let loaded = llama.as_ref().is_some_and(|l| l.loaded_model().as_deref() == Some(self.model.name.as_str()));
                let name = self.model.name.clone();
                // The server keeps the model file open, so it stops first. The folder is deleted rather than moved to the
                // trash, which would keep a converted model's gigabytes until the trash is emptied.
                return (
                    true,
                    Task::perform(
                        async move {
                            if loaded && let Some(l) = llama {
                                l.stop().await;
                            }
                            crate::llama::forget_saved(&name);
                            std::fs::remove_dir_all(&folder).map_err(|e| e.to_string())
                        },
                        |r| match r {
                            Ok(()) => Message::Settings(SettingsMsg::LabDeleted),
                            Err(e) => Message::Settings(SettingsMsg::LabFailed(e)),
                        },
                    ),
                    Effect::None,
                );
            }
        }
        (false, Task::none(), Effect::None)
    }

    fn job(&self) -> Result<Job, String> {
        let Some(Ok(d)) = &self.details else { return Err(tr("The model's details are still loading.").into()) };
        let l = d.layers;
        let picked = self.picked_layers();
        Ok(match self.tab {
            Tab::Steer => {
                let (first, last) = (self.steer_first, self.steer_last);
                if first < 1 || first > last || last >= l.main {
                    return Err(trf("Steer layers from 1 to {last}.", &[("last", &(l.main.saturating_sub(1)))]));
                }
                Job::Steer { toward: self.toward.clone(), away: self.away.clone(), strength: round(self.steer_strength), first, last }
            }
            Tab::Adapter => Job::Adapter { source: self.source.clone(), strength: round(self.adapter_strength) },
            Tab::Strength => {
                lab::check_layers(&l, &picked)?;
                if !d.scalable {
                    return Err(tr("Some of this model's layers are stored in a format the lab cannot scale.").into());
                }
                Job::Strength { part: self.part, layers: picked, percent: self.percent.round() as u32 }
            }
            Tab::Layers => {
                lab::check_layers(&l, &picked)?;
                if picked.len() as u32 + l.group.max(1) > l.main && self.edit == LayerEdit::Remove {
                    return Err(tr("At least one group of layers has to stay.").into());
                }
                Job::Layers { edit: self.edit, layers: picked }
            }
            Tab::Convert => Job::Convert { target: self.target },
        })
    }

    fn start(&mut self, host: Option<&Arc<Host>>) -> Task<Message> {
        let Some(host) = host.cloned() else { return Task::none() };
        self.error = None;
        self.made = None;
        let job = match self.job() {
            Ok(j) => j,
            Err(e) => {
                self.error = Some(e);
                return Task::none();
            }
        };
        // The model menu tells models apart by name.
        if host.llama.models().iter().any(|m| m.name.eq_ignore_ascii_case(self.name.trim())) {
            self.error = Some(tr("A model with that name is already in the models folder. Pick another name.").into());
            return Task::none();
        }
        let Some(tools) = host.llama.tools_dir() else {
            self.error = Some(tr("Scoobert could not find llama.cpp. Reinstall Scoobert, or set the llama-server path in Settings.").into());
            return Task::none();
        };
        // Steering runs the model in a separate program, so the model llama-server holds has to make room for it.
        let steering = matches!(job, Job::Steer { .. });
        if steering && host.model_in_use() {
            self.error = Some(tr("A conversation is using the model. Wait for its task to finish, then try again.").into());
            return Task::none();
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.running = Some(Running { progress: None, cancel: cancel.clone() });
        let (model, name, models_dir, http) = (self.model.clone(), self.name.clone(), host.llama.models_dir(), host.http().clone());
        let stream = iced::stream::channel(64, async move |mut out: iced::futures::channel::mpsc::Sender<Msg>| {
            if steering && host.llama.loaded_model().is_some() {
                host.llama.stop().await;
            }
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Progress>();
            let work = tokio::spawn(lab::make(http, tools, models_dir, model, name, job, cancel, move |p| {
                let _ = tx.send(p);
            }));
            while let Some(p) = rx.recv().await {
                let _ = out.send(Msg::Progress(p)).await;
            }
            let result = match work.await {
                Ok(Ok(name)) => Ok(name),
                Ok(Err(e)) if crate::util::is_cancelled(&e) => Err(tr("Stopped.").to_string()),
                Ok(Err(e)) => Err(format!("{e:#}")),
                Err(e) => Err(e.to_string()),
            };
            let _ = out.send(Msg::Finished(result)).await;
        });
        Task::run(stream, |m| Message::Settings(SettingsMsg::Lab(m)))
    }

    pub fn view(&self) -> Element<'_, Msg> {
        let back = button(row![icon(Icon::ArrowLeft, 14.0), text(tr("Models on this computer")).size(13)].spacing(4).align_y(Alignment::Center))
            .padding([4, 8])
            .style(theme::ghost)
            .on_press(Msg::Back);
        let mut col = column![back, text(trf("Model lab: {model}", &[("model", &self.model.name)])).size(20).font(fonts::ui_semibold())].spacing(12);
        match &self.details {
            None => col = col.push(text(tr("Reading the model file...")).size(13).style(theme::muted)),
            Some(Err(e)) => col = col.push(container(text(e.clone()).size(13)).padding([8, 12]).width(Fill).style(theme::error_box)),
            Some(Ok(d)) => {
                col = col.push(self.details_view(d));
                if let Some(v) = &d.variant {
                    col = col.push(self.variant_view(v));
                }
                col = col.push(self.make_view(d));
            }
        }
        col.into()
    }

    fn details_view<'a>(&'a self, d: &'a Details) -> Element<'a, Msg> {
        let l = d.layers;
        let mut layers = trf("{count} layers", &[("count", &l.main)]);
        if l.extra > 0 {
            layers.push_str(&format!(", {}", trf("plus {count} extra prediction layer", &[("count", &l.extra)])));
        }
        if l.group > 1 {
            layers.push_str(&format!(", {}", trf("in groups of {group}", &[("group", &l.group)])));
        }
        let facts = column![
            text(trf("Architecture: {name}", &[("name", &d.architecture)])).size(13),
            text(layers).size(13),
            text(trf("{billions} billion weights, {size} on disk, {width} values wide", &[
                ("billions", &format!("{:.1}", d.parameters as f64 / 1e9)),
                ("size", &gb(d.file_size)),
                ("width", &d.embedding),
            ]))
            .size(13),
            text(trf("Trained for up to {tokens} tokens of context", &[("tokens", &crate::util::thousands(d.context))])).size(13),
        ]
        .spacing(3);
        let mut formats = Column::new().spacing(2);
        for (kind, bytes, count) in &d.formats {
            let line = if *count == 1 {
                trf("{format}: {size} in 1 tensor", &[("format", kind), ("size", &crate::util::size(*bytes))])
            } else {
                trf("{format}: {size} in {count} tensors", &[("format", kind), ("size", &crate::util::size(*bytes)), ("count", count)])
            };
            formats = formats.push(text(line).size(12).style(theme::muted));
        }
        let toggle = button(text(if self.show_tensors { tr("Hide every tensor and setting") } else { tr("Show every tensor and setting") }).size(12))
            .padding([3, 10])
            .style(theme::secondary)
            .on_press(Msg::ToggleTensors);
        let mut card = column![facts, text(tr("Number formats")).size(13).font(fonts::ui_semibold()), formats, toggle].spacing(8);
        if self.show_tensors {
            let mut meta = Column::new().spacing(1);
            for (k, v) in &d.metadata {
                meta = meta.push(text(format!("{k} = {v}")).size(11).font(fonts::mono()).style(theme::muted));
            }
            let mut tensors = Column::new().spacing(1);
            for t in &d.tensors {
                tensors = tensors.push(text(format!("{}  [{}]  {}  {}", t.name, t.shape, t.kind, crate::util::size(t.bytes))).size(11).font(fonts::mono()));
            }
            card = card.push(text(tr("Settings in the file")).size(13).font(fonts::ui_semibold())).push(meta);
            card = card.push(text(tr("Tensors")).size(13).font(fonts::ui_semibold())).push(tensors);
        }
        container(card).padding(12).width(Fill).style(theme::card).into()
    }

    fn variant_view<'a>(&'a self, v: &'a lab::Variant) -> Element<'a, Msg> {
        let mut col = column![
            text(trf("Made from {model}", &[("model", &v.made_from)])).size(13).font(fonts::ui_semibold()),
        ]
        .spacing(6);
        for c in &v.changes {
            col = col.push(text(format!("- {c}")).size(12).style(theme::muted));
        }
        let mut i = 0;
        for s in &v.steering {
            let value = self.strengths.get(i).copied().unwrap_or(s.strength);
            let at = i;
            col = col.push(strength_row(trf("Steering toward \"{toward}\"", &[("toward", &s.toward)]), value, -6.0..=6.0, move |x| Msg::VariantStrength(at, x)));
            i += 1;
        }
        for a in &v.adapters {
            let value = self.strengths.get(i).copied().unwrap_or(a.strength);
            let at = i;
            col = col.push(strength_row(trf("Adapter {source}", &[("source", &a.source)]), value, 0.0..=2.0, move |x| Msg::VariantStrength(at, x)));
            i += 1;
        }
        let mut actions = row![].spacing(8).align_y(Alignment::Center);
        if i > 0 {
            actions = actions.push(button(text(tr("Save strengths")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::SaveStrengths));
        }
        let delete = if self.confirm_delete { tr("Click again to delete this variant for good") } else { tr("Delete this variant") };
        actions = actions.push(button(text(delete).size(12)).padding([4, 10]).style(theme::danger).on_press(Msg::Delete));
        col = col.push(actions);
        container(col).padding(12).width(Fill).style(theme::card).into()
    }

    fn make_view<'a>(&'a self, d: &'a Details) -> Element<'a, Msg> {
        let tab = |label: &'static str, t: Tab| {
            button(text(label).size(12)).padding([4, 10]).style(if self.tab == t { theme::secondary } else { theme::ghost }).on_press(Msg::Tab(t))
        };
        let tabs = row![
            tab(tr("Steering"), Tab::Steer),
            tab(tr("Adapter"), Tab::Adapter),
            tab(tr("Layer strength"), Tab::Strength),
            tab(tr("Layer surgery"), Tab::Layers),
            tab(tr("Size conversion"), Tab::Convert),
        ]
        .spacing(4)
        .wrap();
        let l = d.layers;
        let steer_layers = {
            let options = single_layers(l, 1);
            let first = options.iter().find(|c| c.first == self.steer_first).copied();
            let last = options.iter().find(|c| c.last == self.steer_last).copied();
            let menu = |picked: Option<LayerChoice>, on: fn(LayerChoice) -> Msg| {
                pick_list(options.clone(), picked, on).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu)
            };
            row![
                text(tr("First layer")).size(13),
                menu(first, Msg::SteerFirst),
                space().width(8),
                text(tr("Last layer")).size(13),
                menu(last, Msg::SteerLast),
            ]
            .spacing(8)
            .align_y(Alignment::Center)
        };
        let body: Element<'a, Msg> = match self.tab {
            Tab::Steer => column![
                text(tr("Scoobert runs the model on 24 pairs of the same question with two different personas, measures how each layer responds differently, and adds that difference while the variant runs. Describe each persona as an instruction to the model.")).size(12).style(theme::muted),
                text_input(tr("Toward, such as: You answer like a pirate captain."), &self.toward).on_input(Msg::Toward).size(13).padding([6, 10]).style(theme::input),
                text_input(tr("Away from, such as: You answer plainly. (Optional)"), &self.away).on_input(Msg::Away).size(13).padding([6, 10]).style(theme::input),
                strength_row(tr("Strength").to_string(), self.steer_strength, -6.0..=6.0, Msg::SteerStrength),
                steer_layers,
                text(tr("The middle layers respond best, so the menus start on the middle half of the model. A narrower range pushes harder on each layer, so it may need a lower strength.")).size(12).style(theme::muted),
                text(trf("At a strength of 2 the replies lean clearly toward the persona, at 4 they take it on fully, and at about 6 they stop making sense. A negative strength steers away from it. Making the variant takes a few minutes, and llama-server unloads the model meanwhile, so it needs about {memory} of free memory.", &[("memory", &gb(self.model.size + 1_000_000_000))])).size(12).style(theme::muted),
            ]
            .spacing(8)
            .into(),
            Tab::Adapter => column![
                text(tr("A LoRA adapter in GGUF format, trained for this kind of model, changes how it writes. Give a Hugging Face repository as owner/repository, or a file on this computer.")).size(12).style(theme::muted),
                row![
                    text_input(tr("owner/repository, or a file"), &self.source).on_input(Msg::Source).size(13).padding([6, 10]).style(theme::input),
                    button(text(tr("Browse")).size(13)).padding([6, 12]).style(theme::secondary).on_press(Msg::BrowseAdapter),
                ]
                .spacing(8)
                .align_y(Alignment::Center),
                strength_row(tr("Strength").to_string(), self.adapter_strength, 0.0..=2.0, Msg::AdapterStrength),
            ]
            .spacing(8)
            .into(),
            Tab::Strength => {
                let parts = [Part::Attention, Part::FeedForward, Part::Both].map(PartChoice);
                column![
                    text(tr("Turns up or down how much the chosen layers add to what the model is working on. Attention is what a layer takes from the earlier text, and feed-forward is what it adds from what it learned. Scoobert changes the scale stored with each block of weights, so nothing else in the model changes.")).size(12).style(theme::muted),
                    row![
                        text(tr("Part")).size(13),
                        pick_list(parts, Some(PartChoice(self.part)), Msg::Part).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                    self.layer_list(l),
                    row![
                        text(tr("Strength")).size(13).width(110),
                        slider(0.0..=300.0, self.percent, Msg::Percent).step(5.0f32).style(theme::slider),
                        text(format!("{:.0}%", self.percent)).size(13).width(50),
                    ]
                    .spacing(10)
                    .align_y(Alignment::Center),
                    self.list_note(l),
                ]
                .spacing(8)
                .into()
            }
            Tab::Layers => {
                let choice = |label: &'static str, e: LayerEdit| {
                    button(text(label).size(12)).padding([4, 10]).style(if self.edit == e { theme::secondary } else { theme::ghost }).on_press(Msg::Edit(e))
                };
                column![
                    text(tr("Repeating layers makes the model run them twice, which often changes how it reasons and writes. Removing layers makes it smaller and faster, and usually worse.")).size(12).style(theme::muted),
                    row![choice(tr("Repeat"), LayerEdit::Repeat), choice(tr("Remove"), LayerEdit::Remove)].spacing(4),
                    self.layer_list(l),
                    self.list_note(l),
                ]
                .spacing(8)
                .into()
            }
            Tab::Convert => {
                let options: Vec<FormatChoice> = CONVERSIONS.iter().map(|c| FormatChoice(c.0)).collect();
                let bits = CONVERSIONS.iter().find(|c| c.0 == self.target).map(|c| c.1).unwrap_or(8.0);
                column![
                    text(tr("Writes a copy in another number format. A smaller format needs less memory and runs faster but makes more mistakes. Converting an already compressed model loses a little more each time, and a larger format cannot bring back what was lost.")).size(12).style(theme::muted),
                    row![
                        text(tr("Format")).size(13),
                        pick_list(options, Some(FormatChoice(self.target)), Msg::Target).padding([5, 10]).text_size(13).style(theme::select).menu_style(theme::menu),
                        text(trf("about {size}", &[("size", &gb((d.parameters as f64 * bits / 8.0) as u64))])).size(12).style(theme::muted),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                ]
                .spacing(8)
                .into()
            }
        };
        let name = row![
            text(tr("New model's name")).size(13),
            text_input("", &self.name).on_input(Msg::Name).size(13).padding([6, 10]).style(theme::input),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let mut col = column![text(tr("Make a variant")).size(15).font(fonts::ui_semibold()), tabs, body, name].spacing(10);
        if let Some(r) = &self.running {
            let (done, total, status) = r.progress.as_ref().map(|p| (p.done, p.total.max(1), p.status.clone())).unwrap_or((0, 1, tr("Starting...").into()));
            col = col.push(
                row![
                    column![progress_bar(0.0..=total as f32, done as f32).girth(6).style(theme::meter), text(status).size(12).style(theme::muted)].spacing(4).width(Fill),
                    button(text(tr("Stop")).size(12)).padding([4, 10]).style(theme::secondary).on_press(Msg::Cancel),
                ]
                .spacing(10)
                .align_y(Alignment::Center),
            );
        } else {
            col = col.push(button(text(tr("Make the variant")).size(13)).padding([6, 16]).style(theme::primary).on_press(Msg::Start));
        }
        if let Some(e) = &self.error {
            col = col.push(container(text(e.clone()).size(13)).padding([8, 12]).width(Fill).style(theme::error_box));
        }
        if let Some(n) = &self.made {
            col = col.push(text(trf("Made {model}. Pick it in the model menu to try it.", &[("model", n)])).size(13).style(theme::accent_text));
        }
        container(col).padding(12).width(Fill).style(theme::card).into()
    }
}

impl Lab {
    fn picked_layers(&self) -> Vec<u32> {
        self.picked.iter().flat_map(|c| c.first..=c.last).collect()
    }

    /// The picked layers as buttons that take them off the list, then a menu that adds more.
    fn layer_list(&self, l: lab::Layers) -> Element<'_, Msg> {
        let grouped = l.group > 1;
        let left: Vec<LayerChoice> = layer_groups(l).into_iter().filter(|c| !self.picked.contains(c)).collect();
        let add = if grouped { tr("Add a group") } else { tr("Add a layer") };
        let menu = pick_list(left, None::<LayerChoice>, Msg::Pick).placeholder(add).padding([5, 10]).text_size(13).width(150).style(theme::select).menu_style(theme::menu);
        let top = row![text(if grouped { tr("Groups") } else { tr("Layers") }).size(13), menu].spacing(8).align_y(Alignment::Center);
        if self.picked.is_empty() {
            return top.into();
        }
        let mut chips = row![].spacing(8);
        for &c in &self.picked {
            let label = row![text(c.to_string()).size(13), icon(Icon::Close, 11.0)].spacing(6).align_y(Alignment::Center);
            chips = chips.push(button(label).padding([5, 8]).style(theme::secondary).on_press(Msg::Unpick(c)));
        }
        column![top, chips.wrap().vertical_spacing(6)].spacing(8).into()
    }

    /// What the picked layers do, or why they cannot be used.
    fn list_note(&self, l: lab::Layers) -> Element<'_, Msg> {
        let picked = self.picked_layers();
        if picked.is_empty() {
            let line = if l.group > 1 {
                trf("This model's layers come in groups of {group}, and the lab changes whole groups. Add groups from the menu.", &[("group", &l.group)])
            } else {
                tr("Add layers from the menu.").into()
            };
            return text(line).size(12).style(theme::muted).into();
        }
        let count = picked.len() as u32;
        let line = match (self.tab, self.edit) {
            (Tab::Layers, LayerEdit::Repeat) => {
                // Each repeated block is listed on its own, so 12 to 19 then 12 to 19 again reads as a repeat.
                let mut parts = Vec::new();
                let mut next = 0;
                for (first, last) in lab::blocks(&picked) {
                    parts.push((next, last));
                    parts.push((first, last));
                    next = last + 1;
                }
                if next < l.main {
                    parts.push((next, l.main - 1));
                }
                trf(
                    "Layers in the new model: {order}. That is {total} layers instead of {main}.",
                    &[("order", &lab::describe(&parts)), ("total", &(l.main + count)), ("main", &l.main)],
                )
            }
            (Tab::Layers, LayerEdit::Remove) if count + l.group.max(1) > l.main => {
                return text(tr("At least one group of layers has to stay.")).size(12).style(theme::warn_text).into();
            }
            (Tab::Layers, LayerEdit::Remove) => trf(
                "The new model skips layers {layers}, so it has {total} layers instead of {main}.",
                &[("layers", &lab::describe(&lab::blocks(&picked))), ("total", &(l.main - count)), ("main", &l.main)],
            ),
            _ => trf("Changes {count} of the model's {main} layers.", &[("count", &count), ("main", &l.main)]),
        };
        text(line).size(12).style(theme::muted).into()
    }
}

/// Two decimals are enough for a strength, and the server's argument stays short.
fn round(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

fn strength_row<'a>(label: String, value: f32, range: std::ops::RangeInclusive<f32>, on_change: impl Fn(f32) -> Msg + 'a) -> Element<'a, Msg> {
    row![
        text(label).size(13).width(Fill),
        slider(range, value, on_change).step(0.05f32).width(200).style(theme::slider),
        text(format!("{value:.2}")).size(13).width(44),
        space().width(4),
    ]
    .spacing(10)
    .align_y(Alignment::Center)
    .into()
}
