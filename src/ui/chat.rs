//! The conversation transcript: messages, tool calls, approvals, and the streaming reply.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use iced::widget::{Column, button, column, container, markdown, progress_bar, rich_text, row, rule, scrollable, space, text};
use iced::{Alignment, Element, Fill, Length, Padding, Theme};

use super::Message;
use super::fonts;
use super::icons::{Icon, icon};
use super::theme::{self, tokens};
use crate::agent::conversation::{AssistantMessage, Message as AgentMessage, StopReason, ToolCall, ToolResult};
use crate::agent::stream::Delta;
use crate::agent::{ConvId, Decision, Snapshot};
use crate::store::Thinking;
use crate::util::{about_duration, clip, thousands};

pub const TRANSCRIPT_ID: &str = "transcript";
const COMPACTED: &str = "Scoobert summarized the messages above to make room and continued from the summary. The summary is in the Tasks folder of the notes.";
pub const MAX_WIDTH: f32 = 820.0;

pub struct ToolCard {
    pub call: ToolCall,
    pub result: Option<ToolResult>,
}

pub enum Entry {
    /// `index` is the message's position in the conversation, which a rewind goes back to.
    User { text: String, notes: Vec<String>, images: usize, time: i64, index: usize },
    Assistant { key: String, thinking: String, text: String, md: markdown::Content, tools: Vec<ToolCard>, error: Option<String>, stop: StopReason, time: i64 },
    Notice(String),
}

#[derive(Default)]
pub struct Streaming {
    pub thinking: String,
    pub text: String,
    pub md: markdown::Content,
    pub tool: Option<String>,
    pub tool_chars: usize,
    /// Tokens the local model has generated for this reply, including any it holds back until a step completes.
    pub generated: u64,
}

pub struct Chat {
    pub id: ConvId,
    pub cwd: PathBuf,
    pub file: PathBuf,
    pub title: String,
    pub model: String,
    pub thinking: Thinking,
    pub context: u64,
    pub entries: Vec<Entry>,
    pub running: bool,
    pub activity: Option<String>,
    pub progress: Option<(u64, u64)>,
    /// When the current read started and how far it was then, for the time left.
    progress_since: Option<(std::time::Instant, u64)>,
    pub stream: Option<Streaming>,
    pub approvals: Vec<(u64, ToolCall)>,
    pub live_output: HashMap<String, String>,
    pub expanded: HashSet<String>,
    pub notice: Option<String>,
    pub error: Option<String>,
    /// A message sent but not yet confirmed by the host, shown right away.
    pub pending: Option<String>,
    /// The last task stopped before it finished, so the transcript offers Continue.
    pub interrupted: bool,
    counter: usize,
    /// Messages added so far, so each user message knows its position.
    seen: usize,
}

impl Chat {
    pub fn from_snapshot(s: Snapshot) -> Chat {
        let mut chat = Chat {
            id: s.id,
            cwd: s.cwd,
            file: s.file,
            title: s.title,
            model: s.model,
            thinking: s.thinking,
            context: s.context,
            entries: Vec::new(),
            running: s.running,
            activity: None,
            progress: None,
            progress_since: None,
            stream: None,
            approvals: Vec::new(),
            live_output: HashMap::new(),
            expanded: HashSet::new(),
            notice: s.notice,
            error: None,
            pending: None,
            interrupted: s.interrupted,
            counter: 0,
            seen: 0,
        };
        for (i, m) in s.messages.into_iter().enumerate() {
            if s.compacted_at == Some(i) {
                chat.entries.push(Entry::Notice(COMPACTED.into()));
            }
            chat.push(m);
        }
        chat
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.pending.is_none()
    }

    fn push(&mut self, m: AgentMessage) {
        let index = self.seen;
        self.seen += 1;
        match m {
            AgentMessage::User(u) => {
                let notes = note_paths(&u.context);
                if self.entries.is_empty() && self.title == "New conversation" {
                    self.title = crate::agent::conversation::quick_title(&u.text);
                }
                self.entries.push(Entry::User { text: u.text, notes, images: u.images.len(), time: u.time, index });
            }
            AgentMessage::Assistant(a) => self.push_assistant(a),
            AgentMessage::Tool(t) => {
                for e in self.entries.iter_mut().rev() {
                    if let Entry::Assistant { tools, .. } = e
                        && let Some(card) = tools.iter_mut().find(|c| c.call.id == t.call_id)
                    {
                        self.live_output.remove(&t.call_id);
                        card.result = Some(t);
                        return;
                    }
                }
            }
        }
    }

    fn push_assistant(&mut self, a: AssistantMessage) {
        self.counter += 1;
        let key = format!("a{}", self.counter);
        let tools = a.tool_calls.into_iter().map(|call| ToolCard { call, result: None }).collect();
        self.entries.push(Entry::Assistant {
            key,
            thinking: a.thinking,
            md: markdown::Content::parse(&a.text),
            text: a.text,
            tools,
            error: a.error,
            stop: a.stop,
            time: a.time,
        });
    }

    /// Applies one host event for this conversation. Returns true when the view should follow the bottom.
    pub fn apply(&mut self, event: crate::agent::Event) {
        use crate::agent::Event as E;
        match event {
            E::Activity { text, .. } => {
                self.activity = text;
                if self.activity.is_none() {
                    self.progress = None;
                }
            }
            // Reading progress is shown on the activity line, so it does not open an empty reply.
            E::Delta { delta: Delta::Progress { done, total, prompt }, .. } => {
                if self.progress.is_none_or(|(d, t)| done < d || total != t) {
                    self.progress_since = Some((std::time::Instant::now(), done));
                }
                self.progress = Some((done, total));
                if prompt > 0 {
                    self.context = prompt;
                }
            }
            E::Delta { delta, .. } => {
                let s = self.stream.get_or_insert_with(Streaming::default);
                match delta {
                    Delta::Thinking(t) => s.thinking.push_str(&t),
                    Delta::Text(t) => {
                        s.text.push_str(&t);
                        s.md.push_str(&t);
                    }
                    Delta::ToolCall(name) => {
                        s.tool = Some(name);
                        s.tool_chars = 0;
                    }
                    Delta::ToolInput(chars) => s.tool_chars = chars,
                    Delta::Generated(n) => {
                        s.generated = n;
                        self.activity = None;
                        self.progress = None;
                    }
                    Delta::Progress { .. } => {}
                }
            }
            E::Message { message, .. } => {
                // A tool result joins the context before the next request counts it, so it is estimated now.
                if let AgentMessage::Tool(t) = &message {
                    self.context += (t.output.chars().count() as f64 / 3.5) as u64;
                }
                if matches!(message, AgentMessage::User(_)) {
                    self.pending = None;
                    self.error = None;
                }
                if let AgentMessage::Assistant(a) = &message {
                    self.stream = None;
                    self.progress = None;
                    // The meter follows each step of a task, not only the end of it.
                    if let Some(u) = a.usage {
                        self.context = u.input + u.output;
                    }
                }
                self.push(message);
            }
            E::ToolStarted { call, .. } => {
                self.live_output.insert(call.id.clone(), String::new());
            }
            E::ToolOutput { call_id, tail, .. } => {
                self.live_output.insert(call_id, tail);
            }
            E::Approval { id, call, .. } => self.approvals.push((id, call)),
            E::Settled { context, interrupted, .. } => {
                self.running = false;
                self.interrupted = interrupted;
                self.stream = None;
                self.activity = None;
                self.progress = None;
                self.approvals.clear();
                self.pending = None;
                if context > 0 {
                    self.context = context;
                }
            }
            E::Error { message, .. } => self.error = Some(message),
            E::Compacted { .. } => self.entries.push(Entry::Notice(COMPACTED.into())),
            E::Titled { title, .. } => self.title = title,
            _ => {}
        }
    }

    /// The activity, with how far a long read has got and the time left once its speed is known.
    fn activity_label(&self, activity: &str) -> String {
        let Some((done, total)) = self.progress.filter(|(_, t)| *t > 0) else { return activity.to_string() };
        let activity = activity.trim_end_matches("...");
        let mut label = format!("{activity}: {} of {} tokens", thousands(done), thousands(total));
        if let Some((since, from)) = self.progress_since
            && done > from
            && total > done
        {
            let rate = (done - from) as f64 / since.elapsed().as_secs_f64().max(0.001);
            label.push_str(&format!(", {} left", about_duration(((total - done) as f64 / rate) as u64)));
        }
        label
    }

    pub fn view<'a>(&'a self, md_settings: markdown::Settings) -> Element<'a, Message> {
        let mut items = Column::new().spacing(14).padding(Padding { top: 24.0, right: 24.0, bottom: 24.0, left: 24.0 }).max_width(MAX_WIDTH);
        if let Some(n) = &self.notice {
            items = items.push(container(text(n).size(13)).padding([8, 12]).width(Fill).style(theme::banner));
        }
        for entry in &self.entries {
            items = items.push(self.entry(entry, md_settings));
        }
        if let Some(p) = &self.pending {
            items = items.push(user_bubble(p, &[], 0, None));
        }
        if let Some(s) = &self.stream {
            items = items.push(self.streaming(s, md_settings));
        }
        for (id, call) in &self.approvals {
            items = items.push(approval(*id, call));
        }
        if let Some(a) = &self.activity {
            let mut line = column![row![working_dot(), text(self.activity_label(a)).size(13).style(theme::muted)].spacing(8).align_y(Alignment::Center)].spacing(6);
            if let Some((done, total)) = self.progress.filter(|(_, t)| *t > 0) {
                let bar = progress_bar(0.0..=total as f32, done.min(total) as f32).girth(4).style(theme::meter);
                line = line.push(container(bar).max_width(420).padding(Padding { left: 16.0, ..Padding::ZERO }));
            }
            items = items.push(line);
        } else if self.running && self.approvals.is_empty() && self.live_output.is_empty() && self.stream.as_ref().is_none_or(|s| s.text.is_empty() && s.thinking.is_empty() && s.tool.is_none()) {
            let generated = self.stream.as_ref().map(|s| s.generated).unwrap_or(0);
            let label = if generated > 0 { format!("Working... {} tokens written so far", thousands(generated)) } else { "Working...".to_string() };
            items = items.push(row![working_dot(), text(label).size(13).style(theme::muted)].spacing(8).align_y(Alignment::Center));
        }
        if self.interrupted && !self.running && self.pending.is_none() {
            items = items.push(
                container(
                    row![
                        column![
                            text("This task stopped before it finished.").size(14).font(fonts::ui_semibold()),
                            text("Scoobert was closed or stopped partway through. Continue picks up from the last saved step.").size(12).style(theme::muted),
                        ]
                        .spacing(2)
                        .width(Fill),
                        button(text("Continue").size(13)).padding([6, 16]).style(theme::primary).on_press(Message::Continue),
                    ]
                    .spacing(12)
                    .align_y(Alignment::Center),
                )
                .padding(12)
                .width(Fill)
                .style(theme::banner),
            );
        }
        if let Some(e) = &self.error {
            items = items.push(container(text(e).size(13)).padding([10, 12]).width(Fill).style(theme::error_box));
        }
        scrollable(container(items).center_x(Fill))
            .id(TRANSCRIPT_ID)
            .anchor_bottom()
            .height(Fill)
            .style(theme::scrollbar)
            .into()
    }

    fn entry<'a>(&'a self, entry: &'a Entry, md_settings: markdown::Settings) -> Element<'a, Message> {
        match entry {
            Entry::User { text, notes, images, time, index } => sent_at(user_bubble(text, notes, *images, Some(*index)), *time),
            Entry::Notice(n) => row![
                rule::horizontal(1).style(theme::divider),
                text(n.clone()).size(12).style(theme::muted).width(Length::Shrink),
                rule::horizontal(1).style(theme::divider),
            ]
            .spacing(10)
            .align_y(Alignment::Center)
            .into(),
            Entry::Assistant { key, thinking, text: raw, md, tools, error, stop, time } => {
                let mut col = Column::new().spacing(10);
                if !thinking.trim().is_empty() {
                    col = col.push(self.thinking_block(key, thinking));
                }
                if !md.items().is_empty() {
                    col = col.push(markdown::view_with(md.items(), md_settings, &Viewer));
                    col = col.push(reply_actions(raw));
                }
                for (i, card) in tools.iter().enumerate() {
                    col = col.push(self.tool_card(&format!("{key}-t{i}"), card));
                }
                if *stop == StopReason::Aborted {
                    col = col.push(text("Stopped.").size(13).style(theme::muted));
                }
                if *stop == StopReason::Length {
                    col = col.push(text("The reply reached its length limit.").size(13).style(theme::warn_text));
                }
                if let Some(e) = error
                    && self.error.as_deref() != Some(e.as_str())
                {
                    col = col.push(container(text(e).size(13)).padding([10, 12]).width(Fill).style(theme::error_box));
                }
                sent_at(col.into(), *time)
            }
        }
    }

    fn thinking_block<'a>(&'a self, key: &str, thinking: &'a str) -> Element<'a, Message> {
        let id = format!("{key}-think");
        let open = self.expanded.contains(&id);
        let header = button(
            row![icon(if open { Icon::ChevronDown } else { Icon::ChevronRight }, 14.0), text("Thinking").size(13)]
                .spacing(6)
                .align_y(Alignment::Center),
        )
        .padding([2, 6])
        .style(theme::ghost)
        .on_press(Message::Toggle(id));
        let mut col = column![header].spacing(6);
        if open {
            col = col.push(text(thinking.trim()).size(13).style(theme::muted));
        }
        row![container(space()).width(2).height(Length::Shrink).style(theme::thread_line), col].spacing(10).into()
    }

    fn streaming<'a>(&'a self, s: &'a Streaming, md_settings: markdown::Settings) -> Element<'a, Message> {
        let mut col = Column::new().spacing(10);
        if !s.thinking.trim().is_empty() {
            let tail: String = {
                let t = s.thinking.trim();
                let start = t.char_indices().rev().nth(400).map(|(i, _)| i).unwrap_or(0);
                t[start..].to_string()
            };
            let open = self.expanded.contains("stream-think");
            let label = if s.text.is_empty() && s.tool.is_none() { "Thinking..." } else { "Thinking" };
            let header = button(
                row![icon(if open { Icon::ChevronDown } else { Icon::ChevronRight }, 14.0), text(label).size(13)]
                    .spacing(6)
                    .align_y(Alignment::Center),
            )
            .padding([2, 6])
            .style(theme::ghost)
            .on_press(Message::Toggle("stream-think".into()));
            let body: Element<'a, Message> = if open {
                text(s.thinking.trim()).size(13).style(theme::muted).into()
            } else if s.text.is_empty() {
                text(tail).size(12).style(theme::muted).into()
            } else {
                space().into()
            };
            col = col.push(row![container(space()).width(2).style(theme::thread_line), column![header, body].spacing(6)].spacing(10));
        }
        if !s.md.items().is_empty() {
            col = col.push(markdown::view_with(s.md.items(), md_settings, &Viewer));
        }
        if let Some(tool) = &s.tool {
            col = col.push(
                row![
                    working_dot(),
                    text(format!("{} ...", verb(tool))).size(13).font(fonts::ui_semibold()),
                    text(if s.tool_chars > 0 { format!("{} characters so far", thousands(s.tool_chars as u64)) } else { String::new() })
                        .size(12)
                        .style(theme::muted),
                ]
                    .spacing(8)
                    .align_y(Alignment::Center),
            );
        }
        col.into()
    }

    fn tool_card<'a>(&'a self, id: &str, card: &'a ToolCard) -> Element<'a, Message> {
        let open = self.expanded.contains(id);
        let running = card.result.is_none() && self.live_output.contains_key(&card.call.id);
        let status: fn(&theme::Tokens) -> iced::Color = match &card.result {
            Some(r) if r.is_error => |t| t.danger,
            Some(_) => |t| t.ok,
            None if running => |t| t.warn,
            None => |t| t.muted,
        };
        let detail = card
            .result
            .as_ref()
            .and_then(|r| r.diff.as_ref().map(|d| diff_stats(d)).or_else(|| (card.call.name == "read" && !r.is_error).then(|| format!("{} lines", r.output.lines().count()))))
            .unwrap_or_default();
        let header = button(
            row![
                container(space()).width(8).height(8).style(theme::dot(status)),
                text(verb(&card.call.name)).size(14).font(fonts::ui_semibold()),
                text(clip(&target(&card.call), 90)).size(13).font(fonts::mono()).style(theme::muted),
                space::horizontal(),
                text(detail).size(12).style(theme::muted),
                icon(if open { Icon::ChevronDown } else { Icon::ChevronRight }, 14.0),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        )
        .width(Fill)
        .padding([4, 0])
        .style(theme::row_button)
        .on_press(Message::Toggle(id.to_string()));
        let mut col = column![header].spacing(6);
        let live = self.live_output.get(&card.call.id).filter(|o| !o.is_empty());
        if open || running && live.is_some() {
            let body: Element<'a, Message> = match (&card.result, live) {
                (Some(r), _) => match &r.diff {
                    Some(d) if !r.is_error => diff_view(d),
                    _ => output_view(&r.output),
                },
                (None, Some(out)) => output_view(out),
                (None, None) => output_view(&serde_json::to_string_pretty(&card.call.arguments).unwrap_or_default()),
            };
            col = col.push(row![container(space()).width(2).style(theme::thread_line), body].spacing(10));
        }
        col.into()
    }
}

/// Shows how long ago a message was sent, and its date and time, while the pointer rests on it.
fn sent_at<'a>(content: Element<'a, Message>, time: i64) -> Element<'a, Message> {
    if time <= 0 {
        return content;
    }
    let tip = column![text(crate::util::ago_long(time)).size(12), text(crate::util::date_time(time)).size(11).style(theme::muted)].spacing(1);
    iced::widget::tooltip(content, container(tip).padding([4, 8]).style(theme::tooltip), iced::widget::tooltip::Position::FollowCursor)
        .delay(std::time::Duration::from_millis(700))
        .into()
}

fn user_bubble<'a>(message: &str, notes: &[String], images: usize, index: Option<usize>) -> Element<'a, Message> {
    let mut col = column![text(message.to_string()).size(15)].spacing(8);
    let mut chips = row![].spacing(6);
    for n in notes {
        chips = chips.push(container(text(format!("Note: {n}")).size(12).font(fonts::mono())).padding([2, 8]).style(theme::chip));
    }
    if images > 0 {
        let label = if images == 1 { "1 image".to_string() } else { format!("{images} images") };
        chips = chips.push(container(text(label).size(12)).padding([2, 8]).style(theme::chip));
    }
    if !notes.is_empty() || images > 0 {
        col = col.push(chips.wrap());
    }
    let copy = button(icon(Icon::Copy, 14.0)).padding(4).style(theme::ghost).on_press(Message::Copy(message.to_string()));
    let mut actions = row![copy].spacing(2);
    if let Some(index) = index {
        let rewind = button(icon(Icon::Undo, 14.0)).padding(4).style(theme::ghost).on_press(Message::AskRewind(index, message.to_string()));
        actions = actions.push(iced::widget::tooltip(
            rewind,
            container(text("Rewind to this message").size(12)).padding([4, 8]).style(theme::tooltip),
            iced::widget::tooltip::Position::Top,
        ));
    }
    container(row![col.width(Fill), actions].spacing(8)).padding([12, 16]).width(Fill).style(theme::bubble).into()
}

fn reply_actions<'a>(raw: &str) -> Element<'a, Message> {
    let action = |i: Icon, label: &'static str, m: Message| {
        button(row![icon(i, 13.0), text(label).size(12)].spacing(4).align_y(Alignment::Center)).padding([2, 6]).style(theme::ghost).on_press(m)
    };
    row![
        action(Icon::Copy, "Copy", Message::Copy(raw.to_string())),
        action(Icon::File, "Save as note", Message::SaveAsNote(raw.to_string())),
    ]
    .spacing(4)
    .into()
}

fn approval<'a>(id: u64, call: &'a ToolCall) -> Element<'a, Message> {
    let what = match call.name.as_str() {
        "edit" => format!("Scoobert wants to edit {}", call.arg("path")),
        "write" => format!("Scoobert wants to write {}", call.arg("path")),
        "bash" | "powershell" => "Scoobert wants to run a command".to_string(),
        other => format!("Scoobert wants to use {other}"),
    };
    let preview: Element<'a, Message> = match call.name.as_str() {
        "edit" => diff_view(&crate::agent::tools::unified_diff(
            call.arguments.get("old_text").and_then(|v| v.as_str()).unwrap_or_default(),
            call.arguments.get("new_text").and_then(|v| v.as_str()).unwrap_or_default(),
        )),
        "write" => output_view(&clip(call.arg("content"), 3000)),
        "bash" | "powershell" => output_view(call.arg("command")),
        _ => output_view(&serde_json::to_string_pretty(&call.arguments).unwrap_or_default()),
    };
    let buttons = row![
        button(text("Allow").size(13)).padding([6, 14]).style(theme::primary).on_press(Message::Approve(id, Decision::Allow)),
        button(text(format!("Allow {} for this conversation", verb(&call.name).to_lowercase())).size(13))
            .padding([6, 14])
            .style(theme::secondary)
            .on_press(Message::Approve(id, Decision::Always)),
        space::horizontal(),
        button(text("Deny").size(13)).padding([6, 14]).style(theme::danger).on_press(Message::Approve(id, Decision::Deny)),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    container(column![text(what).size(14).font(fonts::ui_semibold()), preview, buttons].spacing(10))
        .padding(14)
        .width(Fill)
        .style(theme::selected_card)
        .into()
}

fn working_dot<'a>() -> Element<'a, Message> {
    container(space()).width(8).height(8).style(theme::dot(|t| t.accent)).into()
}

pub fn verb(tool: &str) -> String {
    match tool {
        "read" => "Read",
        "edit" => "Edit",
        "write" => "Write",
        "bash" | "powershell" => "Run",
        "web_search" => "Search",
        "web_read" => "Read page",
        "new_project" => "Start project",
        other => other,
    }
    .to_string()
}

fn target(call: &ToolCall) -> String {
    match call.name.as_str() {
        "bash" | "powershell" => call.arg("command").lines().next().unwrap_or_default().to_string(),
        "web_search" => call.arg("query").to_string(),
        "web_read" => match call.arguments.get("find").and_then(|v| v.as_str()) {
            Some(find) => format!("{} (looking for {find})", call.arg("url")),
            None => call.arg("url").to_string(),
        },
        "new_project" => call.arg("name").to_string(),
        _ => call.arg("path").to_string(),
    }
}

fn diff_stats(diff: &str) -> String {
    let added = diff.lines().filter(|l| l.starts_with('+') && !l.starts_with("+++")).count();
    let removed = diff.lines().filter(|l| l.starts_with('-') && !l.starts_with("---")).count();
    format!("+{added} -{removed}")
}

const MAX_SHOWN_LINES: usize = 400;

fn output_view<'a>(output: &str) -> Element<'a, Message> {
    let lines: Vec<&str> = output.lines().collect();
    let shown = if lines.len() > MAX_SHOWN_LINES { lines[lines.len() - MAX_SHOWN_LINES..].join("\n") } else { lines.join("\n") };
    let body = scrollable(text(shown).size(12.5).font(fonts::mono()))
        .direction(scrollable::Direction::Both { vertical: scrollable::Scrollbar::new(), horizontal: scrollable::Scrollbar::new() })
        .style(theme::scrollbar);
    container(body).padding(10).width(Fill).max_height(320).style(theme::code_block).into()
}

fn diff_view<'a>(diff: &str) -> Element<'a, Message> {
    let mut col = Column::new();
    for line in diff.lines().filter(|l| !l.starts_with("---") && !l.starts_with("+++")).take(MAX_SHOWN_LINES) {
        let kind = line.chars().next().unwrap_or(' ');
        let color: fn(&Theme) -> iced::widget::text::Style = match kind {
            '@' => theme::muted,
            _ => |t: &Theme| iced::widget::text::Style { color: Some(tokens(t).text) },
        };
        col = col.push(container(text(line.to_string()).size(12.5).font(fonts::mono()).style(color)).width(Fill).padding([0, 6]).style(theme::diff_line(kind)));
    }
    let body = scrollable(col)
        .direction(scrollable::Direction::Both { vertical: scrollable::Scrollbar::new(), horizontal: scrollable::Scrollbar::new() })
        .style(theme::scrollbar);
    container(body).padding([8, 4]).width(Fill).max_height(360).style(theme::code_block).into()
}

fn note_paths(context: &str) -> Vec<String> {
    let re = regex::Regex::new(r#"<note name="[^"]*" path="([^"]+)""#).unwrap();
    re.captures_iter(context).map(|c| c[1].to_string()).collect()
}

/// Renders code blocks with the app's colors and a Copy button.
pub struct Viewer;

impl<'a> markdown::Viewer<'a, Message> for Viewer {
    fn on_link_click(url: markdown::Uri) -> Message {
        Message::Link(url)
    }

    fn code_block(&self, settings: markdown::Settings, language: Option<&'a str>, code: &'a str, lines: &'a [markdown::Text]) -> Element<'a, Message> {
        let header = row![
            text(language.unwrap_or("text")).size(12).style(theme::muted),
            space::horizontal(),
            button(row![icon(Icon::Copy, 13.0), text("Copy").size(12)].spacing(4).align_y(Alignment::Center))
                .padding([2, 6])
                .style(theme::ghost)
                .on_press(Message::Copy(code.to_string())),
        ]
        .align_y(Alignment::Center);
        let body = Column::with_children(lines.iter().map(|l| {
            rich_text(l.spans(settings.style))
                .on_link_click(Self::on_link_click)
                .font(settings.style.code_block_font)
                .size(settings.code_size)
                .into()
        }));
        let scroll = scrollable(container(body).padding([4, 0]))
            .direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::new()))
            .style(theme::scrollbar);
        container(column![header, scroll].spacing(4)).padding([8, 12]).width(Fill).style(theme::code_block).into()
    }
}

/// Markdown settings in the app's fonts and colors.
pub fn markdown_settings(theme: &Theme) -> markdown::Settings {
    let t = tokens(theme);
    let mut style = markdown::Style::from_palette(theme.palette());
    style.font = fonts::ui();
    style.inline_code_font = fonts::mono();
    style.code_block_font = fonts::mono();
    style.inline_code_color = t.text;
    style.inline_code_highlight = markdown::Highlight {
        background: t.surface2.into(),
        border: iced::Border { color: t.line, width: 1.0, radius: 4.0.into() },
    };
    style.inline_code_padding = Padding::from([1, 4]);
    style.link_color = t.accent_ink;
    let mut settings = markdown::Settings::with_text_size(15, style);
    settings.code_size = 13.0.into();
    settings
}
