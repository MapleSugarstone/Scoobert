//! Conversation messages and their storage: one JSON Lines file per conversation, in a folder per project.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::paths;
use crate::store::Thinking;
use crate::util::{now_millis, random_hex, sha256_hex};

const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Image {
    pub mime: String,
    /// Base64 without the data URL prefix.
    pub data: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct UserMessage {
    /// What the user typed.
    pub text: String,
    /// Notes and environment details Scoobert attached, sent after the text and hidden in the transcript.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub context: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    pub time: i64,
}

impl UserMessage {
    pub fn full_text(&self) -> String {
        if self.context.is_empty() { self.text.clone() } else { format!("{}\n\n{}", self.text, self.context) }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn arg(&self, key: &str) -> &str {
        self.arguments.get(key).and_then(Value::as_str).unwrap_or_default()
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    #[default]
    Stop,
    ToolUse,
    Length,
    Aborted,
    Error,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    #[serde(default)]
    pub cached: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct AssistantMessage {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    /// A provider's signature over the thinking text, which it requires back with the next request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub stop: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub model: String,
    pub time: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub output: String,
    #[serde(default)]
    pub is_error: bool,
    /// A unified diff of the change, for edits and writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    pub time: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolResult),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Entry {
    Header { version: u32, id: String, cwd: PathBuf, created: i64, model: String, thinking: Thinking },
    Message { message: Message },
    Title { title: String },
    Model { model: String },
    Thinking { thinking: Thinking },
    Compaction { summary: String, kept_from: usize },
}

/// A summary that replaces every message before `kept_from` when Scoobert sends the conversation.
#[derive(Clone, Debug, PartialEq)]
pub struct Compaction {
    pub summary: String,
    pub kept_from: usize,
}

#[derive(Clone, Debug)]
pub struct Conversation {
    pub id: String,
    pub cwd: PathBuf,
    pub file: PathBuf,
    pub created: i64,
    pub title: Option<String>,
    pub model: String,
    pub thinking: Thinking,
    pub messages: Vec<Message>,
    pub compaction: Option<Compaction>,
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub file: PathBuf,
    pub id: String,
    pub title: String,
    pub modified: i64,
    pub messages: usize,
}

impl Conversation {
    pub fn new(cwd: &Path, model: &str, thinking: Thinking) -> Self {
        let id = random_hex(8);
        let created = now_millis();
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let file = project_dir(cwd).join(format!("{stamp}-{id}.jsonl"));
        Conversation {
            id,
            cwd: cwd.to_path_buf(),
            file,
            created,
            title: None,
            model: model.to_string(),
            thinking,
            messages: Vec::new(),
            compaction: None,
        }
    }

    pub fn load(file: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(file).with_context(|| format!("Could not open {}", file.display()))?;
        let mut conv: Option<Conversation> = None;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            // A line cut short by a crash is skipped rather than losing the conversation.
            let Ok(entry) = serde_json::from_str::<Entry>(line) else { continue };
            match (entry, conv.as_mut()) {
                (Entry::Header { id, cwd, created, model, thinking, .. }, None) => {
                    conv = Some(Conversation {
                        id,
                        cwd,
                        file: file.to_path_buf(),
                        created,
                        title: None,
                        model,
                        thinking,
                        messages: Vec::new(),
                        compaction: None,
                    });
                }
                (Entry::Message { message }, Some(c)) => c.messages.push(message),
                (Entry::Title { title }, Some(c)) => c.title = Some(title),
                (Entry::Model { model }, Some(c)) => c.model = model,
                (Entry::Thinking { thinking }, Some(c)) => c.thinking = thinking,
                (Entry::Compaction { summary, kept_from }, Some(c)) => c.compaction = Some(Compaction { summary, kept_from }),
                _ => {}
            }
        }
        conv.with_context(|| format!("{} is not a Scoobert conversation", file.display()))
    }

    fn append(&self, entry: &Entry) -> anyhow::Result<()> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let is_new = !self.file.exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.file)?;
        if is_new {
            let header = Entry::Header {
                version: FORMAT_VERSION,
                id: self.id.clone(),
                cwd: self.cwd.clone(),
                created: self.created,
                model: self.model.clone(),
                thinking: self.thinking,
            };
            writeln!(f, "{}", serde_json::to_string(&header)?)?;
        }
        writeln!(f, "{}", serde_json::to_string(entry)?)?;
        Ok(())
    }

    /// Adds a message and writes it to disk. The file is created with the first message.
    pub fn push(&mut self, message: Message) -> anyhow::Result<()> {
        let entry = Entry::Message { message };
        let result = self.append(&entry);
        let Entry::Message { message } = entry else { unreachable!() };
        self.messages.push(message);
        result
    }

    pub fn set_title(&mut self, title: &str) -> anyhow::Result<()> {
        self.title = Some(title.to_string());
        if self.file.exists() { self.append(&Entry::Title { title: title.to_string() }) } else { Ok(()) }
    }

    pub fn set_model(&mut self, model: &str) -> anyhow::Result<()> {
        self.model = model.to_string();
        if self.file.exists() { self.append(&Entry::Model { model: model.to_string() }) } else { Ok(()) }
    }

    pub fn set_thinking(&mut self, thinking: Thinking) -> anyhow::Result<()> {
        self.thinking = thinking;
        if self.file.exists() { self.append(&Entry::Thinking { thinking }) } else { Ok(()) }
    }

    pub fn set_compaction(&mut self, summary: String, kept_from: usize) -> anyhow::Result<()> {
        self.append(&Entry::Compaction { summary: summary.clone(), kept_from })?;
        self.compaction = Some(Compaction { summary, kept_from });
        Ok(())
    }

    pub fn has_user_message(&self) -> bool {
        self.messages.iter().any(|m| matches!(m, Message::User(_)))
    }

    pub fn display_title(&self) -> String {
        self.title.clone().unwrap_or_else(|| first_user_text(&self.messages).unwrap_or_else(|| "New conversation".into()))
    }

    /// Tokens in the context after the last reply, from the provider's own count.
    pub fn context_tokens(&self) -> u64 {
        self.messages
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::Assistant(a) => a.usage.map(|u| u.input + u.output),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Lists a project's conversations, newest first.
    pub fn list(cwd: &Path) -> Vec<Summary> {
        let dir = project_dir(cwd);
        let mut out: Vec<Summary> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".jsonl"))
            .filter_map(|e| {
                let conv = Conversation::load(&e.path()).ok()?;
                let modified = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(conv.created);
                Some(Summary {
                    file: e.path(),
                    id: conv.id.clone(),
                    title: conv.display_title(),
                    modified,
                    messages: conv.messages.len(),
                })
            })
            .collect();
        out.sort_by(|a, b| b.modified.cmp(&a.modified));
        out
    }

    /// Moves a conversation file to the system trash.
    pub fn delete(file: &Path) -> anyhow::Result<()> {
        trash::delete(file).map_err(|err| anyhow::anyhow!("Could not delete the conversation: {err}"))
    }
}

pub fn first_user_text(messages: &[Message]) -> Option<String> {
    messages.iter().find_map(|m| match m {
        Message::User(u) => {
            let flat = u.text.split_whitespace().collect::<Vec<_>>().join(" ");
            Some(crate::util::clip(&flat, 200))
        }
        _ => None,
    })
}

/// A project's conversation folder: its name plus a hash of its full path, so two projects with the same
/// folder name stay apart.
pub fn project_dir(cwd: &Path) -> PathBuf {
    let key = cwd.to_string_lossy();
    let key = if cfg!(windows) { key.to_lowercase() } else { key.into_owned() };
    let name: String = cwd
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".into())
        .chars()
        .map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' })
        .take(40)
        .collect();
    paths::get().sessions().join(format!("{name}-{}", &sha256_hex(&key)[..10]))
}

/// Checks that a file is a conversation inside Scoobert's own sessions folder.
pub fn is_session_file(file: &Path) -> bool {
    let root = paths::get().sessions();
    file.extension().is_some_and(|e| e == "jsonl")
        && match (file.canonicalize(), root.canonicalize()) {
            (Ok(f), Ok(r)) => f.starts_with(r),
            _ => false,
        }
}
