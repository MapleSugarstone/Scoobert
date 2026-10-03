use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::paths;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Thinking {
    Off,
    Low,
    #[default]
    Medium,
    High,
}

impl Thinking {
    pub const ALL: [Thinking; 4] = [Thinking::Off, Thinking::Low, Thinking::Medium, Thinking::High];

    /// Thinking tokens per level. A laptop CPU generates one to four tokens per second.
    pub fn budget(self) -> u32 {
        match self {
            Thinking::Off => 0,
            Thinking::Low => 256,
            Thinking::Medium => 1024,
            Thinking::High => 4096,
        }
    }

    pub fn label(self) -> &'static str {
        use crate::i18n::key;
        match self {
            Thinking::Off => key("Off"),
            Thinking::Low => key("Low"),
            Thinking::Medium => key("Medium"),
            Thinking::High => key("High"),
        }
    }
}

impl std::fmt::Display for Thinking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(crate::i18n::tr(self.label()))
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum Approvals {
    #[default]
    Ask,
    /// Edits inside the project run without asking, commands run in the sandbox, and anything else is refused.
    /// Nobody needs to watch, so this is the mode for leaving a task running.
    Project,
    /// Everything runs without asking.
    Auto,
}

impl Approvals {
    pub const ALL: [Approvals; 3] = [Approvals::Ask, Approvals::Project, Approvals::Auto];
}

impl std::fmt::Display for Approvals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(crate::i18n::tr(match self {
            Approvals::Ask => "Ask before changes",
            Approvals::Project => "Work unattended in the project",
            Approvals::Auto => "Allow everything",
        }))
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemeChoice {
    #[default]
    System,
    Light,
    Dark,
}

impl ThemeChoice {
    pub const ALL: [ThemeChoice; 3] = [ThemeChoice::System, ThemeChoice::Light, ThemeChoice::Dark];
}

impl std::fmt::Display for ThemeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(crate::i18n::tr(match self {
            ThemeChoice::System => "Match the system",
            ThemeChoice::Light => "Light",
            ThemeChoice::Dark => "Dark",
        }))
    }
}

/// A hosted model the user added to the model menu.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostedModel {
    pub provider: String,
    pub id: String,
    pub name: String,
    pub context: u32,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub reasoning: bool,
}

impl HostedModel {
    /// The name stored in settings and conversations: "provider/model id".
    pub fn reference(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

/// A hosted provider the user added by address, for any server that speaks the OpenAI chat format.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct CustomProvider {
    pub id: String,
    pub name: String,
    pub base_url: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Settings {
    pub model: String,
    pub context_sizes: BTreeMap<String, u32>,
    pub thinking: Thinking,
    pub approvals: Approvals,
    pub notes_folder: String,
    pub activity_log: bool,
    pub remember_step: bool,
    /// Offers the web_search and web_read tools.
    pub web_access: bool,
    pub keep_alive_minutes: u32,
    /// Loads the default model when Scoobert starts, instead of when the user starts typing.
    pub preload_model: bool,
    pub models_dir: String,
    pub llama_server_path: String,
    pub hosted_models: Vec<HostedModel>,
    pub custom_providers: Vec<CustomProvider>,
    pub theme: ThemeChoice,
    /// The interface and reply language code, or empty until the user or the installer picks one.
    pub language: String,
    /// Runs local models on the graphics card where they fit.
    pub use_gpu: bool,
    /// Models the graphics card failed to load, which run on the processor. Turning the card on again clears it.
    pub gpu_failed: Vec<String>,
    /// Lets a model larger than free memory run with part of its weights read from disk as virtual memory, slowly.
    pub models_from_disk: bool,
    /// What the first message of a new conversation in a project does before any code.
    pub plan_first: PlanFirst,
    /// System prompts the user wrote, by model name. A model without one uses Scoobert's own.
    pub model_prompts: BTreeMap<String, String>,
    /// Model files the user added from other folders, which Scoobert runs where they are.
    pub model_files: Vec<String>,
    /// Models that run with the direction they learned from ratings, with its strength.
    pub learning: BTreeMap<String, f32>,
    /// How each model predicts words ahead for itself to check: "mtp" for its own prediction layers, "ngram" for text
    /// already in the conversation, or "draft:<model>" for a smaller model that drafts. A model left out does not.
    pub speculation: BTreeMap<String, String>,
    /// Keeps the context at 8 bits instead of 16, which halves its memory.
    pub compact_context: bool,
}

/// Whether a new conversation's first task plans the project in its notes before it writes code.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum PlanFirst {
    #[default]
    Off,
    /// Writes the plan, then stops for the user to change it.
    Discuss,
    /// Writes the plan, then builds from it.
    Build,
}

impl PlanFirst {
    pub const ALL: [PlanFirst; 3] = [PlanFirst::Off, PlanFirst::Discuss, PlanFirst::Build];
}

impl std::fmt::Display for PlanFirst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(crate::i18n::tr(match self {
            PlanFirst::Off => "Build right away",
            PlanFirst::Discuss => "Plan, then discuss",
            PlanFirst::Build => "Plan, then build",
        }))
    }
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model: "Qwen3.8-27B-UD-IQ4_XS".into(),
            context_sizes: BTreeMap::new(),
            thinking: Thinking::Medium,
            approvals: Approvals::Ask,
            notes_folder: "Notes".into(),
            activity_log: true,
            remember_step: true,
            web_access: false,
            keep_alive_minutes: 30,
            preload_model: false,
            models_dir: String::new(),
            llama_server_path: String::new(),
            hosted_models: Vec::new(),
            custom_providers: Vec::new(),
            theme: ThemeChoice::System,
            language: String::new(),
            // Macs share memory between the processor and the graphics chip, so the model runs on the chip by default.
            use_gpu: cfg!(target_os = "macos"),
            gpu_failed: Vec::new(),
            models_from_disk: false,
            plan_first: PlanFirst::Off,
            model_prompts: BTreeMap::new(),
            model_files: Vec::new(),
            learning: BTreeMap::new(),
            speculation: BTreeMap::new(),
            compact_context: false,
        }
    }
}

impl Settings {
    /// The user's own system prompt for `model`, if they wrote one.
    pub fn model_prompt(&self, model: &str) -> Option<&str> {
        self.model_prompts.get(model).map(|p| p.as_str()).filter(|p| !p.trim().is_empty())
    }

    /// The language in use: the one picked, else the one chosen in the Windows installer, else the computer's own.
    pub fn language(&self) -> &'static crate::i18n::Language {
        let picked = Some(self.language.as_str()).filter(|c| !c.is_empty()).and_then(crate::i18n::find);
        picked.or_else(|| crate::sys::installer_language().as_deref().and_then(crate::i18n::find)).or_else(|| crate::i18n::system().and_then(crate::i18n::find)).unwrap_or(&crate::i18n::LANGUAGES[0])
    }

    pub fn models_dir(&self) -> PathBuf {
        if self.models_dir.trim().is_empty() {
            paths::get().default_models_dir()
        } else {
            PathBuf::from(self.models_dir.trim())
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Project {
    pub path: PathBuf,
    #[serde(default)]
    pub last_session: Option<PathBuf>,
}

impl Project {
    pub fn name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| paths::display(&self.path))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct State {
    pub settings: Settings,
    pub projects: Vec<Project>,
    pub current_project: Option<PathBuf>,
    /// The last conversation opened with no project selected.
    pub general_last_session: Option<PathBuf>,
    pub window: Option<(f32, f32)>,
    /// Stored as closed so the pane starts open for new users and for state saved before this field.
    pub notes_closed: bool,
    /// The user hid the sidebar. A narrow window hides it as well, without changing this.
    pub sidebar_closed: bool,
    pub update_checked_at: i64,
    /// The version whose saved prompts were built for the default model. A new version rebuilds them once.
    pub prepared_version: String,
    pub setup_done: bool,
}

impl State {
    pub fn load() -> State {
        let file = paths::get().state_file();
        match std::fs::read_to_string(&file) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|err| {
                eprintln!("[state] {} is unreadable ({err}); starting from defaults", file.display());
                let _ = std::fs::copy(&file, file.with_extension("json.bad"));
                State::default()
            }),
            Err(_) => State::from_picode().unwrap_or_default(),
        }
    }

    /// Carries over the project list and main settings from PiCode, the app Scoobert replaced, on first run.
    fn from_picode() -> Option<State> {
        // Test profiles and portable copies start fresh.
        if std::env::var_os("SCOOBERT_HOME").is_some() || paths::portable_folder().is_some() {
            return None;
        }
        let file = std::path::PathBuf::from(std::env::var_os("APPDATA")?).join("PiCode").join("state.json");
        let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(file).ok()?).ok()?;
        let mut state = State::default();
        for p in old["projects"].as_array()? {
            let path = PathBuf::from(p["path"].as_str()?);
            if path.is_dir() {
                state.projects.push(Project { path, last_session: None });
            }
        }
        let s = &old["settings"];
        if let Some(model) = s["model"].as_str() {
            state.settings.model = model.to_string();
        }
        if let Ok(thinking) = serde_json::from_value(s["thinking"].clone()) {
            state.settings.thinking = thinking;
        }
        if let Some(folder) = s["notesFolder"].as_str().filter(|f| !f.is_empty()) {
            state.settings.notes_folder = folder.to_string();
        }
        if let Some(minutes) = s["keepAliveMinutes"].as_u64() {
            state.settings.keep_alive_minutes = minutes as u32;
        }
        if let Some(dir) = s["modelsDir"].as_str() {
            state.settings.models_dir = dir.to_string();
        }
        if let Some(sizes) = s["contextSizes"].as_object() {
            for (k, v) in sizes {
                if let Some(n) = v.as_u64() {
                    state.settings.context_sizes.insert(k.clone(), n as u32);
                }
            }
        }
        state.current_project = old["lastProject"].as_str().map(PathBuf::from).filter(|p| state.project(p).is_some());
        state.setup_done = !state.projects.is_empty();
        Some(state)
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let file = paths::get().state_file();
        write_atomic(&file, serde_json::to_string_pretty(self)?.as_bytes())
    }

    pub fn project(&self, path: &Path) -> Option<&Project> {
        self.projects.iter().find(|p| p.path == path)
    }

    pub fn project_mut(&mut self, path: &Path) -> Option<&mut Project> {
        self.projects.iter_mut().find(|p| p.path == path)
    }
}

/// Writes through a temporary file so a crash never leaves a half-written file behind.
pub fn write_atomic(file: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, file)?;
    Ok(())
}
