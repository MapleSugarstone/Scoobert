//! The tools the model can call: read, edit, and write files, and run shell commands.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

use super::conversation::ToolCall;
use super::sandbox::{self, Isolation};
use crate::paths;

const MAX_LINES: usize = 2000;
const MAX_BYTES: usize = 50 * 1024;
const DEFAULT_TIMEOUT_SECS: u64 = 600;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shell {
    Bash(PathBuf),
    /// Inside a Flatpak sandbox commands run on the host through `flatpak-spawn`.
    FlatpakHost,
    PowerShell,
}

impl Shell {
    pub fn detect() -> Shell {
        if cfg!(windows) {
            return find_git_bash().map(Shell::Bash).unwrap_or(Shell::PowerShell);
        }
        if Path::new("/.flatpak-info").exists() {
            return Shell::FlatpakHost;
        }
        let bash = ["/bin/bash", "/usr/bin/bash"].into_iter().map(PathBuf::from).find(|p| p.is_file());
        Shell::Bash(bash.unwrap_or_else(|| PathBuf::from("/bin/sh")))
    }

    pub fn tool_name(&self) -> &'static str {
        match self {
            Shell::PowerShell => "powershell",
            _ => "bash",
        }
    }

    /// One line for the environment block that tells the model which shell its commands run in.
    pub fn describe(&self) -> &'static str {
        match self {
            Shell::Bash(_) if cfg!(windows) => "Windows. The bash tool runs Git Bash.",
            Shell::PowerShell => "Windows. Run shell commands with the powershell tool.",
            _ if cfg!(target_os = "macos") => "macOS. The bash tool runs bash.",
            _ => "Linux. The bash tool runs bash.",
        }
    }
}

fn find_git_bash() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let candidates = [
        env("ProgramFiles").map(|p| p.join("Git/bin/bash.exe")),
        env("ProgramFiles(x86)").map(|p| p.join("Git/bin/bash.exe")),
        env("LOCALAPPDATA").map(|p| p.join("Programs/Git/bin/bash.exe")),
    ];
    if let Some(found) = candidates.into_iter().flatten().find(|p| p.is_file()) {
        return Some(found);
    }
    let path = std::env::var_os("PATH")?;
    // System32 holds the launcher named bash.exe that runs Linux instead of Windows commands.
    std::env::split_paths(&path)
        .filter(|d| !d.to_string_lossy().to_lowercase().contains("\\windows\\system32"))
        .map(|d| d.join("bash.exe"))
        .find(|p| p.is_file())
}

/// Tool definitions in the OpenAI function format. A conversation without a project also gets new_project, and a
/// computer with a Chromium-based browser gets the browser tools.
pub fn specs(shell: &Shell, no_project: bool, web: bool, browser: bool) -> Vec<Value> {
    let path = json!({ "type": "string", "description": "File path, relative to the project folder or absolute." });
    let tool = |name: &str, description: &str, properties: Value, required: &[&str]| {
        json!({ "type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": { "type": "object", "properties": properties, "required": required },
        }})
    };
    let port = if *shell == Shell::PowerShell { "$env:PORT" } else { "$PORT" };
    let shell_params = json!({
        "command": { "type": "string" },
        "timeout": { "type": "integer", "description": "Seconds." },
        "background": {
            "type": "boolean",
            "description": format!("Set to true for a server or another command that keeps running, such as npm run dev. Scoobert sets PORT to a free port for it, so start a server on {port} and 127.0.0.1, as in python -m http.server {port} --bind 127.0.0.1."),
        },
    });
    let shell_tool = match shell {
        Shell::PowerShell => tool(
            "powershell",
            "Run a PowerShell command in the project folder and return its output. Commands stop after 10 minutes unless you set timeout. Set background to true to start a server or watcher without waiting for it to end.",
            shell_params,
            &["command"],
        ),
        _ => tool(
            "bash",
            "Run a bash command in the project folder and return its output. Commands stop after 10 minutes unless you set timeout. Set background to true to start a server or watcher without waiting for it to end.",
            shell_params,
            &["command"],
        ),
    };
    let mut all = vec![
        tool(
            "read",
            "Read a text file. Returns up to 2000 lines. Use offset and limit for longer files. Read a note by its name in double brackets, such as [[Auth design]]; the result also lists the note's links and the notes that link to it.",
            json!({
                "path": path,
                "offset": { "type": "integer", "description": "First line to read, starting at 1." },
                "limit": { "type": "integer", "description": "Number of lines to read." },
            }),
            &["path"],
        ),
        tool(
            "edit",
            "Replace an exact piece of text in a file. old_text must match the file exactly, including indentation, and appear only once. Include nearby lines to make it unique, or set replace_all to true to replace every place it appears.",
            json!({ "path": path, "old_text": { "type": "string" }, "new_text": { "type": "string" }, "replace_all": { "type": "boolean" } }),
            &["path", "old_text", "new_text"],
        ),
        tool(
            "write",
            "Create a file or replace a whole file. Creates missing folders. Set append to true to add content to the end of an existing file, which is how to write a long file in parts. Give path and append before content.",
            json!({ "path": path, "append": { "type": "boolean" }, "content": { "type": "string" } }),
            &["path", "content"],
        ),
        shell_tool,
        tool(
            "job_output",
            "Read the new output of a command started with background set to true, and whether it still runs. Set wait to wait up to that many seconds for output to arrive.",
            json!({ "id": { "type": "integer" }, "wait": { "type": "integer", "description": "Seconds, up to 60." } }),
            &["id"],
        ),
        tool("job_stop", "Stop a command started with background set to true.", json!({ "id": { "type": "integer" } }), &["id"]),
    ];
    if browser {
        let element = json!({ "type": "integer", "description": "The number of a control, from the list browser_open and browser_read return." });
        all.push(tool(
            "browser_open",
            "Open a page in a browser to test it, such as http://localhost:3000 or an HTML file in the project. Returns the page's text, its controls numbered for browser_click and browser_type, and console errors. Start a server first with the shell tool and background set to true.",
            json!({ "url": { "type": "string" } }),
            &["url"],
        ));
        all.push(tool(
            "browser_read",
            "Read the open page again: its text, its numbered controls, and new console messages. Give offset to read on in a long page.",
            json!({ "offset": { "type": "integer", "description": "Character position to read from." } }),
            &[],
        ));
        all.push(tool(
            "browser_click",
            "Click a control on the open page by its number, or a point by x and y in pixels from the page's top left, such as a spot on a game's canvas.",
            json!({ "element": element, "x": { "type": "number" }, "y": { "type": "number" } }),
            &[],
        ));
        all.push(tool(
            "browser_type",
            "Type text on the open page, into the numbered control or wherever the focus is. Set submit to true to press Enter after it.",
            json!({ "text": { "type": "string" }, "element": element, "submit": { "type": "boolean" } }),
            &["text"],
        ));
        all.push(tool(
            "browser_key",
            "Press a key on the open page, such as Enter, Escape, Space, ArrowLeft, or a letter, with modifiers as in Control+a. For a game, set hold_ms to hold the key down and times to press it more than once.",
            json!({ "key": { "type": "string" }, "hold_ms": { "type": "integer" }, "times": { "type": "integer" } }),
            &["key"],
        ));
        all.push(tool(
            "browser_script",
            "Run JavaScript in the open page and return its value, for example to read a game's state. The code can use await.",
            json!({ "code": { "type": "string" } }),
            &["code"],
        ));
        all.push(tool(
            "browser_screenshot",
            "Take a screenshot of the open page to check how it looks. Give question to say what to look for.",
            json!({ "question": { "type": "string" } }),
            &[],
        ));
    }
    if web {
        all.push(tool(
            "web_search",
            "Search the web. Returns titles, addresses, and snippets. Use it only when the user asks you to look something up or research online.",
            json!({ "query": { "type": "string" } }),
            &["query"],
        ));
        all.push(tool(
            "web_read",
            "Read a web page as plain text. Give find to get only the passages about a topic, or offset to read on from a position.",
            json!({
                "url": { "type": "string" },
                "find": { "type": "string", "description": "Words to look for on the page." },
                "offset": { "type": "integer", "description": "Character position to read from." },
            }),
            &["url"],
        ));
    }
    if no_project {
        all.push(tool(
            "new_project",
            "Start a project folder for work that needs its own files, such as a program or a website. Relative paths then resolve inside it. Call it once, before creating the files.",
            json!({ "name": { "type": "string", "description": "A short folder name, such as snake-game." } }),
            &["name"],
        ));
    }
    all
}

/// Tools that need no approval: they only read, or act only inside the throwaway browser, or on the model's own
/// background commands. browser_script needs approval, since a script can do anything the page can.
pub fn is_read_only(name: &str) -> bool {
    matches!(
        name,
        "read" | "web_search" | "web_read" | "job_output" | "job_stop" | "browser_open" | "browser_read" | "browser_click" | "browser_type" | "browser_key" | "browser_screenshot"
    )
}

pub fn changes_files(name: &str) -> bool {
    name == "edit" || name == "write"
}

#[derive(Debug, Clone, Default)]
pub struct Outcome {
    pub output: String,
    pub is_error: bool,
    pub diff: Option<String>,
    /// A screenshot, which a model that can see images gets with the output and another model describes otherwise.
    pub image: Option<super::conversation::Image>,
}

impl Outcome {
    fn ok(output: impl Into<String>) -> Self {
        Outcome { output: output.into(), ..Default::default() }
    }

    fn err(output: impl Into<String>) -> Self {
        Outcome { output: output.into(), is_error: true, ..Default::default() }
    }
}

/// Looks up an argument under the names different models use for it.
fn arg<'a>(call: &'a ToolCall, names: &[&str]) -> Option<&'a str> {
    names.iter().find_map(|n| call.arguments.get(*n).and_then(Value::as_str))
}

fn arg_u64(call: &ToolCall, name: &str) -> Option<u64> {
    let v = call.arguments.get(name)?;
    v.as_u64().or_else(|| v.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)).or_else(|| v.as_str().and_then(|s| s.trim().trim_matches(['[', ']']).parse().ok()))
}

fn arg_f64(call: &ToolCall, name: &str) -> Option<f64> {
    let v = call.arguments.get(name)?;
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

fn arg_bool(call: &ToolCall, name: &str) -> bool {
    call.arguments.get(name).is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"))
}

pub fn resolve(cwd: &Path, p: &str) -> PathBuf {
    let p = p.trim();
    if let Some(rest) = p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\")) {
        return paths::get().home.join(rest);
    }
    let path = Path::new(p);
    if path.is_absolute() { path.to_path_buf() } else { cwd.join(path) }
}

/// The file a tool call names. A note can be named as [[Note name]], which resolves the way links do.
pub fn target_path(cwd: &Path, call: &ToolCall, notes: Option<&Path>) -> Option<PathBuf> {
    let raw = arg(call, &["path", "file_path", "filePath", "file", "note"])?;
    if let (Some(notes), Some(name)) = (notes, note_link(raw)) {
        let vault = crate::notes::Vault::new(notes);
        let files: Vec<String> = vault.files().into_iter().filter(|f| !f.is_dir).map(|f| f.path).collect();
        let rel = crate::notes::links::resolve_link(&name, &files).cloned().unwrap_or_else(|| format!("{}.md", crate::notes::sanitize_name(&name)));
        return Some(notes.join(rel));
    }
    Some(resolve(cwd, raw))
}

/// The target of a path written as a wikilink, such as [[Auth design]] or [[Auth design#Tokens|tokens]].
fn note_link(raw: &str) -> Option<String> {
    let inner = raw.trim().strip_prefix("[[")?.strip_suffix("]]")?;
    let target = crate::notes::links::parse_link(inner).target;
    (!target.is_empty()).then_some(target)
}

/// The conversation's browser, which starts with the first browser_open and closes when the task ends.
pub type BrowserSlot = Arc<tokio::sync::Mutex<Option<super::browser::Browser>>>;

/// Restrictions on commands for the current approval mode.
#[derive(Clone)]
pub struct Limits {
    pub unattended: bool,
    pub isolation: Isolation,
    /// Output size that fits comfortably in the model's context.
    pub max_output: usize,
    /// The project's notes folder, for reading notes by name and listing their links.
    pub notes: Option<PathBuf>,
    /// Whether the user turned on web search.
    pub web: bool,
    /// Where each file's previous version is kept before a change, so a rewind can undo it.
    pub checkpoints: Option<PathBuf>,
    pub jobs: Option<Arc<super::jobs::Jobs>>,
    pub browser: Option<BrowserSlot>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { unattended: false, isolation: Isolation::Refuse, max_output: MAX_BYTES, notes: None, web: false, checkpoints: None, jobs: None, browser: None }
    }
}

pub async fn run(
    call: &ToolCall,
    cwd: &Path,
    shell: &Shell,
    limits: &Limits,
    cancel: &CancellationToken,
    on_output: impl Fn(String) + Send + Sync + 'static,
) -> Outcome {
    if call.arguments.is_string() {
        return Outcome::err("The arguments were not valid JSON. Call the tool again with a JSON object.");
    }
    let result = match call.name.as_str() {
        "read" => read(call, cwd, limits).await,
        "edit" => edit(call, cwd, limits).await,
        "write" => write(call, cwd, limits).await,
        "bash" | "powershell" => command(call, cwd, shell, limits, cancel, on_output).await,
        "web_search" if limits.web => match super::web::search(arg(call, &["query", "q"]).unwrap_or_default()).await {
            Ok(results) => Ok(Outcome::ok(super::web::format_results(arg(call, &["query", "q"]).unwrap_or_default(), &results))),
            Err(e) => Err(format!("{e:#}")),
        },
        "web_read" if limits.web => match super::web::fetch(arg(call, &["url", "address", "link"]).unwrap_or_default()).await {
            Ok(page) => {
                let offset = arg_u64(call, "offset").unwrap_or(0) as usize;
                Ok(Outcome::ok(super::web::format_page(&page, offset, arg(call, &["find", "search"]), limits.max_output)))
            }
            Err(e) => Err(format!("{e:#}")),
        },
        "web_search" | "web_read" => Err("Web search is turned off. Tell the user they can turn it on under Settings, then continue without it.".into()),
        "job_output" | "job_stop" => job_tool(call, limits).await,
        name if name.starts_with("browser_") => browser_tool(call, cwd, limits).await,
        other => Err(format!("There is no tool named {other}. Use read, edit, write, or {}.", shell.tool_name())),
    };
    result.unwrap_or_else(Outcome::err)
}

async fn job_tool(call: &ToolCall, limits: &Limits) -> Result<Outcome, String> {
    let jobs = limits.jobs.as_ref().ok_or("Background commands are not available here.")?;
    let Some(id) = arg_u64(call, "id").or_else(|| arg_u64(call, "job")) else {
        return Err(format!("{} needs id, the number of a background job.\n{}", call.name, jobs.describe()));
    };
    let id = id as u32;
    if call.name == "job_stop" {
        return jobs.stop(id, limits.max_output).map(Outcome::ok);
    }
    let wait = Duration::from_secs(arg_u64(call, "wait").unwrap_or(0).min(60));
    jobs.output(id, wait, limits.max_output).await.map(Outcome::ok)
}

async fn browser_tool(call: &ToolCall, cwd: &Path, limits: &Limits) -> Result<Outcome, String> {
    use super::browser::{Browser, resolve_address};
    let slot = limits.browser.as_ref().ok_or("The browser tools are not available here.")?;
    // The address is checked before a browser starts for it.
    let address = if call.name == "browser_open" {
        Some(resolve_address(arg(call, &["url", "address", "path"]).unwrap_or_default(), cwd, limits.web).map_err(|e| format!("{e:#}"))?)
    } else {
        None
    };
    let mut guard = slot.lock().await;
    if guard.is_none() {
        if address.is_none() {
            return Err("No page is open. Open one with browser_open first.".into());
        }
        *guard = Some(Browser::launch(limits.web).await.map_err(|e| format!("{e:#}"))?);
    }
    let Some(browser) = guard.as_mut() else { return Err("The browser did not start.".into()) };
    let element = arg_u64(call, "element").or_else(|| arg_u64(call, "ref"));
    let result = match call.name.as_str() {
        "browser_open" => browser.open(address.as_deref().unwrap_or("about:blank")).await.map(Outcome::ok),
        "browser_read" => browser.read(arg_u64(call, "offset").unwrap_or(0) as usize).await.map(Outcome::ok),
        "browser_click" => {
            let point = arg_f64(call, "x").zip(arg_f64(call, "y"));
            browser.click(element, point).await.map(Outcome::ok)
        }
        "browser_type" => browser.type_text(arg(call, &["text"]).unwrap_or_default(), element, arg_bool(call, "submit")).await.map(Outcome::ok),
        "browser_key" => {
            let times = arg_u64(call, "times").unwrap_or(1).min(50) as u32;
            browser.keys(arg(call, &["key", "keys"]).unwrap_or_default(), arg_u64(call, "hold_ms").unwrap_or(0), times).await.map(Outcome::ok)
        }
        "browser_script" => browser.script(arg(call, &["code", "script", "expression"]).unwrap_or_default()).await.map(Outcome::ok),
        "browser_screenshot" => browser.screenshot().await.map(|(image, line)| Outcome { output: line, image: Some(image), ..Default::default() }),
        other => return Err(format!("There is no tool named {other}.")),
    };
    // A browser that closed or stopped answering starts again with the next browser_open.
    if result.as_ref().is_err_and(|e| e.to_string().starts_with("The browser closed")) {
        *guard = None;
    }
    result.map_err(|e| format!("{e:#}"))
}

async fn read(call: &ToolCall, cwd: &Path, limits: &Limits) -> Result<Outcome, String> {
    let max_bytes = limits.max_output;
    if let Some(id) = arg(call, &["path", "file"]).and_then(|p| p.trim().strip_prefix("conversation:")) {
        return read_conversation(cwd, id.trim(), max_bytes);
    }
    let path = target_path(cwd, call, limits.notes.as_deref()).ok_or("read needs a path.")?;
    let bytes = tokio::fs::read(&path).await.map_err(|e| format!("Could not read {}: {e}", paths::display(&path)))?;
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return Err(format!("{} is a binary file.", paths::display(&path)));
    }
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let start = arg_u64(call, "offset").unwrap_or(1).max(1) as usize - 1;
    if start >= lines.len() && !lines.is_empty() {
        return Err(format!("The file has {} lines, so offset {} is past its end.", lines.len(), start + 1));
    }
    let limit = arg_u64(call, "limit").map(|l| l as usize).unwrap_or(MAX_LINES).min(MAX_LINES);
    let mut out = String::new();
    let mut end = start;
    for line in lines.iter().skip(start).take(limit) {
        if out.len() + line.len() + 1 > max_bytes {
            break;
        }
        out.push_str(line);
        out.push('\n');
        end += 1;
    }
    if end < lines.len() {
        out.push_str(&format!("\n[Showing lines {}-{end} of {}. Use offset={} to read more.]", start + 1, lines.len(), end + 1));
    }
    if out.is_empty() {
        out = "(The file is empty.)".into();
    }
    if let Some(notes) = limits.notes.as_deref()
        && let Ok(rel) = path.strip_prefix(notes)
        && path.extension().is_some_and(|e| e.eq_ignore_ascii_case("md"))
    {
        let rel = rel.to_string_lossy().replace('\\', "/");
        let folder = notes.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let links = super::prompt::link_summary(&crate::notes::Vault::new(notes), &rel, &folder);
        if !links.is_empty() {
            out.push_str(&format!("\n{links}"));
        }
    }
    Ok(Outcome::ok(out))
}

/// Another conversation in this project, which the user asked about, as a compact transcript.
fn read_conversation(cwd: &Path, id: &str, max_bytes: usize) -> Result<Outcome, String> {
    let found = super::conversation::Conversation::list(cwd).into_iter().find(|s| !id.is_empty() && s.id.starts_with(id));
    let summary = found.ok_or_else(|| format!("There is no conversation {id} in this project."))?;
    let conv = super::conversation::Conversation::load(&summary.file).map_err(|e| format!("{e:#}"))?;
    Ok(Outcome::ok(conv.transcript(max_bytes)))
}

async fn write(call: &ToolCall, cwd: &Path, limits: &Limits) -> Result<Outcome, String> {
    let path = target_path(cwd, call, limits.notes.as_deref()).ok_or("write needs a path.")?;
    save_checkpoint(limits.checkpoints.as_deref(), &call.id, &path).await;
    let content = arg(call, &["content", "text", "contents"]).ok_or("write needs content.")?;
    // The conversation shows earlier writes as a short note, and a model can copy that note as a file's content.
    if SAVED_WRITE_NOTE.is_match(content.trim()) {
        return Err(format!(
            "Nothing was written, because content held a note about an earlier write instead of file text. The write tool works, and every earlier write is complete on disk. Put the full text you want in {} in content.",
            arg(call, &["path"]).unwrap_or_default()
        ));
    }
    let old = tokio::fs::read_to_string(&path).await.ok();
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| format!("Could not create {}: {e}", paths::display(dir)))?;
    }
    let append = call.arguments.get("append").is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"));
    if append && content.is_empty() {
        return Err("Nothing was added, because content was empty. Put the text to add in content.".into());
    }
    // A model writing a long file in parts sometimes leaves out append, and each part then replaces the whole file.
    let wipes = if append { None } else { old.as_deref().and_then(|existing| drops_most(existing, content)) };
    if !confirmed_replace(&path, content, wipes.is_some())
        && let Some((lines, kept)) = wipes
    {
        return Err(format!(
            "Nothing was written. {} has {lines} lines, and this write keeps only {kept} of them, so it would delete the rest. To add this text to the end of the file, call write again with append set to true. To change part of the file, use edit. If you do mean to replace the whole file with this text, send the same write again.",
            paths::display(&path)
        ));
    }
    let new = match (&old, append) {
        (Some(existing), true) if !existing.is_empty() && !existing.ends_with('\n') => format!("{existing}\n{content}"),
        (Some(existing), true) => format!("{existing}{content}"),
        _ => content.to_string(),
    };
    tokio::fs::write(&path, &new).await.map_err(|e| format!("Could not write {}: {e}", paths::display(&path)))?;
    let shown = paths::display(&path);
    let verb = match (&old, append) {
        (Some(_), true) => "Added to",
        (Some(_), false) => "Replaced",
        (None, _) => "Created",
    };
    let mut output = format!("{verb} {shown} (now {} bytes).", new.len());
    // The note that replaces long content in the conversation reads like a failed write unless it is explained.
    if content.len() > SHORTENED_WRITE {
        output.push_str(SHORTENED_NOTICE);
    }
    // After a part is added, the file's top-level lines show what earlier parts already declared.
    if append {
        let lines = outline(&new, path.extension().is_some_and(|e| e.eq_ignore_ascii_case("md")));
        if !lines.is_empty() {
            output.push_str("\nTop-level lines in the file now, without the lines inside classes and functions:");
            for line in lines.iter().take(OUTLINE_LINES) {
                output.push_str(&format!("\n{line}"));
            }
            if lines.len() > OUTLINE_LINES {
                output.push_str(&format!("\n[{} more]", lines.len() - OUTLINE_LINES));
            }
        }
    }
    Ok(Outcome { output, diff: Some(unified_diff(old.as_deref().unwrap_or(""), &new)), ..Default::default() })
}

const OUTLINE_LINES: usize = 40;

/// The existing file's line count and how many of its lines `new` keeps, when a file of ten or more lines that say
/// something would lose more than half of them. Short lines such as a closing brace match by chance, so they do not
/// count.
fn drops_most(old: &str, new: &str) -> Option<(usize, usize)> {
    let lines: Vec<&str> = old.lines().map(str::trim).filter(|l| l.len() > 2).collect();
    if lines.len() < 10 {
        return None;
    }
    let have: std::collections::HashSet<&str> = new.lines().map(str::trim).collect();
    let kept = lines.iter().filter(|l| have.contains(*l)).count();
    (kept * 2 < lines.len()).then_some((old.lines().count(), kept))
}

/// Whether a write that `wipes` most of `path` repeats the one refused just before it, which means the model meant to
/// replace the file. It remembers this write's refusal, and any other write to the file forgets it.
fn confirmed_replace(path: &Path, content: &str, wipes: bool) -> bool {
    use std::hash::{Hash, Hasher};
    static REFUSED: Mutex<Option<std::collections::HashMap<PathBuf, u64>>> = Mutex::new(None);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    let hash = hasher.finish();
    let mut refused = REFUSED.lock().unwrap();
    let refused = refused.get_or_insert_with(std::collections::HashMap::new);
    let repeat = refused.remove(path) == Some(hash);
    if wipes && !repeat {
        refused.insert(path.to_path_buf(), hash);
    }
    repeat
}

/// Writes longer than this show as a short note in later requests.
pub const SHORTENED_WRITE: usize = 1500;

/// Ends the result of a long write. `shorten_saved_writes` looks for it, so writes made before it existed keep the
/// older, shorter note and their conversations keep their saved caches.
pub const SHORTENED_NOTICE: &str = " From here on the conversation shows this call by its first and last lines, to save room. The file holds all of it.";

/// Whether a command writes a file from a heredoc or a PowerShell here-string. Only the line that opens a heredoc
/// is checked for a redirect, since the text inside it is often code with `>` in it.
fn writes_heredoc(command: &str) -> bool {
    use std::sync::LazyLock;
    static HEREDOC: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r#"<<-?\s*['"]?[A-Za-z_]\w*['"]?"#).unwrap());
    static REDIRECT: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r#"(?:^|[^0-9&>=-])>{1,2}\s*['"]?[\w./~$]|\btee\b"#).unwrap());
    static HERE_STRING: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r#"@['"]\s*$"#).unwrap());
    static PS_WRITE: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r#"(?i)\b(?:Set-Content|Add-Content|Out-File|WriteAllText)\b"#).unwrap());
    let bash = command.lines().any(|l| HEREDOC.is_match(l) && REDIRECT.is_match(&HEREDOC.replace_all(l, " ")));
    let powershell = command.lines().any(|l| HERE_STRING.is_match(l)) && PS_WRITE.is_match(command);
    bash || powershell
}

/// The notes `shorten_saved_writes` put in place of an earlier write's content before 0.2.6, and the
/// `content_saved` texts it has used since, any of which a model may copy into content.
static SAVED_WRITE_NOTE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^\[\d+ characters(?: written to .+\. Read the file to see them\.|, now in .+\.)\]$|^\d+ characters, saved in full in \S+$|^\d+ lines saved in full in .+, shown here by")
        .unwrap()
});

/// The lines at a file's left margin, which name what it imports and declares, or a Markdown file's headings.
fn outline(text: &str, markdown: bool) -> Vec<String> {
    text.lines()
        .map(str::trim_end)
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with(char::is_whitespace)
                && !["//", "/*", "*", "--", ";"].iter().any(|p| l.starts_with(p))
                && markdown == l.starts_with('#')
                && !l.chars().all(|c| "{}[]();,".contains(c))
        })
        .map(|l| crate::util::clip(l, 100))
        .collect()
}

async fn edit(call: &ToolCall, cwd: &Path, limits: &Limits) -> Result<Outcome, String> {
    let path = target_path(cwd, call, limits.notes.as_deref()).ok_or("edit needs a path.")?;
    save_checkpoint(limits.checkpoints.as_deref(), &call.id, &path).await;
    let old_text = arg(call, &["old_text", "oldText", "old_string", "old_str"]).ok_or("edit needs old_text.")?;
    let new_text = arg(call, &["new_text", "newText", "new_string", "new_str"]).ok_or("edit needs new_text.")?;
    let all = ["replace_all", "replaceAll"].iter().any(|k| call.arguments.get(*k).is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true")));
    let shown = paths::display(&path);
    if old_text.is_empty() {
        return Err("old_text is empty. Use write to create a file.".into());
    }
    let original = tokio::fs::read_to_string(&path).await.map_err(|e| format!("Could not read {shown}: {e}"))?;
    // Models send \n line endings; a file saved with \r\n is matched and written back with \r\n.
    let crlf = original.contains("\r\n");
    let text = if crlf { original.replace("\r\n", "\n") } else { original.clone() };
    let old_n = old_text.replace("\r\n", "\n");
    let new_n = new_text.replace("\r\n", "\n");
    let count = text.matches(&old_n).count();
    if count == 0 {
        return Err(format!("old_text was not found in {shown}. Read the file again and copy the text exactly, including indentation."));
    }
    if count > 1 && !all {
        return Err(format!(
            "old_text appears {count} times in {shown}. Include more surrounding lines so it matches once, or set replace_all to true to replace all {count}."
        ));
    }
    let updated = if all { text.replace(&old_n, &new_n) } else { text.replacen(&old_n, &new_n, 1) };
    let written = if crlf { updated.replace('\n', "\r\n") } else { updated.clone() };
    tokio::fs::write(&path, &written).await.map_err(|e| format!("Could not write {shown}: {e}"))?;
    let output = if count > 1 { format!("Edited {shown} in {count} places.") } else { format!("Edited {shown}.") };
    Ok(Outcome { output, diff: Some(unified_diff(&text, &updated)), ..Default::default() })
}

/// A file's state before a tool call changed it: its bytes, or that it did not exist.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub path: PathBuf,
    pub existed: bool,
}

/// Saves the file as it is now, before the call changes it.
async fn save_checkpoint(dir: Option<&Path>, call_id: &str, path: &Path) {
    let Some(dir) = dir else { return };
    let meta = dir.join(format!("{call_id}.json"));
    if meta.exists() || tokio::fs::create_dir_all(dir).await.is_err() {
        return;
    }
    let existed = path.is_file();
    if existed && tokio::fs::copy(path, dir.join(format!("{call_id}.bin"))).await.is_err() {
        return;
    }
    let record = Checkpoint { path: path.to_path_buf(), existed };
    if let Ok(json) = serde_json::to_string(&record) {
        let _ = tokio::fs::write(meta, json).await;
    }
}

/// Puts a file back the way a checkpoint saved it: its old bytes, or removed when it did not exist before.
pub fn restore_checkpoint(dir: &Path, call_id: &str) -> Option<Result<PathBuf, String>> {
    let meta = std::fs::read_to_string(dir.join(format!("{call_id}.json"))).ok()?;
    let record: Checkpoint = serde_json::from_str(&meta).ok()?;
    let shown = paths::display(&record.path);
    let result = if record.existed {
        std::fs::copy(dir.join(format!("{call_id}.bin")), &record.path).map(|_| ()).map_err(|e| format!("Could not restore {shown}: {e}"))
    } else if record.path.exists() {
        trash::delete(&record.path).map_err(|e| format!("Could not remove {shown}: {e}"))
    } else {
        Ok(())
    };
    Some(result.map(|_| record.path))
}

/// The file a checkpoint belongs to, without restoring it.
pub fn checkpoint_path(dir: &Path, call_id: &str) -> Option<PathBuf> {
    let meta = std::fs::read_to_string(dir.join(format!("{call_id}.json"))).ok()?;
    serde_json::from_str::<Checkpoint>(&meta).ok().map(|c| c.path)
}

pub fn unified_diff(old: &str, new: &str) -> String {
    similar::TextDiff::from_lines(old, new).unified_diff().context_radius(3).to_string()
}

async fn command(
    call: &ToolCall,
    cwd: &Path,
    shell: &Shell,
    limits: &Limits,
    cancel: &CancellationToken,
    on_output: impl Fn(String) + Send + Sync + 'static,
) -> Result<Outcome, String> {
    let cmd_text = arg(call, &["command", "cmd", "script"]).ok_or("The shell tool needs a command.")?;
    let timeout = Duration::from_secs(arg_u64(call, "timeout").unwrap_or(DEFAULT_TIMEOUT_SECS).clamp(1, 24 * 3600));
    let (mut cmd, cap) = shell_process(cmd_text, cwd, shell, limits)?;
    if arg_bool(call, "background") {
        let jobs = limits.jobs.as_ref().ok_or("Background commands are not available here.")?;
        return match jobs.start(cmd, cmd_text, cap, limits.max_output).await {
            Ok(output) => Ok(Outcome::ok(output)),
            Err(output) => Ok(Outcome::err(output)),
        };
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let child = cmd.spawn().map_err(|e| format!("Could not start the shell: {e}"))?;
    let pid = child.id();
    // The job caps memory for the command and everything it starts, and ends them all when it is dropped.
    #[cfg(windows)]
    let _job = sandbox::Job::new(cap).filter(|job| pid.is_some_and(|p| job.assign(p)));
    #[cfg(unix)]
    let _ = cap;
    run_to_end(child, pid, timeout, limits, cancel, on_output).await
}

/// The process for a shell command, inside the sandbox when working unattended, and on Linux in a scope that caps
/// its memory. The memory cap goes with it for a Windows job.
fn shell_process(cmd_text: &str, cwd: &Path, shell: &Shell, limits: &Limits) -> Result<(tokio::process::Command, u64), String> {
    if writes_heredoc(cmd_text) {
        return Err("Nothing ran. This command writes a file from a heredoc, which can change quotes, backslashes, and dollar signs without an error. Use write to create or replace the file, or edit to change part of it. The file keeps everything you write, even though the conversation later shows a long write as a short note.".into());
    }
    let sandboxed = limits.unattended && matches!(limits.isolation, Isolation::Bubblewrap { .. } | Isolation::Seatbelt);
    if limits.unattended
        && !sandboxed
        && let Some(tool) = sandbox::network_use(cmd_text)
    {
        return Err(format!(
            "Scoobert is working unattended, so it does not run `{tool}`, which downloads or installs software. Continue without it, and tell the user what to run when they are back."
        ));
    }
    let (program, args): (PathBuf, Vec<String>) = match (shell, &limits.isolation) {
        (Shell::Bash(bash), Isolation::Bubblewrap { flatpak_host }) if sandboxed => sandbox::bubblewrap(*flatpak_host, bash, cwd, cmd_text),
        (Shell::FlatpakHost, Isolation::Bubblewrap { .. }) if sandboxed => sandbox::bubblewrap(true, Path::new("/bin/bash"), cwd, cmd_text),
        (Shell::Bash(bash), Isolation::Seatbelt) if sandboxed => sandbox::seatbelt(bash, cwd, cmd_text),
        (Shell::Bash(bash), _) => (bash.clone(), vec!["-c".into(), cmd_text.into()]),
        (Shell::FlatpakHost, _) => (
            PathBuf::from("flatpak-spawn"),
            vec!["--host".into(), format!("--directory={}", cwd.display()), "bash".into(), "-c".into(), cmd_text.into()],
        ),
        (Shell::PowerShell, _) => (
            PathBuf::from("powershell.exe"),
            ["-NoProfile", "-NonInteractive", "-Command", cmd_text].map(String::from).to_vec(),
        ),
    };
    let cap = sandbox::memory_cap();
    #[cfg(unix)]
    let (program, args) = if *shell != Shell::FlatpakHost && sandbox::systemd_scope_available() {
        let mut wrapped = vec![
            "--user".to_string(),
            "--scope".into(),
            "--quiet".into(),
            "--collect".into(),
            "-p".into(),
            format!("MemoryMax={cap}"),
            "-p".into(),
            "MemorySwapMax=0".into(),
            "--".into(),
            program.to_string_lossy().into_owned(),
        ];
        wrapped.extend(args);
        (PathBuf::from("systemd-run"), wrapped)
    } else {
        (program, args)
    };
    let mut cmd = tokio::process::Command::new(&program);
    cmd.args(&args);
    // Python holds back output sent to a pipe, which hides a background server's address until it exits.
    cmd.current_dir(cwd).env("GIT_PAGER", "cat").env("PAGER", "cat").env("GIT_TERMINAL_PROMPT", "0").env("PYTHONUNBUFFERED", "1");
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    #[cfg(unix)]
    cmd.process_group(0);
    Ok((cmd, cap))
}

/// Waits for a command to end, the timeout, or Stop, and returns the end of its output.
async fn run_to_end(
    mut child: tokio::process::Child,
    pid: Option<u32>,
    timeout: Duration,
    limits: &Limits,
    cancel: &CancellationToken,
    on_output: impl Fn(String) + Send + Sync + 'static,
) -> Result<Outcome, String> {
    let output = Arc::new(Mutex::new(String::new()));
    let on_output = Arc::new(on_output);
    let pump = |mut stream: Box<dyn tokio::io::AsyncRead + Unpin + Send>| {
        let output = output.clone();
        let on_output = on_output.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 8192];
            while let Ok(n) = stream.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                let tail = {
                    let mut out = output.lock().unwrap();
                    out.push_str(&String::from_utf8_lossy(&buf[..n]));
                    // Keep memory bounded for commands that print without end.
                    if out.len() > 4 * MAX_BYTES {
                        let cut = floor_char(&out, out.len() - 2 * MAX_BYTES);
                        out.drain(..cut);
                    }
                    let start = floor_char(&out, out.len().saturating_sub(4000));
                    out[start..].to_string()
                };
                on_output(tail);
            }
        })
    };
    let a = pump(Box::new(child.stdout.take().unwrap()));
    let b = pump(Box::new(child.stderr.take().unwrap()));

    let (status, note) = tokio::select! {
        s = child.wait() => (s.ok(), None),
        _ = tokio::time::sleep(timeout) => {
            kill_tree(pid, &mut child).await;
            (None, Some(format!("The command was stopped after {} seconds.", timeout.as_secs())))
        }
        _ = cancel.cancelled() => {
            kill_tree(pid, &mut child).await;
            (None, Some("The user stopped the command.".to_string()))
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        let _ = a.await;
        let _ = b.await;
    })
    .await;
    let full = output.lock().unwrap().clone();
    let mut text = tail_lines(&full, limits.max_output);
    let code = status.and_then(|s| s.code());
    if let Some(note) = &note {
        text.push_str(&format!("\n[{note}]"));
    } else if let Some(code) = code.filter(|&c| c != 0) {
        text.push_str(&format!("\n[Exit code {code}]"));
    }
    if text.trim().is_empty() {
        text = "(No output.)".into();
    }
    Ok(Outcome { output: text, is_error: note.is_some() || code.is_some_and(|c| c != 0), ..Default::default() })
}

async fn kill_tree(pid: Option<u32>, child: &mut tokio::process::Child) {
    #[cfg(windows)]
    if let Some(pid) = pid {
        let _ = tokio::process::Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .creation_flags(0x0800_0000)
            .output()
            .await;
    }
    #[cfg(unix)]
    if let Some(pid) = pid {
        let _ = tokio::process::Command::new("kill").args(["-TERM", &format!("-{pid}")]).output().await;
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The end of a command's output, where errors usually are, within the size limits.
fn tail_lines(text: &str, max_bytes: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut start = lines.len().saturating_sub(MAX_LINES);
    let mut size: usize = lines[start..].iter().map(|l| l.len() + 1).sum();
    while size > max_bytes && start < lines.len() {
        size -= lines[start].len() + 1;
        start += 1;
    }
    let body = lines[start..].join("\n");
    if start > 0 { format!("[Showing the last {} of {} lines.]\n{body}", lines.len() - start, lines.len()) } else { body }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall { id: "1".into(), name: name.into(), arguments: args }
    }

    #[tokio::test]
    async fn edit_keeps_crlf_and_rejects_ambiguous_matches() {
        let dir = std::env::temp_dir().join(format!("scoobert-test-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "one\r\ntwo\r\ntwo\r\n").unwrap();
        let shell = Shell::detect();
        let cancel = CancellationToken::new();
        let dup = run(&call("edit", json!({"path": "a.txt", "old_text": "two", "new_text": "2"})), &dir, &shell, &Limits::default(), &cancel, |_| {}).await;
        assert!(dup.is_error && dup.output.contains("2 times"));
        let ok = run(&call("edit", json!({"path": "a.txt", "old_text": "one\ntwo", "new_text": "1\n2"})), &dir, &shell, &Limits::default(), &cancel, |_| {}).await;
        assert!(!ok.is_error, "{}", ok.output);
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "1\r\n2\r\ntwo\r\n");
        std::fs::write(dir.join("a.txt"), "one\r\ntwo\r\ntwo\r\n").unwrap();
        let all = run(&call("edit", json!({"path": "a.txt", "old_text": "two", "new_text": "2", "replace_all": true})), &dir, &shell, &Limits::default(), &cancel, |_| {}).await;
        assert!(all.output.contains("2 places"), "{}", all.output);
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "one\r\n2\r\n2\r\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn read_pages_long_files() {
        let dir = std::env::temp_dir().join(format!("scoobert-test-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        let text: String = (1..=3000).map(|i| format!("line {i}\n")).collect();
        std::fs::write(dir.join("long.txt"), text).unwrap();
        let out = run(&call("read", json!({"path": "long.txt"})), &dir, &Shell::detect(), &Limits::default(), &CancellationToken::new(), |_| {}).await;
        assert!(out.output.contains("Showing lines 1-2000 of 3000"));
        let out = run(&call("read", json!({"path": "long.txt", "offset": 2999})), &dir, &Shell::detect(), &Limits::default(), &CancellationToken::new(), |_| {}).await;
        assert_eq!(out.output, "line 2999\nline 3000\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn runs_commands() {
        let shell = Shell::detect();
        let cmd = if shell == Shell::PowerShell { "Write-Output hi" } else { "echo hi" };
        let out = run(&call(shell.tool_name(), json!({"command": cmd})), &std::env::temp_dir(), &shell, &Limits::default(), &CancellationToken::new(), |_| {}).await;
        assert_eq!(out.output.trim(), "hi");
        assert!(!out.is_error);
    }

    #[tokio::test]
    async fn reads_notes_by_link_name_with_their_links() {
        let dir = std::env::temp_dir().join(format!("scoobert-test-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(dir.join("Notes/Services")).unwrap();
        std::fs::write(dir.join("Notes/Auth design.md"), "# Auth design\n\nTokens are kept in [[Vault X]] and [[Missing]].\n").unwrap();
        std::fs::write(dir.join("Notes/Services/Vault X.md"), "# Vault X\n\nIt answers on 6380.\n").unwrap();
        let limits = Limits { notes: Some(dir.join("Notes")), ..Limits::default() };
        let out = run(&call("read", json!({"path": "[[Vault X]]"})), &dir, &Shell::detect(), &limits, &CancellationToken::new(), |_| {}).await;
        assert!(out.output.contains("answers on 6380"), "{}", out.output);
        assert!(out.output.contains("Linked from: Notes/Auth design.md"), "{}", out.output);
        let out = run(&call("read", json!({"path": "Notes/Auth design.md"})), &dir, &Shell::detect(), &limits, &CancellationToken::new(), |_| {}).await;
        assert!(out.output.contains("[[Vault X]] is Notes/Services/Vault X.md"), "{}", out.output);
        assert!(out.output.contains("[[Missing]] has no note yet"), "{}", out.output);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn checkpoints_put_files_back() {
        let dir = std::env::temp_dir().join(format!("scoobert-test-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "original").unwrap();
        let limits = Limits { checkpoints: Some(dir.join("checkpoints")), ..Limits::default() };
        let c = call("write", json!({"path": "a.txt", "content": "changed"}));
        run(&c, &dir, &Shell::detect(), &limits, &CancellationToken::new(), |_| {}).await;
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "changed");
        assert!(restore_checkpoint(&dir.join("checkpoints"), &c.id).unwrap().is_ok());
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "original");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_write_that_would_wipe_a_file_needs_sending_twice() {
        let dir = std::env::temp_dir().join(format!("scoobert-test-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        let original: String = (1..=30).map(|i| format!("let value{i} = {i};\n")).collect();
        std::fs::write(dir.join("game.js"), &original).unwrap();
        let (shell, limits, cancel) = (Shell::detect(), Limits::default(), CancellationToken::new());
        let part = call("write", json!({"path": "game.js", "content": "function startGame() {\n  run();\n}\n"}));
        let refused = run(&part, &dir, &shell, &limits, &cancel, |_| {}).await;
        assert!(refused.is_error && refused.output.contains("append set to true"), "{}", refused.output);
        assert_eq!(std::fs::read_to_string(dir.join("game.js")).unwrap(), original);
        let edited: String = original.replace("value3 = 3", "value3 = 4");
        let rewrite = call("write", json!({"path": "game.js", "content": edited}));
        assert!(!run(&rewrite, &dir, &shell, &limits, &cancel, |_| {}).await.is_error);
        let again = run(&part, &dir, &shell, &limits, &cancel, |_| {}).await;
        assert!(again.is_error, "a refusal for other content does not confirm this one");
        let confirmed = run(&part, &dir, &shell, &limits, &cancel, |_| {}).await;
        assert!(!confirmed.is_error, "{}", confirmed.output);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn heredoc_file_writes_are_spotted() {
        assert!(writes_heredoc("cat > src/a.ts << 'EOF'\nconst x = (a) => a > 1;\nEOF"));
        assert!(writes_heredoc("cd proj && cat <<EOF >> notes.md\nline\nEOF"));
        assert!(writes_heredoc("tee src/a.ts <<'EOF'\nx\nEOF"));
        assert!(writes_heredoc("@'\nconst a = 1;\n'@ | Set-Content -Path a.ts"));
        assert!(!writes_heredoc("python - <<'EOF'\nprint(1 > 0)\nx = lambda a: a\nEOF"));
        assert!(!writes_heredoc("npm run build > build.log 2>&1"));
        assert!(!writes_heredoc("node -e \"console.log(1 >> 2)\" 2>&1"));
    }

    #[tokio::test]
    async fn writes_refuse_the_saved_write_note_and_outline_appends() {
        let dir = std::env::temp_dir().join(format!("scoobert-test-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        let (shell, limits, cancel) = (Shell::detect(), Limits::default(), CancellationToken::new());
        let note = call("write", json!({"path": "a.ts", "content": "[4243 characters written to src/game/a.ts. Read the file to see them.]"}));
        let out = run(&note, &dir, &shell, &limits, &cancel, |_| {}).await;
        assert!(out.is_error && !dir.join("a.ts").exists(), "{}", out.output);
        let first = call("write", json!({"path": "a.ts", "content": "import x from \"y\";\n\nexport const PASSIVES = {\n  a: 1,\n};\n"}));
        assert!(!run(&first, &dir, &shell, &limits, &cancel, |_| {}).await.output.contains("Top-level"));
        let second = call("write", json!({"path": "a.ts", "content": "// weapons\nfunction W() {\n  return 1;\n}\n", "append": true}));
        let out = run(&second, &dir, &shell, &limits, &cancel, |_| {}).await;
        assert!(out.output.contains("Top-level lines in the file now, without the lines inside classes and functions:\nimport x from \"y\";\nexport const PASSIVES = {\nfunction W() {"), "{}", out.output);
        let _ = std::fs::remove_dir_all(dir);
    }
}
