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

/// Tool definitions in the OpenAI function format.
pub fn specs(shell: &Shell) -> Vec<Value> {
    let path = json!({ "type": "string", "description": "File path, relative to the project folder or absolute." });
    let tool = |name: &str, description: &str, properties: Value, required: &[&str]| {
        json!({ "type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": { "type": "object", "properties": properties, "required": required },
        }})
    };
    let shell_tool = match shell {
        Shell::PowerShell => tool(
            "powershell",
            "Run a PowerShell command in the project folder and return its output. Commands stop after 10 minutes unless you set timeout.",
            json!({ "command": { "type": "string" }, "timeout": { "type": "integer", "description": "Seconds." } }),
            &["command"],
        ),
        _ => tool(
            "bash",
            "Run a bash command in the project folder and return its output. Commands stop after 10 minutes unless you set timeout.",
            json!({ "command": { "type": "string" }, "timeout": { "type": "integer", "description": "Seconds." } }),
            &["command"],
        ),
    };
    vec![
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
            "Replace one exact piece of text in a file. old_text must match the file exactly, including indentation, and appear only once. Include nearby lines to make it unique.",
            json!({ "path": path, "old_text": { "type": "string" }, "new_text": { "type": "string" } }),
            &["path", "old_text", "new_text"],
        ),
        tool(
            "write",
            "Create a file or replace a whole file. Creates missing folders.",
            json!({ "path": path, "content": { "type": "string" } }),
            &["path", "content"],
        ),
        shell_tool,
    ]
}

pub fn is_read_only(name: &str) -> bool {
    name == "read"
}

pub fn changes_files(name: &str) -> bool {
    name == "edit" || name == "write"
}

#[derive(Debug, Clone, Default)]
pub struct Outcome {
    pub output: String,
    pub is_error: bool,
    pub diff: Option<String>,
}

impl Outcome {
    fn ok(output: impl Into<String>) -> Self {
        Outcome { output: output.into(), ..Default::default() }
    }

    fn err(output: impl Into<String>) -> Self {
        Outcome { output: output.into(), is_error: true, diff: None }
    }
}

/// Looks up an argument under the names different models use for it.
fn arg<'a>(call: &'a ToolCall, names: &[&str]) -> Option<&'a str> {
    names.iter().find_map(|n| call.arguments.get(*n).and_then(Value::as_str))
}

fn arg_u64(call: &ToolCall, name: &str) -> Option<u64> {
    let v = call.arguments.get(name)?;
    v.as_u64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
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

/// Restrictions on commands for the current approval mode.
#[derive(Clone, Debug)]
pub struct Limits {
    pub unattended: bool,
    pub isolation: Isolation,
    /// Output size that fits comfortably in the model's context.
    pub max_output: usize,
    /// The project's notes folder, for reading notes by name and listing their links.
    pub notes: Option<PathBuf>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { unattended: false, isolation: Isolation::Refuse, max_output: MAX_BYTES, notes: None }
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
        "edit" => edit(call, cwd, limits.notes.as_deref()).await,
        "write" => write(call, cwd, limits.notes.as_deref()).await,
        "bash" | "powershell" => command(call, cwd, shell, limits, cancel, on_output).await,
        other => Err(format!("There is no tool named {other}. Use read, edit, write, or {}.", shell.tool_name())),
    };
    result.unwrap_or_else(Outcome::err)
}

async fn read(call: &ToolCall, cwd: &Path, limits: &Limits) -> Result<Outcome, String> {
    let max_bytes = limits.max_output;
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

async fn write(call: &ToolCall, cwd: &Path, notes: Option<&Path>) -> Result<Outcome, String> {
    let path = target_path(cwd, call, notes).ok_or("write needs a path.")?;
    let content = arg(call, &["content", "text", "contents"]).ok_or("write needs content.")?;
    let old = tokio::fs::read_to_string(&path).await.ok();
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| format!("Could not create {}: {e}", paths::display(dir)))?;
    }
    tokio::fs::write(&path, content).await.map_err(|e| format!("Could not write {}: {e}", paths::display(&path)))?;
    let shown = paths::display(&path);
    Ok(Outcome {
        output: format!("{} {shown} ({} bytes).", if old.is_some() { "Replaced" } else { "Created" }, content.len()),
        is_error: false,
        diff: Some(unified_diff(old.as_deref().unwrap_or(""), content)),
    })
}

async fn edit(call: &ToolCall, cwd: &Path, notes: Option<&Path>) -> Result<Outcome, String> {
    let path = target_path(cwd, call, notes).ok_or("edit needs a path.")?;
    let old_text = arg(call, &["old_text", "oldText", "old_string", "old_str"]).ok_or("edit needs old_text.")?;
    let new_text = arg(call, &["new_text", "newText", "new_string", "new_str"]).ok_or("edit needs new_text.")?;
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
    if count > 1 {
        return Err(format!("old_text appears {count} times in {shown}. Include more surrounding lines so it matches once."));
    }
    let updated = text.replacen(&old_n, &new_n, 1);
    let written = if crlf { updated.replace('\n', "\r\n") } else { updated.clone() };
    tokio::fs::write(&path, &written).await.map_err(|e| format!("Could not write {shown}: {e}"))?;
    Ok(Outcome { output: format!("Edited {shown}."), is_error: false, diff: Some(unified_diff(&text, &updated)) })
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
    let sandboxed = limits.unattended && matches!(limits.isolation, Isolation::Bubblewrap { .. });
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
    cmd.current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat")
        .env("GIT_TERMINAL_PROMPT", "0")
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    #[cfg(unix)]
    cmd.process_group(0);
    let mut child = cmd.spawn().map_err(|e| format!("Could not start the shell: {e}"))?;
    let pid = child.id();
    // The job caps memory for the command and everything it starts, and ends them all when it is dropped.
    #[cfg(windows)]
    let _job = sandbox::Job::new(cap).filter(|job| pid.is_some_and(|p| job.assign(p)));
    #[cfg(unix)]
    let _ = cap;

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
    Ok(Outcome { output: text, is_error: note.is_some() || code.is_some_and(|c| c != 0), diff: None })
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
}