//! A browser the model drives to test the pages and games it builds: it opens a page, reads its text and controls,
//! clicks, types, presses keys, runs scripts, and takes screenshots. Chrome, Edge, and Chromium speak the Chrome
//! DevTools Protocol, and Firefox speaks WebDriver BiDi, over the same kind of WebSocket.

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::oneshot;

use super::conversation::Image;
use super::prompt::{outside, system_note};
use super::web::{Engine, Found, Sandbox};

const CALL_TIMEOUT: Duration = Duration::from_secs(30);
const START_TIMEOUT: Duration = Duration::from_secs(30);
const LOAD_TIMEOUT: Duration = Duration::from_secs(15);
/// Memory for the browser and its helper processes, enough for a game that draws with WebGL.
#[cfg(windows)]
const BROWSER_MEMORY: u64 = 2_000_000_000;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 800;
const MAX_CONSOLE: usize = 200;
const MAX_MESSAGE: u64 = 64 * 1024 * 1024;
/// Page text shown at once.
const PAGE_CHARS: usize = 4000;
const SCRIPT_CHARS: usize = 20_000;

/// Numbers the visible controls with a data attribute, so a later click or typing finds the same element, and
/// returns them with the page's text.
const SNAPSHOT: &str = r#"(() => {
  document.querySelectorAll('[data-scoobert-ref]').forEach(e => e.removeAttribute('data-scoobert-ref'));
  const selector = 'a[href],button,input,select,textarea,summary,canvas,[role=button],[role=link],[role=checkbox],[role=tab],[role=menuitem],[onclick],[contenteditable=""],[contenteditable=true],[tabindex]:not([tabindex="-1"])';
  const controls = [];
  let n = 0, hidden = 0;
  for (const e of document.querySelectorAll(selector)) {
    const r = e.getBoundingClientRect();
    const style = getComputedStyle(e);
    if (r.width < 1 || r.height < 1 || style.visibility === 'hidden' || style.display === 'none') continue;
    if (n >= 150) { hidden++; continue; }
    n++;
    e.setAttribute('data-scoobert-ref', n);
    const tag = e.tagName.toLowerCase();
    let kind = tag === 'a' ? 'link' : tag;
    if (tag === 'input') kind += ' type=' + (e.type || 'text');
    if (e.getAttribute('role')) kind += ' role=' + e.getAttribute('role');
    const label = (e.getAttribute('aria-label') || (tag === 'input' || tag === 'textarea' || tag === 'select' ? '' : e.innerText) || e.placeholder || e.title || e.alt || '').trim().replace(/\s+/g, ' ').slice(0, 80);
    let extra = '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select') extra += ' value="' + String(e.value).slice(0, 60) + '"';
    if (e.placeholder) extra += ' placeholder="' + e.placeholder.slice(0, 40) + '"';
    if (e.disabled) extra += ' disabled';
    if (e.checked) extra += ' checked';
    if (tag === 'canvas') extra += ' ' + Math.round(r.width) + 'x' + Math.round(r.height) + ' at x=' + Math.round(r.x) + ' y=' + Math.round(r.y);
    if (r.bottom < 0 || r.top > innerHeight) extra += ' (scrolled out of view)';
    controls.push('[' + n + '] ' + kind + (label ? ' "' + label + '"' : '') + extra);
  }
  return { title: document.title, url: location.href, text: document.body ? document.body.innerText : '', controls, hidden };
})()"#;

/// The browser process with its throwaway profile, which closes both when dropped.
struct Process {
    child: tokio::process::Child,
    #[cfg(windows)]
    _job: Option<super::sandbox::Job>,
    profile: PathBuf,
}

impl Drop for Process {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            let _ = std::process::Command::new("kill").args(["-KILL", &format!("-{pid}")]).output();
        }
        let _ = self.child.start_kill();
        // The profile can be removed once the processes that hold its files end.
        let profile = self.profile.clone();
        std::thread::spawn(move || {
            for _ in 0..20 {
                if std::fs::remove_dir_all(&profile).is_ok() || !profile.exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        });
    }
}

pub struct Browser {
    page: Page,
    /// How many console lines the model has seen.
    seen: usize,
    _process: Process,
}

/// The connection to the page the model drives.
enum Page {
    /// Chrome, Edge, or Chromium, through the Chrome DevTools Protocol.
    Cdp(Socket),
    /// Firefox, through WebDriver BiDi, with the browsing context of its one tab.
    Bidi(Socket, String),
}

impl Page {
    fn socket(&self) -> &Socket {
        match self {
            Page::Cdp(socket) | Page::Bidi(socket, _) => socket,
        }
    }
}

impl Browser {
    /// Starts a headless browser with a throwaway profile, so it has no sign-ins, history, or extensions. Without
    /// `web` it sends every request through a proxy that does not exist, so only pages on this computer and files
    /// load. That covers what a page fetches and what a script sends, not only the address the model opens. `cwd`
    /// is the project, which a browser in a Flatpak sandbox is given to read so it can open the project's files.
    pub async fn launch(web: bool, cwd: &Path) -> anyhow::Result<Browser> {
        let found = super::web::find_browser().context("The browser tools need Firefox, Chrome, Edge, or Chromium, and none is installed.")?;
        match found.engine {
            Engine::Firefox => Self::launch_firefox(&found, web, cwd).await,
            // Edge closed as it started in every attempt of one session on Windows, while it started from another
            // program, so a Chromium browser that will not start leaves the work to Firefox when it is installed.
            Engine::Chromium => match Self::launch_chromium(&found.program, web).await {
                Err(err) if err.is::<ClosedAtStart>() => match super::web::find_firefox() {
                    Some(firefox) => Self::launch_firefox(&firefox, web, cwd).await.map_err(|other| anyhow::anyhow!("{err:#}\nFirefox did not start either: {other:#}")),
                    None => Err(err),
                },
                other => other,
            },
        }
    }

    async fn launch_chromium(exe: &Path, web: bool) -> anyhow::Result<Browser> {
        let profile = std::env::temp_dir().join(format!("scoobert-browser-{}", crate::util::random_hex(6)));
        std::fs::create_dir_all(&profile).with_context(|| format!("Could not create {}", profile.display()))?;
        let mut cmd = tokio::process::Command::new(exe);
        cmd.args([
            "--headless=new",
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-sync",
            "--disable-background-networking",
            "--mute-audio",
            "--hide-scrollbars",
            // A game that draws with WebGL gets a software renderer when the browser has no graphics card to use.
            "--enable-unsafe-swiftshader",
            // Warnings and errors go to the error output, which explains a browser that closes as it starts.
            "--enable-logging=stderr",
            "--log-level=1",
        ])
        .arg(format!("--window-size={WIDTH},{HEIGHT}"))
        .arg(format!("--user-data-dir={}", profile.display()));
        // Loopback addresses skip a proxy on their own, and nothing listens on port 9.
        if !web {
            cmd.args(["--proxy-server=http://127.0.0.1:9", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]);
        }
        cmd.arg("about:blank")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd.spawn().with_context(|| format!("Could not start {}", exe.display()))?;
        let output = read_output(child.stderr.take());
        // A job ends the browser's helper processes with it and caps their memory.
        #[cfg(windows)]
        let job = super::sandbox::Job::new(BROWSER_MEMORY).filter(|j| child.id().is_some_and(|p| j.assign(p)));
        let mut process = Process {
            child,
            #[cfg(windows)]
            _job: job,
            profile,
        };
        let port_file = process.profile.join("DevToolsActivePort");
        let start = Instant::now();
        let port: u16 = loop {
            if let Ok(text) = tokio::fs::read_to_string(&port_file).await
                && let Some(port) = text.lines().next().and_then(|l| l.trim().parse().ok())
            {
                break port;
            }
            if let Ok(Some(status)) = process.child.try_wait() {
                return Err(closed(status, &output).await);
            }
            if start.elapsed() > START_TIMEOUT {
                bail!("The browser did not start within {} seconds.", START_TIMEOUT.as_secs());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        let http = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(10)).build()?;
        let targets: Value = http.get(format!("http://127.0.0.1:{port}/json/list")).send().await?.json().await?;
        let socket = targets
            .as_array()
            .into_iter()
            .flatten()
            .find(|t| t["type"] == "page")
            .and_then(|t| t["webSocketDebuggerUrl"].as_str())
            .context("The browser opened no page.")?
            .to_string();
        let cdp = Socket::connect(&socket, false).await?;
        // A page that is not focused pauses some games and ignores keys.
        for (method, params) in [
            ("Page.enable", json!({})),
            ("Runtime.enable", json!({})),
            ("Log.enable", json!({})),
            ("Emulation.setFocusEmulationEnabled", json!({ "enabled": true })),
        ] {
            cdp.call(method, params).await?;
        }
        // A page cannot save files to the computer. Older browsers lack the command, and they deny downloads in
        // headless mode anyway.
        let _ = cdp.call("Page.setDownloadBehavior", json!({ "behavior": "deny" })).await;
        Ok(Browser { page: Page::Cdp(cdp), seen: 0, _process: process })
    }

    async fn launch_firefox(found: &Found, web: bool, cwd: &Path) -> anyhow::Result<Browser> {
        let profile = firefox_profile(found);
        std::fs::create_dir_all(profile.join("downloads")).with_context(|| format!("Could not create {}", profile.display()))?;
        std::fs::write(profile.join("user.js"), firefox_prefs(&profile, web)).with_context(|| format!("Could not write the browser's settings in {}", profile.display()))?;
        let mut cmd = tokio::process::Command::new(&found.program);
        // Headless Firefox takes its screen size from these rather than from --window-size.
        let size = [("MOZ_HEADLESS_WIDTH", WIDTH), ("MOZ_HEADLESS_HEIGHT", HEIGHT)];
        if let Sandbox::Flatpak(app) = found.sandbox {
            cmd.arg("run").arg(format!("--filesystem={}:ro", cwd.display()));
            cmd.args(size.map(|(name, value)| format!("--env={name}={value}")));
            cmd.arg(app);
        }
        cmd.envs(size.map(|(name, value)| (name, value.to_string())));
        cmd.args(["--headless", "--no-remote", "--remote-debugging-port", "0", "-profile"])
            .arg(&profile)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd.spawn().with_context(|| format!("Could not start {}", found.program.display()))?;
        // Firefox names its port in a file in the profile, and in its error output, which a sandboxed one may only
        // have.
        let output = read_output(child.stderr.take());
        #[cfg(windows)]
        let job = super::sandbox::Job::new(BROWSER_MEMORY).filter(|j| child.id().is_some_and(|p| j.assign(p)));
        let mut process = Process {
            child,
            #[cfg(windows)]
            _job: job,
            profile,
        };
        let port_file = process.profile.join("WebDriverBiDiServer.json");
        let start = Instant::now();
        let port: u16 = loop {
            let written = tokio::fs::read_to_string(&port_file).await.ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["ws_port"].as_u64());
            if let Some(port) = written.and_then(|p| u16::try_from(p).ok()).or(output.lock().unwrap().port) {
                break port;
            }
            if let Ok(Some(status)) = process.child.try_wait() {
                return Err(closed(status, &output).await);
            }
            if start.elapsed() > START_TIMEOUT {
                bail!("The browser did not start within {} seconds.", START_TIMEOUT.as_secs());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        let bidi = Socket::connect(&format!("ws://127.0.0.1:{port}/session"), true).await?;
        bidi.call("session.new", json!({ "capabilities": {} })).await?;
        let tree = bidi.call("browsingContext.getTree", json!({ "maxDepth": 0 })).await?;
        let context = match tree["contexts"][0]["context"].as_str() {
            Some(c) => c.to_string(),
            None => bidi.call("browsingContext.create", json!({ "type": "tab" })).await?["context"].as_str().context("The browser opened no page.")?.to_string(),
        };
        bidi.call("session.subscribe", json!({ "events": ["log.entryAdded", "browsingContext.userPromptOpened"] })).await?;
        // Firefox 157 allows this only with access to its own internals, which a test page does not need, so the
        // window size from the command line stands when it refuses.
        let _ = bidi.call("browsingContext.setViewport", json!({ "context": context, "viewport": { "width": WIDTH, "height": HEIGHT } })).await;
        Ok(Browser { page: Page::Bidi(bidi, context), seen: 0, _process: process })
    }

    pub async fn open(&mut self, url: &str) -> anyhow::Result<String> {
        let failed = match &self.page {
            Page::Cdp(cdp) => cdp.call("Page.navigate", json!({ "url": url })).await?["errorText"].as_str().map(str::to_string),
            Page::Bidi(bidi, context) => bidi.call("browsingContext.navigate", json!({ "context": context, "url": url, "wait": "interactive" })).await.err().map(|e| format!("{e:#}")),
        };
        if let Some(err) = failed {
            let hint = if err.contains("CONNECTION_REFUSED") {
                " Nothing answers at that address. Start the server with the shell tool and background set to true, then check its output with job_output."
            } else {
                ""
            };
            bail!("Could not open {url}: {}.{hint}", err.trim_end_matches('.'));
        }
        self.settle().await;
        self.read(0).await
    }

    /// The page's text from `offset`, its numbered controls, and console messages the model has not seen.
    pub async fn read(&mut self, offset: usize) -> anyhow::Result<String> {
        let page = self.eval(SNAPSHOT).await?;
        let url = &outside(page["url"].as_str().unwrap_or_default());
        let title = &outside(page["title"].as_str().unwrap_or_default());
        let text = &outside(page["text"].as_str().unwrap_or_default());
        let mut out = String::new();
        if !is_local(url) {
            out.push_str(super::web::UNTRUSTED);
            out.push('\n');
        }
        out.push_str(&format!("# {}\n{url}\n\n", if title.is_empty() { url } else { title }));
        let total = text.chars().count();
        let start = offset.min(total);
        let section: String = text.chars().skip(start).take(PAGE_CHARS).collect();
        let end = start + section.chars().count();
        if section.trim().is_empty() {
            out.push_str(&format!("{}\n", system_note("The page shows no text.")));
        } else {
            out.push_str(section.trim_end());
            out.push('\n');
        }
        if end < total {
            out.push_str(&format!("{}\n", system_note(&format!("Showing characters {start} to {end} of {total}. Use browser_read with offset={end} to read on."))));
        }
        let controls: Vec<&str> = page["controls"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        if !controls.is_empty() {
            out.push_str("\nControls, by the number browser_click and browser_type take as element:\n");
            for c in controls {
                out.push_str(&outside(c));
                out.push('\n');
            }
            if let Some(n) = page["hidden"].as_u64().filter(|&n| n > 0) {
                out.push_str(&format!("{}\n", system_note(&format!("{n} more controls are not numbered."))));
            }
        }
        out.push_str(&self.console_news());
        Ok(out)
    }

    pub async fn click(&mut self, element: Option<u64>, point: Option<(f64, f64)>) -> anyhow::Result<String> {
        let before = self.location().await;
        let (x, y, what) = match (element, point) {
            (Some(n), _) => {
                let script = format!(
                    r#"(() => {{ const e = document.querySelector('[data-scoobert-ref="{n}"]'); if (!e) return null; e.scrollIntoView({{ block: 'center', inline: 'center' }}); const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()"#
                );
                let found = self.eval(&script).await?;
                let Some(pos) = found.as_array() else { bail!("There is no control [{n}] on the page now. Call browser_read to number the controls again.") };
                (pos[0].as_f64().unwrap_or(0.0), pos[1].as_f64().unwrap_or(0.0), format!("[{n}]"))
            }
            (None, Some((x, y))) => (x, y, format!("the point x={x}, y={y}")),
            _ => bail!("browser_click needs element, the number of a control from browser_read, or x and y."),
        };
        match &self.page {
            Page::Cdp(cdp) => {
                cdp.call("Input.dispatchMouseEvent", json!({ "type": "mouseMoved", "x": x, "y": y })).await?;
                for (kind, buttons) in [("mousePressed", 1), ("mouseReleased", 0)] {
                    cdp.call("Input.dispatchMouseEvent", json!({ "type": kind, "x": x, "y": y, "button": "left", "buttons": buttons, "clickCount": 1 })).await?;
                }
            }
            Page::Bidi(bidi, context) => {
                let (x, y) = (x.round() as i64, y.round() as i64);
                let steps = json!([{ "type": "pointerMove", "x": x, "y": y }, { "type": "pointerDown", "button": 0 }, { "type": "pointerUp", "button": 0 }]);
                let mouse = json!({ "type": "pointer", "id": "mouse", "parameters": { "pointerType": "mouse" }, "actions": steps });
                bidi.call("input.performActions", json!({ "context": context, "actions": [mouse] })).await?;
            }
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        Ok(self.after(format!("Clicked {what}."), &before).await)
    }

    pub async fn type_text(&mut self, text: &str, element: Option<u64>, submit: bool) -> anyhow::Result<String> {
        let before = self.location().await;
        if let Some(n) = element {
            let script = format!(
                r#"(() => {{ const e = document.querySelector('[data-scoobert-ref="{n}"]'); if (!e) return false; e.scrollIntoView({{ block: 'center' }}); e.focus(); if (typeof e.select === 'function') e.select(); return true; }})()"#
            );
            if self.eval(&script).await? != true {
                bail!("There is no control [{n}] on the page now. Call browser_read to number the controls again.");
            }
        }
        match &self.page {
            Page::Cdp(cdp) => {
                cdp.call("Input.insertText", json!({ "text": text })).await?;
            }
            Page::Bidi(bidi, context) => {
                let presses: Vec<Value> = text
                    .chars()
                    .filter(|&c| c != '\r')
                    .flat_map(|c| {
                        let value = match c {
                            '\n' => "\u{E007}".to_string(),
                            '\t' => "\u{E004}".to_string(),
                            c => c.to_string(),
                        };
                        [json!({ "type": "keyDown", "value": value }), json!({ "type": "keyUp", "value": value })]
                    })
                    .collect();
                let keyboard = json!({ "type": "key", "id": "keyboard", "actions": presses });
                bidi.call("input.performActions", json!({ "context": context, "actions": [keyboard] })).await?;
            }
        }
        if submit {
            self.press(&parse_key("Enter")?, Duration::ZERO).await?;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        let done = format!("Typed {} characters{}.", text.chars().count(), if submit { " and pressed Enter" } else { "" });
        Ok(self.after(done, &before).await)
    }

    pub async fn keys(&mut self, spec: &str, hold_ms: u64, times: u32) -> anyhow::Result<String> {
        let key = parse_key(spec)?;
        let before = self.location().await;
        let times = times.clamp(1, 50);
        let hold = Duration::from_millis(hold_ms.min(10_000));
        for i in 0..times {
            if i > 0 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            self.press(&key, hold).await?;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut done = format!("Pressed {spec}");
        if times > 1 {
            done.push_str(&format!(" {times} times"));
        }
        if !hold.is_zero() {
            done.push_str(&format!(", holding it {} ms each time", hold.as_millis()));
        }
        done.push('.');
        Ok(self.after(done, &before).await)
    }

    /// Runs JavaScript in the page and returns its value. The code can use await.
    pub async fn script(&mut self, code: &str) -> anyhow::Result<String> {
        let result = match &self.page {
            Page::Cdp(cdp) => {
                let params = json!({ "expression": code, "returnByValue": true, "awaitPromise": true, "userGesture": true, "replMode": true, "timeout": 20000 });
                let r = cdp.call("Runtime.evaluate", params).await?;
                match r.get("exceptionDetails") {
                    Some(details) => Err(exception_text(details)),
                    None => {
                        let result = &r["result"];
                        Ok(match result.get("value") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => serde_json::to_string_pretty(other).unwrap_or_default(),
                            None => result["unserializableValue"].as_str().or(result["description"].as_str()).unwrap_or("undefined").to_string(),
                        })
                    }
                }
            }
            Page::Bidi(bidi, context) => {
                // A script runs as one, where await outside a function is a syntax error or a plain name. Code that
                // awaits runs as an async function instead: as the expression it returns, or failing that as its body.
                let r = if code.contains("await") {
                    let r = evaluate(bidi, context, &format!("(async () => ({code}\n))()")).await?;
                    if r.as_ref().is_err_and(|e| e.contains("SyntaxError")) { evaluate(bidi, context, &async_body(code)).await? } else { r }
                } else {
                    evaluate(bidi, context, code).await?
                };
                r.map(|v| match (v["type"].as_str(), plain(&v)) {
                    (Some("undefined"), _) => "undefined".to_string(),
                    (_, Value::String(s)) => s,
                    (_, other) => serde_json::to_string_pretty(&other).unwrap_or_default(),
                })
            }
        };
        match result {
            Ok(value) => {
                let mut out = crate::util::clip(&outside(&value), SCRIPT_CHARS);
                out.push_str(&self.console_news());
                Ok(out)
            }
            Err(error) => bail!("The script threw an error: {}{}", outside(&error), self.console_news()),
        }
    }

    /// An image of what the page shows, and a line saying what it is.
    pub async fn screenshot(&mut self) -> anyhow::Result<(Image, String)> {
        let (mime, r) = match &self.page {
            Page::Cdp(cdp) => ("image/jpeg", cdp.call("Page.captureScreenshot", json!({ "format": "jpeg", "quality": 80 })).await?),
            Page::Bidi(bidi, context) => {
                let jpeg = json!({ "context": context, "format": { "type": "image/jpeg", "quality": 0.8 } });
                match bidi.call("browsingContext.captureScreenshot", jpeg).await {
                    Ok(r) => ("image/jpeg", r),
                    // A browser that cannot write JPEG gives its default, PNG.
                    Err(_) => ("image/png", bidi.call("browsingContext.captureScreenshot", json!({ "context": context })).await?),
                }
            }
        };
        let data = r["data"].as_str().context("The browser returned no screenshot.")?.to_string();
        let url = outside(&self.location().await);
        let title = outside(&self.eval("document.title").await.ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default());
        let what = if title.is_empty() { url.clone() } else { format!("{title} ({url})") };
        let mut line = format!("Took a screenshot of {what}.");
        line.push_str(&self.console_news());
        Ok((Image { mime: mime.into(), data }, line))
    }

    pub async fn location(&self) -> String {
        self.eval("location.href").await.ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
    }

    /// The value of a JavaScript expression, as JSON.
    async fn eval(&self, expression: &str) -> anyhow::Result<Value> {
        match &self.page {
            Page::Cdp(cdp) => {
                let r = cdp.call("Runtime.evaluate", json!({ "expression": expression, "returnByValue": true, "awaitPromise": true })).await?;
                if let Some(details) = r.get("exceptionDetails") {
                    bail!("{}", exception_text(details));
                }
                Ok(r["result"]["value"].clone())
            }
            Page::Bidi(bidi, context) => match evaluate(bidi, context, expression).await? {
                Ok(v) => Ok(plain(&v)),
                Err(e) => bail!("{e}"),
            },
        }
    }

    /// Waits for the page to finish loading, and a moment more for the scripts that run after it.
    async fn settle(&self) {
        let start = Instant::now();
        while start.elapsed() < LOAD_TIMEOUT {
            if self.eval("document.readyState").await.is_ok_and(|v| v == "complete") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    /// What an action did: the new page when it went to one, and the console messages it caused.
    async fn after(&mut self, done: String, before: &str) -> String {
        let now = self.location().await;
        if now != before {
            self.settle().await;
            if let Ok(page) = self.read(0).await {
                return format!("{done} The browser went to another page.\n\n{page}");
            }
        }
        format!("{done}{}", self.console_news())
    }

    async fn press(&self, key: &Key, hold: Duration) -> anyhow::Result<()> {
        let cdp = match &self.page {
            Page::Cdp(cdp) => cdp,
            Page::Bidi(bidi, context) => {
                // WebDriver names the modifiers and special keys by code points from U+E000.
                let mods: Vec<&str> = [(8, "\u{E008}"), (2, "\u{E009}"), (1, "\u{E00A}"), (4, "\u{E03D}")].into_iter().filter(|(bit, _)| key.modifiers & bit != 0).map(|(_, v)| v).collect();
                let value = webdriver_key(key);
                let mut steps: Vec<Value> = mods.iter().map(|m| json!({ "type": "keyDown", "value": m })).collect();
                steps.push(json!({ "type": "keyDown", "value": value }));
                if !hold.is_zero() {
                    steps.push(json!({ "type": "pause", "duration": hold.as_millis() as u64 }));
                }
                steps.push(json!({ "type": "keyUp", "value": value }));
                steps.extend(mods.iter().rev().map(|m| json!({ "type": "keyUp", "value": m })));
                let keyboard = json!({ "type": "key", "id": "keyboard", "actions": steps });
                bidi.call("input.performActions", json!({ "context": context, "actions": [keyboard] })).await?;
                return Ok(());
            }
        };
        let mut down = json!({
            "type": if key.text.is_some() { "keyDown" } else { "rawKeyDown" },
            "key": key.key, "code": key.code, "windowsVirtualKeyCode": key.key_code, "nativeVirtualKeyCode": key.key_code, "modifiers": key.modifiers,
        });
        if let Some(text) = &key.text {
            down["text"] = text.clone().into();
            down["unmodifiedText"] = text.clone().into();
        }
        cdp.call("Input.dispatchKeyEvent", down).await?;
        if !hold.is_zero() {
            tokio::time::sleep(hold).await;
        }
        let up = json!({ "type": "keyUp", "key": key.key, "code": key.code, "windowsVirtualKeyCode": key.key_code, "nativeVirtualKeyCode": key.key_code, "modifiers": key.modifiers });
        cdp.call("Input.dispatchKeyEvent", up).await?;
        Ok(())
    }

    /// Console messages and errors since the model last saw them.
    fn console_news(&mut self) -> String {
        let console = self.page.socket().console.lock().unwrap();
        let first = console.total - console.lines.len();
        let new: Vec<&String> = console.lines.iter().skip(self.seen.saturating_sub(first)).collect();
        let skipped = first.saturating_sub(self.seen);
        self.seen = console.total;
        if new.is_empty() {
            return String::new();
        }
        let mut out = String::from("\n\nConsole messages and errors since the last look:\n");
        if skipped > 0 {
            out.push_str(&format!("{}\n", system_note(&format!("{skipped} earlier messages are gone."))));
        }
        for line in new {
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

/// A browser that exited before it opened its remote protocol.
#[derive(Debug)]
struct ClosedAtStart(String);

impl std::fmt::Display for ClosedAtStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ClosedAtStart {}

/// The last lines a browser printed, which explain a start that fails, and the port of its remote protocol when it
/// prints one.
#[derive(Default)]
struct Output {
    lines: VecDeque<String>,
    port: Option<u16>,
}

fn read_output(stderr: Option<tokio::process::ChildStderr>) -> Arc<Mutex<Output>> {
    let output: Arc<Mutex<Output>> = Arc::default();
    if let Some(stderr) = stderr {
        let output = output.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut o = output.lock().unwrap();
                if let Some(port) = line.split("ws://").nth(1).and_then(|rest| rest.split([':', '/']).nth(1)).and_then(|p| p.trim().parse().ok()) {
                    o.port = Some(port);
                }
                o.lines.push_back(crate::util::clip(&line, 300));
                if o.lines.len() > 12 {
                    o.lines.pop_front();
                }
            }
        });
    }
    output
}

async fn closed(status: std::process::ExitStatus, output: &Mutex<Output>) -> anyhow::Error {
    // The last lines can arrive just after the exit.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let printed: Vec<String> = output.lock().unwrap().lines.iter().cloned().collect();
    let tail = if printed.is_empty() { String::new() } else { format!(" Its last output:\n{}", outside(&printed.join("\n"))) };
    ClosedAtStart(format!("The browser closed as it started ({status}).{tail}")).into()
}

/// The page's HTML after its scripts ran, for a browser that cannot print it from the command line.
pub async fn page_html(url: &str) -> Option<String> {
    let browser = Browser::launch(true, &std::env::temp_dir()).await.ok()?;
    if let Page::Bidi(bidi, context) = &browser.page {
        bidi.call("browsingContext.navigate", json!({ "context": context, "url": url, "wait": "interactive" })).await.ok()?;
    }
    browser.settle().await;
    // Pages that build themselves from data they fetch after loading get a little longer.
    tokio::time::sleep(Duration::from_secs(2)).await;
    browser.eval("document.documentElement.outerHTML").await.ok()?.as_str().map(str::to_string)
}

/// Runs `expression` in the page through WebDriver BiDi. The outer error is the connection's, and the inner one is the
/// script's own, as text.
async fn evaluate(bidi: &Socket, context: &str, expression: &str) -> anyhow::Result<Result<Value, String>> {
    let params = json!({ "expression": expression, "target": { "context": context }, "awaitPromise": true, "resultOwnership": "none", "userActivation": true });
    let r = bidi.call("script.evaluate", params).await?;
    if r["type"] == "exception" {
        let details = &r["exceptionDetails"];
        let text = details["text"].as_str().unwrap_or("The script threw an error.");
        return Ok(Err(text.lines().take(6).collect::<Vec<_>>().join("\n")));
    }
    Ok(Ok(r["result"].clone()))
}

/// `code` as an async function that returns its last statement's value when that statement is an expression, as a
/// browser console shows it.
fn async_body(code: &str) -> String {
    let code = code.trim().trim_end_matches(';');
    let chars: Vec<(usize, char)> = code.char_indices().collect();
    let (mut depth, mut quote, mut split, mut i) = (0i32, None, 0, 0);
    while i < chars.len() {
        let (at, c) = chars[i];
        let next = chars.get(i + 1).map(|&(_, n)| n);
        match quote {
            Some(_) if c == '\\' => i += 1,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None => match c {
                '"' | '\'' | '`' => quote = Some(c),
                '/' if next == Some('/') => {
                    while i < chars.len() && chars[i].1 != '\n' {
                        i += 1;
                    }
                    continue;
                }
                '/' if next == Some('*') => {
                    i += 2;
                    while i + 1 < chars.len() && !(chars[i].1 == '*' && chars[i + 1].1 == '/') {
                        i += 1;
                    }
                    i += 1;
                }
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ';' | '\n' if depth == 0 => split = at + c.len_utf8(),
                _ => {}
            },
        }
        i += 1;
    }
    let (head, last) = code.split_at(split);
    let last = last.trim();
    let statements = ["const ", "let ", "var ", "if", "for", "while", "do ", "function", "class ", "return", "try", "switch", "throw", "}", "//", "/*"];
    if last.is_empty() || statements.iter().any(|s| last.starts_with(s)) {
        format!("(async () => {{\n{code}\n}})()")
    } else {
        format!("(async () => {{\n{head}\nreturn ({last}\n);\n}})()")
    }
}

/// A WebDriver BiDi value as plain JSON. Values JSON cannot hold, such as functions and elements, become their type.
fn plain(v: &Value) -> Value {
    let items = || v["value"].as_array().cloned().unwrap_or_default();
    match v["type"].as_str().unwrap_or_default() {
        "undefined" | "null" => Value::Null,
        "string" | "boolean" | "number" | "bigint" | "date" => v["value"].clone(),
        "array" | "set" => Value::Array(items().iter().map(plain).collect()),
        "object" | "map" => Value::Object(
            items()
                .iter()
                .map(|pair| {
                    let key = match &pair[0] {
                        Value::String(s) => s.clone(),
                        other => plain(other).as_str().map(str::to_string).unwrap_or_else(|| other.to_string()),
                    };
                    (key, plain(&pair[1]))
                })
                .collect(),
        ),
        "regexp" => format!("/{}/{}", v["value"]["pattern"].as_str().unwrap_or_default(), v["value"]["flags"].as_str().unwrap_or_default()).into(),
        other => format!("[{other}]").into(),
    }
}

/// A key as WebDriver names it: the character it types, or a code point from U+E000 for a special key.
fn webdriver_key(key: &Key) -> String {
    let special = match key.key.as_str() {
        "Enter" => 0xE007,
        "Tab" => 0xE004,
        "Escape" => 0xE00C,
        "Backspace" => 0xE003,
        "Delete" => 0xE017,
        "ArrowLeft" => 0xE012,
        "ArrowUp" => 0xE013,
        "ArrowRight" => 0xE014,
        "ArrowDown" => 0xE015,
        "Home" => 0xE011,
        "End" => 0xE010,
        "PageUp" => 0xE00E,
        "PageDown" => 0xE00F,
        "Shift" => 0xE008,
        "Control" => 0xE009,
        "Alt" => 0xE00A,
        f if f.len() > 1 && f.starts_with('F') => f[1..].parse::<u32>().ok().filter(|n| (1..=12).contains(n)).map_or(0, |n| 0xE031 + n - 1),
        _ => 0,
    };
    char::from_u32(special).filter(|_| special != 0).map_or_else(|| key.key.clone(), String::from)
}

/// Where a throwaway Firefox profile goes. A Flatpak or a Snap can only read its own folders, which the host sees at
/// the same path, and the system's temporary folder serves every other Firefox.
fn firefox_profile(found: &Found) -> PathBuf {
    let name = format!("scoobert-browser-{}", crate::util::random_hex(6));
    let home = std::env::var_os("HOME").map(PathBuf::from);
    match (&found.sandbox, home) {
        (Sandbox::Flatpak(app), Some(home)) => home.join(".var/app").join(app).join("cache").join(name),
        (Sandbox::Snap, Some(home)) => home.join("snap/firefox/common").join(name),
        _ => std::env::temp_dir().join(name),
    }
}

/// Settings for the throwaway profile. Downloads stay in the profile, which is deleted with it, and without `web`
/// every request goes through a proxy that does not exist. Firefox never sends requests to this computer through a
/// proxy, so pages on it still load.
fn firefox_prefs(profile: &Path, web: bool) -> String {
    let text = |s: &str| serde_json::to_string(s).unwrap_or_default();
    let downloads = text(&profile.join("downloads").to_string_lossy());
    let mut prefs = vec![
        ("browser.shell.checkDefaultBrowser", "false".to_string()),
        ("browser.download.folderList", "2".into()),
        ("browser.download.dir", downloads),
        ("browser.download.useDownloadDir", "true".into()),
        ("browser.download.always_ask_before_handling_new_types", "false".into()),
        ("media.volume_scale", text("0.0")),
    ];
    if !web {
        for (name, value) in [
            ("network.proxy.type", "1".to_string()),
            ("network.proxy.http", text("127.0.0.1")),
            ("network.proxy.http_port", "9".into()),
            ("network.proxy.ssl", text("127.0.0.1")),
            ("network.proxy.ssl_port", "9".into()),
            ("network.proxy.share_proxy_settings", "true".into()),
            ("network.proxy.no_proxies_on", text("")),
            ("network.proxy.allow_hijacking_localhost", "false".into()),
            ("network.dns.disablePrefetch", "true".into()),
            ("network.prefetch-next", "false".into()),
            ("media.peerconnection.enabled", "false".into()),
        ] {
            prefs.push((name, value));
        }
    }
    prefs.iter().map(|(name, value)| format!("user_pref(\"{name}\", {value});\n")).collect()
}

/// The address browser_open loads: a full address, an address on this computer without its http://, or the path of
/// a file relative to the project. Pages on this computer and files always open, and other sites only when web
/// search is on, as for web_read.
pub fn resolve_address(raw: &str, cwd: &Path, web: bool) -> anyhow::Result<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        bail!("browser_open needs url.");
    }
    let url = match reqwest::Url::parse(raw) {
        Ok(u) if matches!(u.scheme(), "http" | "https" | "file") || u.as_str() == "about:blank" => u,
        // A Windows path such as C:\game\index.html parses with its drive letter as the scheme.
        _ => {
            let lower = raw.to_lowercase();
            if lower.starts_with("localhost") || lower.starts_with("127.0.0.1") || lower.starts_with("[::1]") {
                reqwest::Url::parse(&format!("http://{raw}"))?
            } else {
                let path = super::tools::resolve(cwd, raw);
                if !path.exists() {
                    bail!("{raw} is neither a web address nor a file. Use an address such as http://localhost:3000, or the path of an HTML file in the project.");
                }
                reqwest::Url::from_file_path(&path).map_err(|_| anyhow::anyhow!("Could not make an address for {}", path.display()))?
            }
        }
    };
    if matches!(url.scheme(), "http" | "https") && !is_local(url.as_str()) {
        let host = url.host_str().unwrap_or_default().trim_matches(['[', ']']).to_lowercase();
        let private = host.ends_with(".local")
            || match host.parse::<IpAddr>() {
                Ok(IpAddr::V4(ip)) => ip.is_private() || ip.is_link_local() || ip.is_unspecified(),
                Ok(IpAddr::V6(ip)) => ip.is_unspecified() || (ip.segments()[0] & 0xfe00) == 0xfc00,
                Err(_) => false,
            };
        if private {
            bail!("The browser opens pages on this computer and public sites, not other addresses on the local network.");
        }
        if !web {
            bail!(
                "Web search is turned off, so the browser only opens pages on this computer, such as http://localhost:3000, and files. Tell the user they can turn on web search under Settings."
            );
        }
    }
    Ok(url.to_string())
}

/// Whether an address is a file or a page served by this computer.
fn is_local(url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else { return false };
    match url.scheme() {
        "file" | "about" => true,
        "http" | "https" => {
            let host = url.host_str().unwrap_or_default().trim_matches(['[', ']']).to_lowercase();
            host == "localhost" || host.ends_with(".localhost") || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
        }
        _ => false,
    }
}

fn exception_text(details: &Value) -> String {
    let what = details["exception"]["description"].as_str().or(details["text"].as_str()).unwrap_or("The script threw an error.");
    what.lines().take(6).collect::<Vec<_>>().join("\n")
}

struct Key {
    key: String,
    code: String,
    key_code: u32,
    text: Option<String>,
    modifiers: u32,
}

/// A key such as Enter, ArrowLeft, or a, with optional modifiers, as in Control+a or Shift+Tab.
fn parse_key(spec: &str) -> anyhow::Result<Key> {
    let spec = spec.trim();
    let parts: Vec<&str> = if spec == "+" { vec!["+"] } else { spec.split('+').map(str::trim).filter(|p| !p.is_empty()).collect() };
    let Some((name, mods)) = parts.split_last() else { bail!("browser_key needs key.") };
    let mut modifiers = 0;
    for m in mods {
        modifiers |= match m.to_lowercase().as_str() {
            "alt" | "option" => 1,
            "ctrl" | "control" => 2,
            "meta" | "cmd" | "command" => 4,
            "shift" => 8,
            other => bail!("{other} is not a modifier key. Use Shift, Control, Alt, or Meta."),
        };
    }
    let named = |key: &str, code: &str, key_code: u32| Key { key: key.into(), code: code.into(), key_code, text: None, modifiers };
    let lower = name.to_lowercase();
    let key = match lower.as_str() {
        "enter" | "return" => Key { text: Some("\r".into()), ..named("Enter", "Enter", 13) },
        "space" | "spacebar" | " " => Key { text: Some(" ".into()), ..named(" ", "Space", 32) },
        "tab" => named("Tab", "Tab", 9),
        "escape" | "esc" => named("Escape", "Escape", 27),
        "backspace" => named("Backspace", "Backspace", 8),
        "delete" | "del" => named("Delete", "Delete", 46),
        "arrowleft" | "left" => named("ArrowLeft", "ArrowLeft", 37),
        "arrowup" | "up" => named("ArrowUp", "ArrowUp", 38),
        "arrowright" | "right" => named("ArrowRight", "ArrowRight", 39),
        "arrowdown" | "down" => named("ArrowDown", "ArrowDown", 40),
        "home" => named("Home", "Home", 36),
        "end" => named("End", "End", 35),
        "pageup" => named("PageUp", "PageUp", 33),
        "pagedown" => named("PageDown", "PageDown", 34),
        "shift" => named("Shift", "ShiftLeft", 16),
        "control" | "ctrl" => named("Control", "ControlLeft", 17),
        "alt" => named("Alt", "AltLeft", 18),
        f if f.len() >= 2 && f.starts_with('f') && f[1..].parse::<u32>().is_ok_and(|n| (1..=12).contains(&n)) => {
            let n: u32 = f[1..].parse().unwrap_or(1);
            named(&format!("F{n}"), &format!("F{n}"), 111 + n)
        }
        _ if name.chars().count() == 1 => {
            let c = name.chars().next().unwrap_or(' ');
            let shifted = modifiers & 8 != 0;
            let shown = if shifted { c.to_ascii_uppercase() } else { c };
            let code = if c.is_ascii_alphabetic() {
                format!("Key{}", c.to_ascii_uppercase())
            } else if c.is_ascii_digit() {
                format!("Digit{c}")
            } else {
                String::new()
            };
            let key_code = if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() as u32 } else { 0 };
            // Control, Alt, and Meta combinations are shortcuts and type nothing.
            let text = (modifiers & 0b0111 == 0).then(|| shown.to_string());
            Key { key: shown.to_string(), code, key_code, text, modifiers }
        }
        _ => bail!(
            "{name} is not a key browser_key knows. Use one character or a name such as Enter, Space, Escape, Tab, Backspace, ArrowLeft, ArrowRight, ArrowUp, ArrowDown, Home, End, or F1 to F12."
        ),
    };
    Ok(key)
}

#[derive(Default)]
struct Console {
    lines: VecDeque<String>,
    /// Every line ever added, including those dropped from the front.
    total: usize,
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;
type Writer = Arc<tokio::sync::Mutex<OwnedWriteHalf>>;

/// A WebSocket connection to the browser, speaking the DevTools Protocol to one page, or WebDriver BiDi when `bidi`.
struct Socket {
    writer: Writer,
    next: AtomicU64,
    pending: Pending,
    console: Arc<Mutex<Console>>,
    reader: tokio::task::JoinHandle<()>,
    bidi: bool,
}

impl Drop for Socket {
    fn drop(&mut self) {
        self.reader.abort();
        // Firefox in a Flatpak runs outside the process group that is killed, so it is asked to close as well.
        if self.bidi
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let writer = self.writer.clone();
            runtime.spawn(async move {
                let close = json!({ "id": 0, "method": "browser.close", "params": {} }).to_string();
                let _ = send(&writer, 0x1, close.as_bytes()).await;
            });
        }
    }
}

impl Socket {
    async fn connect(address: &str, bidi: bool) -> anyhow::Result<Socket> {
        let url = reqwest::Url::parse(address)?;
        let host = url.host_str().context("The browser gave an address without a host.")?.to_string();
        let port = url.port().context("The browser gave an address without a port.")?;
        let stream = TcpStream::connect((host.as_str(), port)).await.context("Could not connect to the browser")?;
        stream.set_nodelay(true)?;
        let (read, mut write) = stream.into_split();
        use base64::Engine;
        let key = base64::engine::general_purpose::STANDARD.encode(rand::random::<[u8; 16]>());
        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n",
            url.path()
        );
        write.write_all(request.as_bytes()).await?;
        let mut read = BufReader::new(read);
        let mut status = String::new();
        read.read_line(&mut status).await?;
        if !status.contains(" 101 ") {
            bail!("The browser refused the connection: {}", status.trim());
        }
        loop {
            let mut line = String::new();
            if read.read_line(&mut line).await? == 0 {
                bail!("The browser closed the connection.");
            }
            if line == "\r\n" {
                break;
            }
        }
        let writer: Writer = Arc::new(tokio::sync::Mutex::new(write));
        let pending: Pending = Arc::default();
        let console: Arc<Mutex<Console>> = Arc::default();
        let reader = tokio::spawn(read_loop(read, writer.clone(), pending.clone(), console.clone()));
        Ok(Socket { writer, next: AtomicU64::new(1), pending, console, reader, bidi })
    }

    async fn call(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        let message = json!({ "id": id, "method": method, "params": params }).to_string();
        send(&self.writer, 0x1, message.as_bytes()).await.context("The browser closed")?;
        let reply = match tokio::time::timeout(CALL_TIMEOUT, rx).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(_)) => bail!("The browser closed."),
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                bail!("The browser did not finish {method} within {} seconds.", CALL_TIMEOUT.as_secs());
            }
        };
        // WebDriver BiDi names the error and explains it in message, and the DevTools Protocol nests both in error.
        if reply["type"] == "error" {
            let why = reply["message"].as_str().filter(|m| !m.is_empty()).or(reply["error"].as_str()).unwrap_or("The browser refused the command.");
            bail!("{why} ({method})");
        }
        if let Some(err) = reply.get("error") {
            bail!("{}", err["message"].as_str().unwrap_or("The browser refused the command."));
        }
        Ok(reply["result"].clone())
    }
}

/// Sends one WebSocket frame. A client masks every frame it sends.
async fn send(writer: &tokio::sync::Mutex<OwnedWriteHalf>, opcode: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut frame = Vec::with_capacity(payload.len() + 14);
    frame.push(0x80 | opcode);
    match payload.len() {
        n if n < 126 => frame.push(0x80 | n as u8),
        n if n <= u16::MAX as usize => {
            frame.push(0x80 | 126);
            frame.extend((n as u16).to_be_bytes());
        }
        n => {
            frame.push(0x80 | 127);
            frame.extend((n as u64).to_be_bytes());
        }
    }
    let mask: [u8; 4] = rand::random();
    frame.extend(mask);
    frame.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
    writer.lock().await.write_all(&frame).await
}

/// The next whole message, joined from its fragments. None when the browser closes the connection.
async fn read_message(read: &mut BufReader<OwnedReadHalf>, writer: &Writer) -> anyhow::Result<Option<Vec<u8>>> {
    let mut message = Vec::new();
    loop {
        let mut head = [0u8; 2];
        read.read_exact(&mut head).await?;
        let fin = head[0] & 0x80 != 0;
        let opcode = head[0] & 0x0f;
        let masked = head[1] & 0x80 != 0;
        let mut len = (head[1] & 0x7f) as u64;
        if len == 126 {
            let mut b = [0u8; 2];
            read.read_exact(&mut b).await?;
            len = u16::from_be_bytes(b) as u64;
        } else if len == 127 {
            let mut b = [0u8; 8];
            read.read_exact(&mut b).await?;
            len = u64::from_be_bytes(b);
        }
        if len > MAX_MESSAGE || message.len() as u64 + len > MAX_MESSAGE {
            bail!("The browser sent a message larger than {MAX_MESSAGE} bytes.");
        }
        let mut mask = [0u8; 4];
        if masked {
            read.read_exact(&mut mask).await?;
        }
        let mut payload = vec![0u8; len as usize];
        read.read_exact(&mut payload).await?;
        if masked {
            payload.iter_mut().enumerate().for_each(|(i, b)| *b ^= mask[i % 4]);
        }
        match opcode {
            0x8 => return Ok(None),
            0x9 => send(writer, 0xA, &payload).await?,
            0xA => {}
            _ => {
                message.extend(payload);
                if fin {
                    return Ok(Some(message));
                }
            }
        }
    }
}

async fn read_loop(mut read: BufReader<OwnedReadHalf>, writer: Writer, pending: Pending, console: Arc<Mutex<Console>>) {
    while let Ok(Some(message)) = read_message(&mut read, &writer).await {
        let Ok(value) = serde_json::from_slice::<Value>(&message) else { continue };
        if let Some(id) = value.get("id").and_then(Value::as_u64) {
            if let Some(tx) = pending.lock().unwrap().remove(&id) {
                let _ = tx.send(value);
            }
            continue;
        }
        let params = &value["params"];
        let line = match value["method"].as_str().unwrap_or_default() {
            "Runtime.consoleAPICalled" => {
                let args: Vec<String> = params["args"].as_array().into_iter().flatten().map(remote_text).collect();
                Some(format!("console.{}: {}", params["type"].as_str().unwrap_or("log"), args.join(" ")))
            }
            "Runtime.exceptionThrown" => {
                let details = &params["exceptionDetails"];
                let at = details["url"]
                    .as_str()
                    .filter(|u| !u.is_empty())
                    .map(|u| format!(" ({u}, line {})", details["lineNumber"].as_u64().unwrap_or(0) + 1))
                    .unwrap_or_default();
                Some(format!("Uncaught error: {}{at}", exception_text(details)))
            }
            "Log.entryAdded" => {
                let entry = &params["entry"];
                let level = entry["level"].as_str().unwrap_or_default();
                let url = entry["url"].as_str().unwrap_or_default();
                let at = if url.is_empty() { String::new() } else { format!(" ({url})") };
                // Browsers ask every site for an icon, and a missing one is no fault of the page.
                (matches!(level, "error" | "warning") && !url.ends_with("/favicon.ico")).then(|| format!("{level}: {}{at}", entry["text"].as_str().unwrap_or_default()))
            }
            // An alert or confirm box stops the page until someone answers it.
            "Page.javascriptDialogOpening" => {
                let answer = json!({ "id": 0, "method": "Page.handleJavaScriptDialog", "params": { "accept": true } }).to_string();
                let _ = send(&writer, 0x1, answer.as_bytes()).await;
                Some(format!("The page showed a {} box, which the system accepted: {}", params["type"].as_str().unwrap_or("dialog"), params["message"].as_str().unwrap_or_default()))
            }
            "browsingContext.userPromptOpened" => {
                let answer = json!({ "id": 0, "method": "browsingContext.handleUserPrompt", "params": { "context": params["context"], "accept": true } }).to_string();
                let _ = send(&writer, 0x1, answer.as_bytes()).await;
                Some(format!("The page showed a {} box, which the system accepted: {}", params["type"].as_str().unwrap_or("dialog"), params["message"].as_str().unwrap_or_default()))
            }
            // WebDriver BiDi reports console calls and uncaught errors as one kind of entry.
            "log.entryAdded" => {
                let text = params["text"].as_str().unwrap_or_default();
                if params["type"] == "console" {
                    Some(format!("console.{}: {text}", params["method"].as_str().unwrap_or("log")))
                } else {
                    let frame = &params["stackTrace"]["callFrames"][0];
                    let at = frame["url"]
                        .as_str()
                        .filter(|u| !u.is_empty())
                        .map(|u| format!(" ({u}, line {})", frame["lineNumber"].as_u64().unwrap_or(0) + 1))
                        .unwrap_or_default();
                    Some(format!("Uncaught error: {text}{at}"))
                }
            }
            _ => None,
        };
        if let Some(line) = line {
            let mut c = console.lock().unwrap();
            c.lines.push_back(crate::util::clip(&outside(&line), 600));
            c.total += 1;
            while c.lines.len() > MAX_CONSOLE {
                c.lines.pop_front();
            }
        }
    }
    // Dropping the waiting senders tells each call that the browser closed.
    pending.lock().unwrap().clear();
}

/// A console argument as text.
fn remote_text(arg: &Value) -> String {
    match arg.get("value") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => arg["description"].as_str().or(arg["unserializableValue"].as_str()).unwrap_or_else(|| arg["type"].as_str().unwrap_or("?")).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_with_modifiers() {
        let k = parse_key("ArrowLeft").unwrap();
        assert_eq!((k.key.as_str(), k.key_code, k.text.is_none()), ("ArrowLeft", 37, true));
        let k = parse_key("Control+a").unwrap();
        assert_eq!((k.code.as_str(), k.modifiers, k.text.is_none()), ("KeyA", 2, true));
        let k = parse_key("Shift+a").unwrap();
        assert_eq!(k.text.as_deref(), Some("A"));
        assert_eq!(parse_key("space").unwrap().text.as_deref(), Some(" "));
        assert!(parse_key("Hyper+x").is_err());
    }

    #[test]
    fn awaiting_scripts_return_their_last_expression() {
        let code = "const reach = u => fetch(u).then(() => 'ok'); // try it\n({ site: await reach('/'), n: 1 })";
        assert_eq!(
            async_body(code),
            "(async () => {\nconst reach = u => fetch(u).then(() => 'ok'); // try it\n\nreturn (({ site: await reach('/'), n: 1 })\n);\n})()"
        );
        assert_eq!(async_body("const s = 'a;b'; await go(s);"), "(async () => {\nconst s = 'a;b';\nreturn (await go(s)\n);\n})()");
        assert_eq!(async_body("for (const x of xs) { await f(x); }"), "(async () => {\nfor (const x of xs) { await f(x); }\n})()");
    }

    #[test]
    fn webdriver_names_special_keys() {
        assert_eq!(webdriver_key(&parse_key("Enter").unwrap()), "\u{E007}");
        assert_eq!(webdriver_key(&parse_key("F5").unwrap()), "\u{E035}");
        assert_eq!(webdriver_key(&parse_key("Shift+a").unwrap()), "A");
        assert_eq!(webdriver_key(&parse_key("Control+c").unwrap()), "c");
    }

    #[test]
    fn bidi_values_become_plain_json() {
        let v = json!({ "type": "object", "value": [["n", { "type": "number", "value": 2 }], ["xs", { "type": "array", "value": [{ "type": "string", "value": "a" }, { "type": "undefined" }] }]] });
        assert_eq!(plain(&v), json!({ "n": 2, "xs": ["a", null] }));
        assert_eq!(plain(&json!({ "type": "function" })), json!("[function]"));
    }

    #[test]
    fn addresses_follow_the_web_setting() {
        let dir = std::env::temp_dir();
        assert_eq!(resolve_address("localhost:3000", &dir, false).unwrap(), "http://localhost:3000/");
        assert!(resolve_address("http://127.0.0.1:8080/game", &dir, false).is_ok());
        assert!(resolve_address("https://example.com", &dir, false).is_err());
        assert!(resolve_address("https://example.com", &dir, true).is_ok());
        assert!(resolve_address("http://192.168.1.4", &dir, true).is_err());
        assert!(resolve_address("no-such-file.html", &dir, true).is_err());
    }
}
