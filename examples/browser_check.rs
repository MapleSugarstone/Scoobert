//! Drives the background command and browser tools against a small page served from a temporary folder.

use std::sync::Arc;

use scoobert::agent::conversation::ToolCall;
use scoobert::agent::jobs::Jobs;
use scoobert::agent::tools::{self, Limits, Shell};
use serde_json::json;
use tokio_util::sync::CancellationToken;

const PAGE: &str = r#"<!doctype html><title>Test game</title>
<h1>Score: <span id=score>0</span></h1>
<input id=name placeholder="Your name">
<button onclick="score.textContent = Number(score.textContent) + 1">Add point</button>
<canvas id=c width=200 height=100 style="background:#eee"></canvas>
<script>
const keys = [];
addEventListener('keydown', e => { keys.push(e.key); document.title = 'Keys ' + keys.join(','); });
c.addEventListener('click', e => console.log('canvas click at', e.offsetX, e.offsetY));
console.error('test error');
</script>"#;

#[tokio::main]
async fn main() {
    let dir = std::env::temp_dir().join(format!("scoobert-browser-check-{}", scoobert::util::random_hex(4)));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("index.html"), PAGE).unwrap();
    let shell = Shell::detect();
    let limits = Limits {
        jobs: Some(Arc::new(Jobs::new(Arc::new(|jobs| println!("[jobs now: {jobs:?}]"))))),
        browser: Some(tools::BrowserSlot::default()),
        ..Limits::default()
    };
    let cancel = CancellationToken::new();
    let steps = [
        (shell.tool_name(), json!({ "command": "python -m http.server 8765 --bind 127.0.0.1", "background": true })),
        ("job_output", json!({ "id": 1, "wait": 3 })),
        ("browser_open", json!({ "url": "localhost:8765" })),
        ("browser_click", json!({ "element": 2 })),
        ("browser_type", json!({ "text": "Maple", "element": 1, "submit": true })),
        ("browser_key", json!({ "key": "ArrowLeft", "times": 2 })),
        ("browser_click", json!({ "x": 60, "y": 150 })),
        ("browser_script", json!({ "code": "({ score: score.textContent, name: document.querySelector('#name').value, title: document.title })" })),
        ("browser_screenshot", json!({})),
        (
            "browser_script",
            json!({ "code": "const reach = u => fetch(u, { mode: 'no-cors' }).then(() => 'reached', e => 'blocked (' + e.message + ')'); ({ site: await reach('https://example.com'), ip: await reach('http://1.1.1.1'), local: await reach('/') })" }),
        ),
        ("browser_open", json!({ "url": "index.html" })),
        ("job_stop", json!({ "id": 1 })),
        ("browser_open", json!({ "url": "https://example.com" })),
    ];
    for (i, (name, args)) in steps.into_iter().enumerate() {
        let call = ToolCall { id: i.to_string(), name: name.to_string(), arguments: args };
        let out = tools::run(&call, &dir, &shell, &limits, &cancel, |_| {}).await;
        let image = out.image.as_ref().map(|img| format!(" [image {} bytes of base64 {}]", img.data.len(), img.mime)).unwrap_or_default();
        if let (Some(img), Some(path)) = (&out.image, std::env::var_os("SHOT_PATH")) {
            use base64::Engine;
            std::fs::write(path, base64::engine::general_purpose::STANDARD.decode(&img.data).unwrap()).unwrap();
        }
        println!("=== {name}{} {}{image}\n{}\n", if out.is_error { " (error)" } else { "" }, call.arguments, scoobert::util::clip(&out.output, 900));
    }
    drop(limits);
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    let _ = std::fs::remove_dir_all(dir);
}
