//! Commands the model starts in the background, such as a development server, which keep running while it works and
//! after it replies, until the model or the user stops them or the conversation closes.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::AsyncReadExt;

/// Output kept per command, from its end.
const KEEP_BYTES: usize = 200 * 1024;
/// How long a new command runs before its first output is reported.
const FIRST_WAIT: Duration = Duration::from_secs(5);

pub type Notify = Arc<dyn Fn(Vec<(u32, String)>) + Send + Sync>;

#[derive(Default)]
struct Output {
    text: String,
    /// Bytes dropped from the front, so positions stay comparable after trimming.
    dropped: usize,
    /// The end of what the model has read, as a position in all output so far.
    read: usize,
    exit: Option<Option<i32>>,
}

struct Job {
    id: u32,
    command: String,
    pid: Option<u32>,
    output: Arc<Mutex<Output>>,
    watcher: tokio::task::JoinHandle<()>,
    #[cfg(windows)]
    _job: Option<super::sandbox::Job>,
}

impl Drop for Job {
    fn drop(&mut self) {
        kill_tree(self.pid);
        self.watcher.abort();
    }
}

fn kill_tree(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("taskkill").args(["/T", "/F", "/PID", &pid.to_string()]).creation_flags(0x0800_0000).output();
    }
    #[cfg(unix)]
    let _ = std::process::Command::new("kill").args(["-TERM", &format!("-{pid}")]).output();
}

pub struct Jobs {
    list: Mutex<Vec<Job>>,
    next: AtomicU32,
    notify: Notify,
}

impl Jobs {
    pub fn new(notify: Notify) -> Jobs {
        Jobs { list: Mutex::new(Vec::new()), next: AtomicU32::new(1), notify }
    }

    /// Starts a command and reports its first output, or all of it when it ends within a few seconds. The command gets
    /// a free port in PORT, so servers from several tests do not collide.
    pub async fn start(self: &Arc<Self>, mut cmd: tokio::process::Command, command: &str, memory_cap: u64, max_output: usize) -> Result<String, String> {
        let port = free_port();
        cmd.env("PORT", port.to_string());
        cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| format!("Could not start the shell: {e}"))?;
        let pid = child.id();
        #[cfg(windows)]
        let job = super::sandbox::Job::new(memory_cap).filter(|j| pid.is_some_and(|p| j.assign(p)));
        #[cfg(unix)]
        let _ = memory_cap;
        let output: Arc<Mutex<Output>> = Arc::default();
        let pump = |mut stream: Box<dyn tokio::io::AsyncRead + Unpin + Send>| {
            let output = output.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                while let Ok(n) = stream.read(&mut buf).await {
                    if n == 0 {
                        break;
                    }
                    let mut out = output.lock().unwrap();
                    out.text.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if out.text.len() > KEEP_BYTES {
                        let cut = floor_char(&out.text, out.text.len() - KEEP_BYTES / 2);
                        out.text.drain(..cut);
                        out.dropped += cut;
                    }
                }
            })
        };
        let a = pump(Box::new(child.stdout.take().unwrap()));
        let b = pump(Box::new(child.stderr.take().unwrap()));
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let watcher = {
            let output = output.clone();
            let jobs = Arc::downgrade(self);
            tokio::spawn(async move {
                let status = child.wait().await.ok();
                let _ = tokio::time::timeout(Duration::from_secs(2), async {
                    let _ = a.await;
                    let _ = b.await;
                })
                .await;
                output.lock().unwrap().exit = Some(status.and_then(|s| s.code()));
                if let Some(jobs) = jobs.upgrade() {
                    jobs.changed();
                }
            })
        };
        self.list.lock().unwrap().push(Job {
            id,
            command: command.to_string(),
            pid,
            output: output.clone(),
            watcher,
            #[cfg(windows)]
            _job: job,
        });
        self.changed();
        let start = std::time::Instant::now();
        while start.elapsed() < FIRST_WAIT && output.lock().unwrap().exit.is_none() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let (text, exit) = self.take_new(id, max_output);
        let busy = PORT_IN_USE.is_match(&text).then(|| {
            format!(
                "\n\nThe server's port is already in use, probably by a server from another test. Port {} is free, so start it there, or use $PORT in the command, which Scoobert sets to a free port. Stop an old server of yours with job_stop.\n{}",
                free_port(),
                self.describe()
            )
        });
        match exit {
            Some(code) => {
                self.remove(id);
                let mut text = if text.trim().is_empty() { "(No output.)".to_string() } else { text };
                text.push_str(busy.as_deref().unwrap_or_default());
                if let Some(code) = code.filter(|&c| c != 0) {
                    text.push_str(&format!("\n[Exit code {code}]"));
                    return Err(text);
                }
                Ok(text)
            }
            None => {
                let mut out = format!("Job {id} is running `{}` in the background with PORT set to {port}.", crate::util::clip(command, 120));
                if let Some(address) = local_address(&text) {
                    out.push_str(&format!(" It printed {address}, which browser_open can open."));
                }
                out.push_str(&format!(
                    " Read its new output with job_output, and stop it with job_stop when you no longer need it. It keeps running after your reply while this conversation stays open, so the user can use it.\n\nOutput so far:\n{}",
                    if text.trim().is_empty() { "(None yet.)" } else { &text }
                ));
                out.push_str(busy.as_deref().unwrap_or_default());
                Ok(out)
            }
        }
    }

    /// New output since the model last read it, after waiting up to `wait` for some to arrive or for the command to end.
    pub async fn output(&self, id: u32, wait: Duration, max_output: usize) -> Result<String, String> {
        let output = self.find(id)?;
        let start = std::time::Instant::now();
        while start.elapsed() < wait {
            let ready = {
                let o = output.lock().unwrap();
                o.exit.is_some() || o.dropped + o.text.len() > o.read
            };
            if ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let (text, exit) = self.take_new(id, max_output);
        let mut out = if text.trim().is_empty() { "(No new output.)".to_string() } else { text };
        match exit {
            Some(code) => {
                self.remove(id);
                out.push_str(&match code {
                    Some(code) => format!("\n[Job {id} ended with exit code {code}.]"),
                    None => format!("\n[Job {id} ended.]"),
                });
            }
            None => out.push_str(&format!("\n[Job {id} is still running.]")),
        }
        Ok(out)
    }

    pub fn stop(&self, id: u32, max_output: usize) -> Result<String, String> {
        self.find(id)?;
        let (text, _) = self.take_new(id, max_output);
        self.remove(id);
        Ok(if text.trim().is_empty() { format!("Stopped job {id}.") } else { format!("Stopped job {id}. Its last output:\n{text}") })
    }

    /// Stops a command the user chose to stop.
    pub fn stop_quietly(&self, id: u32) {
        self.remove(id);
    }

    pub fn stop_all(&self) {
        let stopped: Vec<Job> = std::mem::take(&mut *self.list.lock().unwrap());
        if !stopped.is_empty() {
            drop(stopped);
            self.changed();
        }
    }

    /// The running commands, for the window.
    pub fn running(&self) -> Vec<(u32, String)> {
        self.list.lock().unwrap().iter().filter(|j| j.output.lock().unwrap().exit.is_none()).map(|j| (j.id, j.command.clone())).collect()
    }

    /// The jobs the model started, for the tool result that lists them.
    pub fn describe(&self) -> String {
        let list = self.list.lock().unwrap();
        if list.is_empty() {
            return "No background jobs are running.".into();
        }
        list.iter().map(|j| format!("Job {}: {}", j.id, crate::util::clip(&j.command, 120))).collect::<Vec<_>>().join("\n")
    }

    fn find(&self, id: u32) -> Result<Arc<Mutex<Output>>, String> {
        let list = self.list.lock().unwrap();
        list.iter().find(|j| j.id == id).map(|j| j.output.clone()).ok_or_else(|| {
            let ids: Vec<String> = list.iter().map(|j| j.id.to_string()).collect();
            if ids.is_empty() { format!("There is no job {id}, and no background jobs are running.") } else { format!("There is no job {id}. The running jobs are {}.", ids.join(", ")) }
        })
    }

    fn take_new(&self, id: u32, max_output: usize) -> (String, Option<Option<i32>>) {
        let Ok(output) = self.find(id) else { return (String::new(), Some(None)) };
        let mut o = output.lock().unwrap();
        let from = floor_char(&o.text, o.read.saturating_sub(o.dropped).min(o.text.len()));
        let mut text = o.text[from..].to_string();
        o.read = o.dropped + o.text.len();
        if text.len() > max_output {
            let cut = floor_char(&text, text.len() - max_output);
            text = format!("[Showing the last {} bytes.]\n{}", text.len() - cut, &text[cut..]);
        }
        (text, o.exit)
    }

    fn remove(&self, id: u32) {
        let removed: Vec<Job> = {
            let mut list = self.list.lock().unwrap();
            let (gone, kept) = std::mem::take(&mut *list).into_iter().partition(|j| j.id == id);
            *list = kept;
            gone
        };
        drop(removed);
        self.changed();
    }

    fn changed(&self) {
        (self.notify)(self.running());
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        self.list.get_mut().unwrap().clear();
    }
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// A free port from a random start between 20000 and 29999, counting up past taken ones. Above the usual
/// development ports and below the range Windows hands out on its own.
pub fn free_port() -> u16 {
    let start: u16 = rand::random_range(20000..30000);
    (0..2000u16)
        .map(|i| 20000 + (start - 20000 + i) % 10000)
        // Only loopback is tried, since listening on every interface makes Windows ask about its firewall.
        .find(|&p| std::net::TcpListener::bind(("127.0.0.1", p)).is_ok())
        .unwrap_or(start)
}

/// The errors servers print when their port is taken, from Node, Python on Linux, macOS, and Windows, and others.
static PORT_IN_USE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?i)EADDRINUSE|address already in use|only one usage of each socket address|port \d+ is (already )?in use|WinError 10048|Errno 98\b|Errno 48\b").unwrap()
});

/// A web address on this computer that a server printed, such as http://localhost:5173/.
fn local_address(text: &str) -> Option<String> {
    use std::sync::LazyLock;
    static ADDRESS: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1?\])(?::\d+)?[^\s'\x22<>]*").unwrap());
    ADDRESS.find(text).map(|m| m.as_str().replace("0.0.0.0", "localhost").trim_end_matches(['.', ',', ')']).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_address_a_server_prints() {
        assert_eq!(local_address("  VITE ready\n  Local:   http://localhost:5173/\n").as_deref(), Some("http://localhost:5173/"));
        assert_eq!(local_address("Serving on http://0.0.0.0:8000.").as_deref(), Some("http://localhost:8000"));
        assert_eq!(local_address("listening on port 3000"), None);
    }

    #[test]
    fn spots_a_port_in_use_and_finds_a_free_one() {
        assert!(PORT_IN_USE.is_match("Error: listen EADDRINUSE: address already in use :::3000"));
        assert!(PORT_IN_USE.is_match("OSError: [WinError 10048] Only one usage of each socket address"));
        assert!(!PORT_IN_USE.is_match("Serving HTTP on 127.0.0.1 port 8000"));
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = free_port();
        assert!((20000..30000).contains(&port) && port != taken.local_addr().unwrap().port());
    }
}
