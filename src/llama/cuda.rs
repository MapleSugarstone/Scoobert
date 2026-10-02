//! NVIDIA support: llama.cpp's CUDA build and NVIDIA's CUDA runtime libraries, downloaded on request into Scoobert's
//! data folder for a computer with an NVIDIA card. Both come from the llama.cpp release Scoobert bundles, so the
//! CUDA server takes the same arguments as the bundled one.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, bail};
use futures::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::i18n::{tr, trf};
use crate::paths;
use crate::util::Cancelled;

/// The oldest NVIDIA driver that runs CUDA 12 builds through CUDA's minor version compatibility.
pub const MIN_DRIVER: u32 = 528;
/// CUDA 13 needs this driver, and it leaves out cards older than the Turing generation, which CUDA 12 still runs.
const CUDA_13_DRIVER: u32 = 580;

#[derive(Debug, Clone, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub driver: String,
}

impl Gpu {
    fn driver_major(&self) -> u32 {
        self.driver.split('.').next().and_then(|m| m.trim().parse().ok()).unwrap_or(0)
    }

    /// The CUDA version of the build that fits this driver, or None when the driver is too old for any.
    pub fn cuda(&self) -> Option<&'static str> {
        let major = self.driver_major();
        if major >= CUDA_13_DRIVER {
            Some("13.4")
        } else if major >= MIN_DRIVER {
            Some(if cfg!(windows) { "12.4" } else { "12.8" })
        } else {
            None
        }
    }

    /// The approximate download for that build, from the sizes llama.cpp published for b11193.
    pub fn download_size(&self) -> u64 {
        match (self.cuda(), cfg!(windows)) {
            (Some("13.4"), true) => 550_000_000,
            (Some("13.4"), false) => 565_000_000,
            (_, true) => 625_000_000,
            (_, false) => 730_000_000,
        }
    }
}

static DETECTED: OnceLock<Option<Gpu>> = OnceLock::new();

/// Looks for an NVIDIA card with `nvidia-smi`, which comes with the driver. Runs once and can take a moment, so
/// the app calls it on a background thread at startup.
pub fn detect() -> Option<Gpu> {
    DETECTED
        .get_or_init(|| {
            if cfg!(target_os = "macos") {
                return None;
            }
            let mut cmd = std::process::Command::new("nvidia-smi");
            cmd.args(["--query-gpu=name,driver_version", "--format=csv,noheader"]).stdin(std::process::Stdio::null());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x0800_0000);
            }
            let out = cmd.output().ok().filter(|o| o.status.success())?;
            let text = String::from_utf8_lossy(&out.stdout);
            let (name, driver) = text.lines().next()?.split_once(',')?;
            Some(Gpu { name: name.trim().to_string(), driver: driver.trim().to_string() })
        })
        .clone()
}

/// The card `detect` found, without waiting for it: None while it still looks, and Some(None) when there is none.
pub fn detected() -> Option<Option<Gpu>> {
    DETECTED.get().cloned()
}

fn dir() -> PathBuf {
    paths::get().data.join("llama-cuda")
}

/// The CUDA build's server, when NVIDIA support is installed.
pub fn server() -> Option<PathBuf> {
    Some(dir().join(paths::llama_server_name())).filter(|p| p.is_file())
}

/// The CUDA version the installed support was built for.
pub fn installed_version() -> Option<String> {
    server()?;
    let text = std::fs::read_to_string(dir().join("CUDA.txt")).ok()?;
    text.lines().next().and_then(|l| l.strip_prefix("CUDA ")).map(|v| v.split_whitespace().next().unwrap_or(v).to_string())
}

/// The llama.cpp release Scoobert bundles, from the address in the bundled VERSION.txt.
fn release() -> Option<String> {
    let text = std::fs::read_to_string(paths::bundled_llama_dir()?.join("VERSION.txt")).ok()?;
    let re = regex::Regex::new(r"/releases/download/(b\d+)/").unwrap();
    re.captures(&text).map(|c| c[1].to_string())
}

/// Removes NVIDIA support. The caller stops the server first, since it may be running from this folder.
pub fn remove() -> anyhow::Result<()> {
    let dir = dir();
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| trf("Could not remove {path}", &[("path", &dir.display())]))?;
    }
    Ok(())
}

struct Asset {
    name: String,
    url: String,
    size: u64,
    sha256: Option<String>,
}

/// Downloads the CUDA build and runtime that fit `gpu`, checks them, and unpacks them into the data folder,
/// reporting bytes done and the total.
pub async fn install(gpu: &Gpu, cancel: &CancellationToken, on_progress: impl Fn(u64, u64)) -> anyhow::Result<()> {
    let cuda = gpu.cuda().context(trf(
        "NVIDIA support needs driver version {min} or later, and this computer has {driver}. Update the driver from NVIDIA's website first.",
        &[("min", &MIN_DRIVER), ("driver", &gpu.driver)],
    ))?;
    let tag = release().context(tr("Could not tell which llama.cpp release Scoobert includes. Reinstall Scoobert, then try again."))?;
    let (server_name, runtime_name) = if cfg!(windows) {
        (format!("llama-{tag}-bin-win-cuda-{cuda}-x64.zip"), format!("cudart-llama-bin-win-cuda-{cuda}-x64.zip"))
    } else {
        (format!("llama-{tag}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"), format!("cudart-llama-{tag}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"))
    };
    let http = reqwest::Client::builder().user_agent(concat!("Scoobert/", env!("CARGO_PKG_VERSION"))).build()?;
    let body: serde_json::Value = http
        .get(format!("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"))
        .send()
        .await
        .context(tr("Could not reach GitHub."))?
        .error_for_status()?
        .json()
        .await?;
    let find = |name: &str| -> anyhow::Result<Asset> {
        let a = body["assets"].as_array().into_iter().flatten().find(|a| a["name"] == name).context(trf(
            "llama.cpp {release} has no {file} to download.",
            &[("release", &tag), ("file", &name)],
        ))?;
        Ok(Asset {
            name: name.to_string(),
            url: a["browser_download_url"].as_str().filter(|u| u.starts_with("https://github.com/")).context("The download has no address")?.to_string(),
            size: a["size"].as_u64().unwrap_or(0),
            sha256: a["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).map(str::to_lowercase),
        })
    };
    let assets = [find(&server_name)?, find(&runtime_name)?];
    let total: u64 = assets.iter().map(|a| a.size).sum();
    if let Some(free) = crate::sys::free_space(&paths::get().data)
        && free < total * 3
    {
        bail!(trf(
            "NVIDIA support needs about {need} of disk space to download and unpack, and {free} is free.",
            &[("need", &crate::util::gb(total * 3)), ("free", &crate::util::gb(free))]
        ));
    }
    let work = paths::get().data.join("llama-cuda.download");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)?;
    let result = async {
        let mut done = 0;
        let mut archives = Vec::new();
        for asset in &assets {
            let file = work.join(&asset.name);
            fetch(&http, asset, &file, cancel, |n| on_progress(done + n, total)).await?;
            done += asset.size;
            archives.push(file);
        }
        let staged = work.join("llama-cuda");
        std::fs::create_dir_all(&staged)?;
        for (i, archive) in archives.iter().enumerate() {
            let out = work.join(format!("unpacked-{i}"));
            std::fs::create_dir_all(&out)?;
            unpack(archive, &out)?;
            let _ = std::fs::remove_file(archive);
            // The server archive keeps its programs in a folder of its own, and the runtime archive is flat. Both go
            // into one folder, so the server finds the runtime libraries beside it.
            let from = if i == 0 { find_server_dir(&out).context(tr("The download holds no llama-server."))? } else { out.clone() };
            move_files(&from, &staged)?;
        }
        std::fs::write(staged.join("CUDA.txt"), format!("CUDA {cuda} for llama.cpp {tag} on {}\n", gpu.name))?;
        remove()?;
        std::fs::rename(&staged, dir())?;
        anyhow::Ok(())
    }
    .await;
    let _ = std::fs::remove_dir_all(&work);
    result
}

async fn fetch(http: &reqwest::Client, asset: &Asset, file: &Path, cancel: &CancellationToken, on_progress: impl Fn(u64)) -> anyhow::Result<()> {
    let res = http.get(&asset.url).send().await?.error_for_status()?;
    let mut out = tokio::fs::File::create(file).await?;
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut done = 0u64;
    let mut stream = res.bytes_stream();
    loop {
        let chunk = tokio::select! {
            c = stream.next() => c,
            _ = cancel.cancelled() => return Err(Cancelled.into()),
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk?;
        sha2::Digest::update(&mut hasher, &chunk);
        out.write_all(&chunk).await?;
        done += chunk.len() as u64;
        on_progress(done);
    }
    out.flush().await?;
    let actual = hex::encode(sha2::Digest::finalize(hasher));
    if let Some(expected) = &asset.sha256
        && *expected != actual
    {
        bail!(trf("{name} did not match its published checksum and was deleted. Try the download again.", &[("name", &asset.name)]));
    }
    Ok(())
}

/// Unpacks a .zip or .tar.gz with the system's tar, which Windows 10 and 11 include and which reads both.
fn unpack(archive: &Path, out: &Path) -> anyhow::Result<()> {
    let tar = if cfg!(windows) {
        std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| "C:\\Windows".into()).join("System32").join("tar.exe")
    } else {
        PathBuf::from("tar")
    };
    let mut cmd = std::process::Command::new(tar);
    cmd.arg("-xf").arg(archive).arg("-C").arg(out).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let status = cmd.status().context(tr("Could not unpack the download."))?;
    if !status.success() {
        bail!(tr("Could not unpack the download."));
    }
    Ok(())
}

fn find_server_dir(root: &Path) -> Option<PathBuf> {
    if root.join(paths::llama_server_name()).is_file() {
        return Some(root.to_path_buf());
    }
    std::fs::read_dir(root).ok()?.flatten().filter(|e| e.path().is_dir()).find_map(|e| find_server_dir(&e.path()))
}

/// Moves every file under `from` into `to`, flattened. Renames keep the links between Linux library versions.
fn move_files(from: &Path, to: &Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(from)?.flatten() {
        let path = entry.path();
        let kind = std::fs::symlink_metadata(&path)?.file_type();
        if kind.is_dir() {
            move_files(&path, to)?;
        } else {
            std::fs::rename(&path, to.join(entry.file_name()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Gpu;

    #[test]
    fn the_driver_decides_the_cuda_build() {
        let gpu = |driver: &str| Gpu { name: "NVIDIA GeForce RTX 3060".into(), driver: driver.into() };
        assert_eq!(gpu("581.42").cuda(), Some("13.4"));
        assert_eq!(gpu("552.22").cuda(), Some(if cfg!(windows) { "12.4" } else { "12.8" }));
        assert_eq!(gpu("472.12").cuda(), None);
    }
}
