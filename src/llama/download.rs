//! Downloads GGUF models from Hugging Face. A partial download continues where it stopped, and every file
//! is checked against the SHA-256 that Hugging Face publishes for it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use futures::StreamExt;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use super::model_stem;
use crate::i18n::{tr, trf};
use crate::util::Cancelled;

#[derive(Debug, Clone)]
pub struct Progress {
    pub status: String,
    pub done: u64,
    pub total: u64,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
}

/// Downloads "owner/repository:QUANT" into its own folder under `models_dir`, with the image projector when
/// `projector` is true. `revision` pins a repository commit.
pub async fn download(
    http: &reqwest::Client,
    models_dir: &Path,
    spec: &str,
    projector: bool,
    revision: &str,
    cancel: &CancellationToken,
    on_progress: impl Fn(Progress),
) -> anyhow::Result<PathBuf> {
    let re = regex::Regex::new(r"^(?:https://huggingface\.co/)?([\w.-]+/[\w.-]+)(?::([\w.-]+))?$").unwrap();
    let Some(m) = re.captures(spec.trim()) else {
        bail!(tr("Use the form \"owner/repository:quantization\", for example unsloth/Qwen3.5-9B-GGUF:Q4_K_M."));
    };
    if !regex::Regex::new(r"^[\w.-]+$").unwrap().is_match(revision) {
        bail!(tr("Invalid repository revision."));
    }
    let repo = m[1].to_string();
    let quant = m.get(2).map(|q| q.as_str()).unwrap_or("Q4_K_M").to_lowercase();
    let files = list_ggufs(http, &repo, revision).await?;
    let weights: Vec<&TreeEntry> = files
        .iter()
        .filter(|f| !f.path.to_lowercase().contains("mmproj") && model_stem(&base(&f.path)).to_lowercase().ends_with(&format!("-{quant}")))
        .collect();
    if weights.is_empty() {
        bail!(trf("{repo} has no {quant} file.", &[("repo", &repo), ("quant", &quant.to_uppercase())]));
    }
    let proj = if projector {
        files
            .iter()
            .find(|f| f.path.to_lowercase().ends_with("mmproj-f16.gguf"))
            .or_else(|| files.iter().find(|f| f.path.to_lowercase().contains("mmproj")))
    } else {
        None
    };
    let folder = models_dir.join(model_stem(&base(&weights[0].path)));
    tokio::fs::create_dir_all(&folder).await?;
    let queue: Vec<&TreeEntry> = weights.into_iter().chain(proj).collect();
    fetch(http, &repo, revision, &queue, &folder, cancel, &on_progress).await?;
    Ok(folder)
}

/// Downloads a LoRA adapter in GGUF format from "owner/repository", or from "owner/repository:part-of-name" when the
/// repository holds more than one, into `folder`.
pub async fn download_adapter(http: &reqwest::Client, spec: &str, folder: &Path, cancel: &CancellationToken, on_progress: impl Fn(Progress)) -> anyhow::Result<PathBuf> {
    let re = regex::Regex::new(r"^(?:https://huggingface\.co/)?([\w.-]+/[\w.-]+)(?::(.+))?$").unwrap();
    let Some(m) = re.captures(spec.trim()) else {
        bail!(tr("Give an adapter file, or a Hugging Face repository as owner/repository."));
    };
    let repo = m[1].to_string();
    let files = list_ggufs(http, &repo, "main").await?;
    let wanted = m.get(2).map(|p| p.as_str().to_lowercase());
    let matching: Vec<&TreeEntry> = files.iter().filter(|f| wanted.as_ref().is_none_or(|w| base(&f.path).to_lowercase().contains(w))).collect();
    let file = match matching.as_slice() {
        [one] => *one,
        [] => bail!(trf("{repo} has no matching GGUF file.", &[("repo", &repo)])),
        many => {
            let names: Vec<String> = many.iter().take(6).map(|f| base(&f.path)).collect();
            bail!(trf("{repo} has several GGUF files. Add part of one name after a colon, as in {repo}:{example}. The files: {names}", &[
                ("repo", &repo),
                ("example", &model_stem(&names[0])),
                ("names", &names.join(", ")),
            ]));
        }
    };
    fetch(http, &repo, "main", &[file], folder, cancel, &on_progress).await?;
    Ok(folder.join(base(&file.path)))
}

fn base(p: &str) -> String {
    p.rsplit('/').next().unwrap_or(p).to_string()
}

/// The GGUF files in a Hugging Face repository.
async fn list_ggufs(http: &reqwest::Client, repo: &str, revision: &str) -> anyhow::Result<Vec<TreeEntry>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true");
    let res = http.get(&url).send().await?;
    if !res.status().is_success() {
        bail!(trf("Hugging Face returned {status} for {repo}.", &[("status", &res.status()), ("repo", &repo)]));
    }
    Ok(res.json::<Vec<TreeEntry>>().await?.into_iter().filter(|f| f.kind == "file" && f.path.to_lowercase().ends_with(".gguf")).collect())
}

/// Downloads `queue` into `folder`, continuing partial files and checking each against its published checksum.
async fn fetch(
    http: &reqwest::Client,
    repo: &str,
    revision: &str,
    queue: &[&TreeEntry],
    folder: &Path,
    cancel: &CancellationToken,
    on_progress: &impl Fn(Progress),
) -> anyhow::Result<()> {
    tokio::fs::create_dir_all(folder).await?;
    let total: u64 = queue.iter().map(|f| f.size).sum();
    let mut done = 0u64;

    for f in queue.iter().copied() {
        let name = base(&f.path);
        let target = folder.join(&name);
        if tokio::fs::metadata(&target).await.map(|m| m.len() == f.size).unwrap_or(false) {
            done += f.size;
            continue;
        }
        let part = folder.join(format!("{name}.part"));
        let have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
        if have < f.size {
            let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{}", f.path);
            let mut req = http.get(&url);
            if have > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
            }
            let res = req.send().await?;
            if !res.status().is_success() {
                bail!(trf("Download of {name} failed with {status}.", &[("name", &name), ("status", &res.status())]));
            }
            let resumed = have > 0 && res.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            if resumed {
                done += have;
            }
            let mut out = tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(resumed)
                .truncate(!resumed)
                .open(&part)
                .await
                .with_context(|| trf("Could not write {path}", &[("path", &part.display())]))?;
            let mut stream = res.bytes_stream();
            let mut last = Instant::now();
            loop {
                let chunk = tokio::select! {
                    c = stream.next() => c,
                    _ = cancel.cancelled() => {
                        out.flush().await?;
                        return Err(Cancelled.into());
                    }
                };
                let Some(chunk) = chunk else { break };
                let chunk = chunk?;
                out.write_all(&chunk).await?;
                done += chunk.len() as u64;
                if last.elapsed() > Duration::from_millis(250) {
                    last = Instant::now();
                    on_progress(Progress { status: trf("Downloading {name}", &[("name", &name)]), done, total });
                }
            }
            out.flush().await?;
        } else {
            done += f.size;
        }
        let size = tokio::fs::metadata(&part).await?.len();
        if size != f.size {
            bail!(trf("{name} downloaded incompletely. Try again to continue it.", &[("name", &name)]));
        }
        if let Some(lfs) = &f.lfs {
            on_progress(Progress { status: trf("Checking {name}", &[("name", &name)]), done, total });
            let path = part.clone();
            let actual = tokio::task::spawn_blocking(move || sha256_file(&path)).await??;
            if actual != lfs.oid {
                let _ = tokio::fs::remove_file(&part).await;
                bail!(trf("{name} did not match its published checksum and was deleted. Try the download again.", &[("name", &name)]));
            }
        }
        tokio::fs::rename(&part, &target).await?;
    }
    on_progress(Progress { status: tr("Done").into(), done: total, total });
    Ok(())
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}
