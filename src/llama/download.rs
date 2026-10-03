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

/// One size a repository offers, for the user to pick when the download did not say which.
#[derive(Debug, Clone, PartialEq)]
pub struct SizeChoice {
    /// The number format, such as Q4_K_M, or the file's name when two models in the repository share a format.
    pub label: String,
    pub bytes: u64,
    /// What to download for this size.
    pub spec: String,
}

/// The repository holds several sizes of the model and the download did not name one it has.
#[derive(Debug, Clone, PartialEq)]
pub struct ChooseSize {
    pub message: String,
    /// Smallest first.
    pub sizes: Vec<SizeChoice>,
}

impl std::fmt::Display for ChooseSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ChooseSize {}

/// What a download names: a repository, with a number format or one file in it.
#[derive(Debug, PartialEq)]
struct Wanted {
    repo: String,
    revision: Option<String>,
    quant: Option<String>,
    file: Option<String>,
}

/// Reads "owner/repository", "owner/repository:QUANT", or any Hugging Face address of the repository or of a file in
/// it, as a browser shows them.
fn parse_wanted(spec: &str) -> Option<Wanted> {
    static ADDRESS: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"^(?:https?://)?(?:www\.)?(?:(?:huggingface\.co|hf\.co)/)?([\w.-]+/[\w.-]+?)(?:/(?:tree|blob|resolve)/([\w.-]+)(?:/([^?#:]*))?)?/?(?::([\w.-]+))?(?:[?#].*)?$",
        )
        .unwrap()
    });
    let m = ADDRESS.captures(spec.trim())?;
    let path = m.get(3).map(|p| p.as_str().trim_matches('/').to_string()).filter(|p| !p.is_empty());
    let (file, folder) = match path {
        Some(p) if p.to_lowercase().ends_with(".gguf") => (Some(p), None),
        // A folder such as tree/main/Q4_K_M names the format.
        Some(p) => (None, p.rsplit('/').next().map(str::to_string)),
        None => (None, None),
    };
    Some(Wanted { repo: m[1].to_string(), revision: m.get(2).map(|r| r.as_str().to_string()), quant: m.get(4).map(|q| q.as_str().to_string()).or(folder), file })
}

/// The number format at the end of a model file's name, or the name itself when it shows none.
fn quant_label(stem: &str) -> String {
    super::QUANT_SUFFIX.find(stem).map(|m| m.as_str()[1..].to_uppercase()).unwrap_or_else(|| stem.to_string())
}

/// Downloads a model from Hugging Face into its own folder under `models_dir`, with the image projector when
/// `projector` is true. `revision` pins a repository commit unless the address names another. Without a number
/// format the download takes the repository's only one, and otherwise fails with `ChooseSize`.
pub async fn download(
    http: &reqwest::Client,
    models_dir: &Path,
    spec: &str,
    projector: bool,
    revision: &str,
    cancel: &CancellationToken,
    on_progress: impl Fn(Progress),
) -> anyhow::Result<PathBuf> {
    let Some(wanted) = parse_wanted(spec) else {
        bail!(tr("Give a Hugging Face model as its web address or as owner/repository, for example unsloth/Qwen3.5-9B-GGUF."));
    };
    let revision = wanted.revision.as_deref().unwrap_or(revision);
    if !regex::Regex::new(r"^[\w.-]+$").unwrap().is_match(revision) {
        bail!(tr("Invalid repository revision."));
    }
    let repo = wanted.repo.clone();
    let files = list_ggufs(http, &repo, revision).await?;
    // Each model is its files with one name, since a split model's parts differ only in their part numbers.
    let mut groups: std::collections::BTreeMap<String, Vec<&TreeEntry>> = std::collections::BTreeMap::new();
    for f in files.iter().filter(|f| !base(&f.path).to_lowercase().contains("mmproj")) {
        groups.entry(model_stem(&base(&f.path))).or_default().push(f);
    }
    if groups.is_empty() {
        bail!(trf("{repo} has no GGUF model files.", &[("repo", &repo)]));
    }
    let chosen: Vec<&String> = if let Some(file) = &wanted.file {
        groups.iter().filter(|(_, fs)| fs.iter().any(|f| f.path == *file || base(&f.path) == base(file))).map(|(stem, _)| stem).collect()
    } else if let Some(quant) = &wanted.quant {
        // Q4_K_XL also finds UD-Q4_K_XL, the name some repositories give their own versions of a format.
        let matches = |stem: &str| {
            let label = quant_label(stem);
            label.eq_ignore_ascii_case(quant) || label.strip_prefix("UD-").is_some_and(|l| l.eq_ignore_ascii_case(quant))
        };
        groups.keys().filter(|stem| matches(stem)).collect()
    } else if groups.len() == 1 {
        groups.keys().collect()
    } else {
        // The user picks among the sizes, since the right one depends on the computer's memory.
        Vec::new()
    };
    let [stem] = chosen.as_slice() else {
        let labels: Vec<String> = groups.keys().map(|s| quant_label(s)).collect();
        let unique = labels.iter().collect::<std::collections::HashSet<_>>().len() == labels.len();
        let mut sizes: Vec<SizeChoice> = groups
            .iter()
            .map(|(stem, fs)| {
                let label = if unique { quant_label(stem) } else { stem.clone() };
                let spec = if unique { format!("{repo}:{label}") } else { format!("https://huggingface.co/{repo}/blob/{revision}/{}", fs[0].path) };
                SizeChoice { label, bytes: fs.iter().map(|f| f.size).sum(), spec }
            })
            .collect();
        sizes.sort_by_key(|c| c.bytes);
        let message = match (&wanted.quant, &wanted.file) {
            (_, Some(file)) => trf("{repo} has no file {file}. Pick one of its sizes.", &[("repo", &repo), ("file", &base(file))]),
            (Some(quant), None) => trf("{repo} has no {quant} file. Pick one of its sizes.", &[("repo", &repo), ("quant", &quant.to_uppercase())]),
            (None, None) => trf("{repo} comes in several sizes. Pick one.", &[("repo", &repo)]),
        };
        return Err(ChooseSize { message, sizes }.into());
    };
    let mut weights = groups[*stem].clone();
    weights.sort_by(|a, b| a.path.cmp(&b.path));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_every_form_of_a_model_address() {
        let w = parse_wanted("unsloth/Qwen3.5-9B-GGUF:Q4_K_M").unwrap();
        assert_eq!((w.repo.as_str(), w.quant.as_deref()), ("unsloth/Qwen3.5-9B-GGUF", Some("Q4_K_M")));
        let w = parse_wanted("https://huggingface.co/bartowski/Some-Model-GGUF/").unwrap();
        assert_eq!((w.repo.as_str(), w.quant, w.file), ("bartowski/Some-Model-GGUF", None, None));
        let w = parse_wanted("https://huggingface.co/bartowski/Some-Model-GGUF/blob/main/Some-Model-Q5_K_M.gguf?download=true").unwrap();
        assert_eq!((w.revision.as_deref(), w.file.as_deref()), (Some("main"), Some("Some-Model-Q5_K_M.gguf")));
        let w = parse_wanted("hf.co/unsloth/Big-GGUF/tree/main/UD-Q4_K_XL").unwrap();
        assert_eq!(w.quant.as_deref(), Some("UD-Q4_K_XL"));
        assert!(parse_wanted("not a model").is_none());
    }

    #[test]
    fn finds_the_format_in_each_naming_style() {
        assert_eq!(quant_label("Qwen3.5-9B-Q4_K_M"), "Q4_K_M");
        assert_eq!(quant_label("llama-2-7b.Q8_0"), "Q8_0");
        assert_eq!(quant_label("Qwen3.8-27B-UD-IQ4_XS"), "UD-IQ4_XS");
        assert_eq!(quant_label("model"), "model");
    }
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
