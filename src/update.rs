//! Checks GitHub once a day for a newer release, and when the user asks, downloads it, checks it against the
//! checksum GitHub publishes, and installs it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use futures::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::i18n::{tr, trf};

const DAY_MS: i64 = 24 * 60 * 60 * 1000;

/// The GitHub "owner/name" from the package's repository field, when one is set.
pub fn repository() -> Option<String> {
    let url = env!("CARGO_PKG_REPOSITORY");
    let rest = url.trim_end_matches('/').trim_end_matches(".git").strip_prefix("https://github.com/")?;
    let mut parts = rest.split('/');
    let (owner, name) = (parts.next()?, parts.next()?);
    (!owner.is_empty() && !name.is_empty()).then(|| format!("{owner}/{name}"))
}

pub fn due(last_checked: i64) -> bool {
    repository().is_some() && crate::util::now_millis() - last_checked > DAY_MS
}

#[derive(Debug, Clone, PartialEq)]
pub struct Release {
    pub version: String,
    pub page: String,
    /// The file this copy of Scoobert can install itself from, when there is one.
    pub asset: Option<Asset>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// How this copy of Scoobert was installed, which decides whether it can update itself.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    WindowsInstaller,
    AppImage(PathBuf),
}

pub fn install_kind() -> Option<Kind> {
    if cfg!(windows) && crate::paths::uninstaller().is_some() {
        return Some(Kind::WindowsInstaller);
    }
    std::env::var_os("APPIMAGE").map(PathBuf::from).filter(|p| p.is_file()).map(Kind::AppImage)
}

fn client() -> Option<reqwest::Client> {
    reqwest::Client::builder().user_agent(concat!("Scoobert/", env!("CARGO_PKG_VERSION"))).build().ok()
}

/// The newest release when it is newer than this build.
pub async fn check() -> anyhow::Result<Option<Release>> {
    let repo = repository().context(tr("This build of Scoobert has no release page to check."))?;
    let res = client()
        .context(tr("Could not start the update check."))?
        .get(format!("https://api.github.com/repos/{repo}/releases/latest"))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .context(tr("Could not reach GitHub."))?;
    // GitHub answers 404 until the first release is published.
    if res.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !res.status().is_success() {
        bail!(trf("GitHub returned {status} for the update check.", &[("status", &res.status())]));
    }
    let body: serde_json::Value = res.json().await.context(tr("GitHub sent a release description Scoobert could not read."))?;
    let version = body["tag_name"].as_str().context(tr("The latest release has no version."))?.trim_start_matches('v').to_string();
    let page = body["html_url"].as_str().filter(|u| u.starts_with("https://github.com/")).context(tr("The latest release has no page."))?.to_string();
    if !is_newer(&version, env!("CARGO_PKG_VERSION")) {
        return Ok(None);
    }
    let wanted = |name: &str| match install_kind() {
        Some(Kind::WindowsInstaller) => name == format!("Scoobert-Setup-{version}.exe"),
        Some(Kind::AppImage(_)) => name.ends_with("-x86_64.AppImage"),
        None => false,
    };
    let asset = body["assets"].as_array().into_iter().flatten().find_map(|a| {
        let name = a["name"].as_str()?;
        let url = a["browser_download_url"].as_str().filter(|u| u.starts_with("https://github.com/"))?;
        wanted(name).then(|| Asset {
            name: name.to_string(),
            url: url.to_string(),
            size: a["size"].as_u64().unwrap_or(0),
            sha256: a["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).map(str::to_lowercase),
        })
    });
    Ok(Some(Release { version, page, asset }))
}

fn is_newer(candidate: &str, current: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> { v.split(['.', '-']).take(3).map(|p| p.parse().unwrap_or(0)).collect() };
    parse(candidate) > parse(current)
}

/// Downloads the update, reporting bytes done and total, and checks it against GitHub's checksum.
pub async fn download(asset: &Asset, kind: &Kind, on_progress: impl Fn(u64, u64)) -> anyhow::Result<PathBuf> {
    let target = match kind {
        Kind::WindowsInstaller => std::env::temp_dir().join(&asset.name),
        // Next to the running AppImage, so the finished file can replace it with a rename.
        Kind::AppImage(current) => current.with_extension("AppImage.new"),
    };
    let res = client().context(tr("Could not start the download"))?.get(&asset.url).send().await?;
    if !res.status().is_success() {
        bail!(trf("GitHub returned {status} for the update.", &[("status", &res.status())]));
    }
    let total = res.content_length().unwrap_or(asset.size);
    let mut file = tokio::fs::File::create(&target).await.with_context(|| trf("Could not write {path}", &[("path", &target.display())]))?;
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut done = 0u64;
    let mut stream = res.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        sha2::Digest::update(&mut hasher, &chunk);
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        on_progress(done, total);
    }
    file.flush().await?;
    drop(file);
    let actual = hex::encode(sha2::Digest::finalize(hasher));
    if let Some(expected) = &asset.sha256
        && *expected != actual
    {
        let _ = tokio::fs::remove_file(&target).await;
        bail!(tr("The update did not match its published checksum and was deleted. Try again later."));
    }
    Ok(target)
}

/// Starts the new version. The Windows installer updates in place without its wizard and reopens Scoobert; an
/// AppImage replaces the running file and starts. The caller then closes this copy.
pub fn install(file: &Path, kind: &Kind) -> anyhow::Result<()> {
    match kind {
        Kind::WindowsInstaller => {
            std::process::Command::new(file).args(["/S", "/relaunch"]).spawn().context(tr("Could not start the installer"))?;
        }
        Kind::AppImage(current) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o755))?;
            }
            std::fs::rename(file, current).context(tr("Could not replace the AppImage"))?;
            std::process::Command::new(current).spawn().context(tr("Could not start the new version"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn compares_versions() {
        assert!(super::is_newer("0.2.0", "0.1.9"));
        assert!(super::is_newer("1.0.0", "0.9.9"));
        assert!(!super::is_newer("0.1.0", "0.1.0"));
    }
}
