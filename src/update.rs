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
    /// The .app folder this copy runs from on macOS.
    MacApp(PathBuf),
}

pub fn install_kind() -> Option<Kind> {
    if cfg!(windows) && crate::paths::uninstaller().is_some() {
        return Some(Kind::WindowsInstaller);
    }
    if cfg!(target_os = "macos") {
        return mac_bundle().map(Kind::MacApp);
    }
    std::env::var_os("APPIMAGE").map(PathBuf::from).filter(|p| p.is_file()).map(Kind::AppImage)
}

/// The .app folder around this program, which runs from its Contents/MacOS folder.
fn mac_bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let macos = exe.parent()?;
    let bundle = macos.parent()?.parent()?;
    (macos.ends_with("Contents/MacOS") && bundle.extension().is_some_and(|e| e == "app")).then(|| bundle.to_path_buf())
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
        Some(Kind::MacApp(_)) => name == format!("Scoobert-{version}-macos-arm64.dmg"),
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
        Kind::WindowsInstaller | Kind::MacApp(_) => std::env::temp_dir().join(&asset.name),
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
/// AppImage replaces the running file and starts; a Mac app is swapped for the one in the disk image and opens once
/// this copy has closed. The caller then closes this copy.
pub fn install(file: &Path, kind: &Kind) -> anyhow::Result<()> {
    match kind {
        Kind::MacApp(bundle) => install_mac(file, bundle)?,
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

/// Copies the app out of the disk image beside the running one, swaps the two, and leaves a shell that waits for
/// this copy to exit, deletes the old app, and opens the new one. The download comes from Scoobert itself rather
/// than a browser, so macOS does not ask the user to approve the new copy again.
fn install_mac(dmg: &Path, bundle: &Path) -> anyhow::Result<()> {
    use std::process::{Command, Stdio};
    let old = swap_mac_app(dmg, bundle)?;
    let wait = format!("while kill -0 {} 2>/dev/null; do sleep 0.5; done; rm -rf \"$1\"; open \"$2\"", std::process::id());
    let mut cmd = Command::new("/bin/sh");
    cmd.args(["-c", &wait, "sh"]).arg(&old).arg(bundle).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Its own group, so closing this copy does not end the shell that reopens Scoobert.
        cmd.process_group(0);
    }
    cmd.spawn().context(tr("Could not start the new version"))?;
    Ok(())
}

/// Puts the app from the disk image where `bundle` is, and returns where the old app went.
fn swap_mac_app(dmg: &Path, bundle: &Path) -> anyhow::Result<PathBuf> {
    use std::process::Command;
    let folder = bundle.parent().context(tr("Could not find the folder Scoobert is in."))?;
    let name = bundle.file_name().context(tr("Could not find the folder Scoobert is in."))?.to_string_lossy().into_owned();
    let staged = folder.join(format!("{name}.new"));
    let old = folder.join(format!("{name}.old"));
    let cannot_write = || trf("Could not put the new version in {folder}. Download it from the release page instead.", &[("folder", &folder.display())]);
    let mount = std::env::temp_dir().join(format!("scoobert-update-{}", std::process::id()));
    std::fs::create_dir_all(&mount)?;
    let attached = Command::new("/usr/bin/hdiutil").args(["attach", "-nobrowse", "-noautoopen", "-readonly", "-mountpoint"]).arg(&mount).arg(dmg).output()?;
    if !attached.status.success() {
        let _ = std::fs::remove_dir(&mount);
        bail!(tr("Could not open the update's disk image."));
    }
    let app = std::fs::read_dir(&mount)?.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "app"));
    let _ = std::fs::remove_dir_all(&staged);
    // ditto keeps the app's signatures and links as they are.
    let copied = app.is_some_and(|app| Command::new("/usr/bin/ditto").arg(app).arg(&staged).status().is_ok_and(|s| s.success()));
    let _ = Command::new("/usr/bin/hdiutil").args(["detach", "-quiet"]).arg(&mount).status();
    let _ = std::fs::remove_dir(&mount);
    let _ = std::fs::remove_file(dmg);
    if !copied {
        let _ = std::fs::remove_dir_all(&staged);
        bail!(cannot_write());
    }
    let _ = Command::new("/usr/bin/xattr").args(["-dr", "com.apple.quarantine"]).arg(&staged).status();
    let _ = std::fs::remove_dir_all(&old);
    if std::fs::rename(bundle, &old).is_err() {
        let _ = std::fs::remove_dir_all(&staged);
        bail!(cannot_write());
    }
    if std::fs::rename(&staged, bundle).is_err() {
        let _ = std::fs::rename(&old, bundle);
        bail!(cannot_write());
    }
    Ok(old)
}

#[cfg(all(test, target_os = "macos"))]
mod mac_tests {
    use std::process::Command;

    #[test]
    fn swaps_the_app_for_the_one_in_the_disk_image() {
        let root = std::env::temp_dir().join(format!("scoobert-swap-{}", std::process::id()));
        let app = |dir: &std::path::Path, text: &str| {
            let exe = dir.join("Scoobert.app/Contents/MacOS");
            std::fs::create_dir_all(&exe).unwrap();
            std::fs::write(exe.join("scoobert"), text).unwrap();
        };
        let (installed, image) = (root.join("Applications"), root.join("image"));
        app(&installed, "old");
        app(&image, "new");
        let dmg = root.join("update.dmg");
        let made = Command::new("/usr/bin/hdiutil").args(["create", "-quiet", "-fs", "HFS+", "-format", "UDZO", "-srcfolder"]).arg(&image).arg(&dmg).status().unwrap();
        assert!(made.success());
        let bundle = installed.join("Scoobert.app");
        let old = super::swap_mac_app(&dmg, &bundle).expect("the swap works");
        assert_eq!(std::fs::read_to_string(bundle.join("Contents/MacOS/scoobert")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(old.join("Contents/MacOS/scoobert")).unwrap(), "old");
        assert!(!installed.join("Scoobert.app.new").exists() && !dmg.exists());
        let _ = std::fs::remove_dir_all(&root);
    }
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
