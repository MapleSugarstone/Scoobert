//! Checks GitHub once a day for a newer release. It only tells the user; it never installs anything.

use std::time::Duration;

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

/// Returns the newer version and its release page, or `None` when this build is current.
pub async fn check() -> Option<(String, String)> {
    let repo = repository()?;
    let client = reqwest::Client::builder().user_agent(concat!("Scoobert/", env!("CARGO_PKG_VERSION"))).build().ok()?;
    let res = client
        .get(format!("https://api.github.com/repos/{repo}/releases/latest"))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .ok()?;
    let body: serde_json::Value = res.json().await.ok()?;
    let tag = body["tag_name"].as_str()?.trim_start_matches('v').to_string();
    let url = body["html_url"].as_str().filter(|u| u.starts_with("https://github.com/"))?.to_string();
    is_newer(&tag, env!("CARGO_PKG_VERSION")).then_some((tag, url))
}

fn is_newer(candidate: &str, current: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> { v.split(['.', '-']).take(3).map(|p| p.parse().unwrap_or(0)).collect() };
    parse(candidate) > parse(current)
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
