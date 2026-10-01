use std::time::{SystemTime, UNIX_EPOCH};

/// The user stopped the operation.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Stopped.")
    }
}

impl std::error::Error for Cancelled {}

pub fn is_cancelled(err: &anyhow::Error) -> bool {
    err.downcast_ref::<Cancelled>().is_some()
}

pub fn now_millis() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

pub fn random_hex(bytes: usize) -> String {
    let data: Vec<u8> = (0..bytes).map(|_| rand::random::<u8>()).collect();
    hex::encode(data)
}

pub fn sha256_hex(text: &str) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(text.as_bytes()))
}

/// Shortens text to `max` characters, ending with "..." when cut.
pub fn clip(text: &str, max: usize) -> String {
    crate::notes::clip(text, max)
}

pub fn gb(bytes: u64) -> String {
    let n = format!("{:.1}", bytes as f64 / 1e9).replace('.', &crate::i18n::current().decimal.to_string());
    crate::i18n::trf("{size} GB", &[("size", &n)])
}

/// "2.1K" style token counts for the context meter.
pub fn short_count(n: u64) -> String {
    // Context sizes are powers of two and read as 32K rather than 32.8K.
    if n >= 1024 && n % 1024 == 0 {
        format!("{}K", n / 1024)
    } else if n >= 1000 {
        format!("{:.1}K", n as f64 / 1000.0).replace(".0K", "K")
    } else {
        n.to_string()
    }
}

const MINUTE: i64 = 60;
const HOUR: i64 = 60 * MINUTE;
const DAY: i64 = 24 * HOUR;
const WEEK: i64 = 7 * DAY;
const MONTH: i64 = 30 * DAY;
const YEAR: i64 = 365 * DAY;

/// The largest whole unit of an age in seconds, as (count, index into the unit names).
fn age_unit(secs: i64) -> Option<(i64, usize)> {
    [(YEAR, 5), (MONTH, 4), (WEEK, 3), (DAY, 2), (HOUR, 1), (MINUTE, 0)].iter().find(|(size, _)| secs >= *size).map(|&(size, i)| (secs / size, i))
}

/// A short age for lists, such as "5 min ago" or "3 wk ago".
pub fn ago(millis: i64) -> String {
    use crate::i18n::key;
    const UNITS: [&str; 6] = [key("{n} min ago"), key("{n} h ago"), key("{n} d ago"), key("{n} wk ago"), key("{n} mo ago"), key("{n} yr ago")];
    match age_unit((now_millis() - millis).max(0) / 1000) {
        Some((n, i)) => crate::i18n::trf(UNITS[i], &[("n", &n)]),
        None => crate::i18n::tr("just now").into(),
    }
}

/// A spelled-out age, such as "5 minutes ago" or "2 years ago".
pub fn ago_long(millis: i64) -> String {
    use crate::i18n::key;
    const ONE: [&str; 6] = [key("1 minute ago"), key("1 hour ago"), key("1 day ago"), key("1 week ago"), key("1 month ago"), key("1 year ago")];
    const MANY: [&str; 6] = [key("{n} minutes ago"), key("{n} hours ago"), key("{n} days ago"), key("{n} weeks ago"), key("{n} months ago"), key("{n} years ago")];
    match age_unit((now_millis() - millis).max(0) / 1000) {
        Some((1, i)) => crate::i18n::tr(ONE[i]).into(),
        Some((n, i)) => crate::i18n::trf(MANY[i], &[("n", &n)]),
        None => crate::i18n::tr("Just now").into(),
    }
}

/// The local date and time in the interface language's format, such as "Sep 30, 2026, 8:41 AM".
pub fn date_time(millis: i64) -> String {
    chrono::DateTime::from_timestamp_millis(millis)
        .map(|d| d.with_timezone(&chrono::Local).format(crate::i18n::current().date_time).to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_use_the_largest_whole_unit() {
        let now = now_millis();
        assert_eq!(ago_long(now - 20_000), "Just now");
        assert_eq!(ago_long(now - 60_000), "1 minute ago");
        assert_eq!(ago_long(now - 3 * HOUR * 1000), "3 hours ago");
        assert_eq!(ago_long(now - 10 * DAY * 1000), "1 week ago");
        assert_eq!(ago_long(now - 65 * DAY * 1000), "2 months ago");
        assert_eq!(ago_long(now - 800 * DAY * 1000), "2 years ago");
        assert_eq!(ago(now - 15 * DAY * 1000), "2 wk ago");
    }
}

/// A count with the interface language's separator between groups of three digits, such as 12,345.
pub fn thousands(n: u64) -> String {
    let group = crate::i18n::current().group;
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(group);
        }
        out.push(ch);
    }
    out
}

/// A rough length of time for a progress line, such as about 9 minutes.
pub fn about_duration(secs: u64) -> String {
    use crate::i18n::{tr, trf};
    match secs {
        0..50 => tr("less than a minute").into(),
        50..90 => tr("about a minute").into(),
        90..3_300 => trf("about {n} minutes", &[("n", &((secs + 30) / 60))]),
        _ => {
            let hours = format!("{:.1}", secs as f64 / 3600.0).replace('.', &crate::i18n::current().decimal.to_string());
            trf("about {n} hours", &[("n", &hours)])
        }
    }
}
