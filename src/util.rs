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
    format!("{:.1} GB", bytes as f64 / 1e9)
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
    const UNITS: [&str; 6] = ["min", "h", "d", "wk", "mo", "yr"];
    match age_unit((now_millis() - millis).max(0) / 1000) {
        Some((n, i)) => format!("{n} {} ago", UNITS[i]),
        None => "just now".into(),
    }
}

/// A spelled-out age, such as "5 minutes ago" or "2 years ago".
pub fn ago_long(millis: i64) -> String {
    const UNITS: [&str; 6] = ["minute", "hour", "day", "week", "month", "year"];
    match age_unit((now_millis() - millis).max(0) / 1000) {
        Some((1, i)) => format!("1 {} ago", UNITS[i]),
        Some((n, i)) => format!("{n} {}s ago", UNITS[i]),
        None => "Just now".into(),
    }
}

/// The local date and time, such as "Sep 30, 2026, 8:41 AM".
pub fn date_time(millis: i64) -> String {
    chrono::DateTime::from_timestamp_millis(millis)
        .map(|d| d.with_timezone(&chrono::Local).format("%b %-d, %Y, %-I:%M %p").to_string())
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

/// A count with commas between groups of three digits, such as 12,345.
pub fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A rough length of time for a progress line, such as about 9 minutes.
pub fn about_duration(secs: u64) -> String {
    match secs {
        0..50 => "less than a minute".into(),
        50..90 => "about a minute".into(),
        90..3_300 => format!("about {} minutes", (secs + 30) / 60),
        _ => format!("about {:.1} hours", secs as f64 / 3600.0),
    }
}
