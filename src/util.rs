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

pub fn ago(millis: i64) -> String {
    let secs = (now_millis() - millis).max(0) / 1000;
    match secs {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", secs / 60),
        3600..86_400 => format!("{} h ago", secs / 3600),
        86_400..2_592_000 => format!("{} d ago", secs / 86_400),
        _ => chrono::DateTime::from_timestamp_millis(millis)
            .map(|d| d.with_timezone(&chrono::Local).format("%b %-d, %Y").to_string())
            .unwrap_or_default(),
    }
}
