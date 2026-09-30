//! Reads the key-value metadata at the start of a GGUF model file.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, ensure};

const HEADER_BYTES: usize = 16 * 1024 * 1024;
pub const FALLBACK_KV_BYTES_PER_TOKEN: u64 = 65_536;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    Skipped,
}

impl Value {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(n) if *n >= 0 => Some(*n as u64),
            Value::Array(items) => items.iter().filter_map(Value::as_u64).max(),
            _ => None,
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> anyhow::Result<&[u8]> {
        ensure!(self.pos + n <= self.buf.len(), "GGUF header is larger than the read buffer");
        self.pos += n;
        Ok(&self.buf[self.pos - n..self.pos])
    }

    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn string(&mut self) -> anyhow::Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn value(&mut self, kind: u32) -> anyhow::Result<Value> {
        Ok(match kind {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(i32::from_le_bytes(self.take(4)?.try_into()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(i64::from_le_bytes(self.take(8)?.try_into()?)),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into()?)),
            9 => {
                let item = self.u32()?;
                let n = self.u64()? as usize;
                let width = match item {
                    0 | 1 | 7 => Some(1),
                    2 | 3 => Some(2),
                    4..=6 => Some(4),
                    10..=12 => Some(8),
                    _ => None,
                };
                // Tokenizer tables hold hundreds of thousands of entries and are skipped.
                if let (Some(w), true) = (width, n > 64) {
                    self.take(n * w)?;
                    return Ok(Value::Skipped);
                }
                let mut items = Vec::with_capacity(n.min(64));
                for _ in 0..n {
                    items.push(self.value(item)?);
                }
                if item == 8 { Value::Skipped } else { Value::Array(items) }
            }
            other => bail!("Unknown GGUF value type {other}"),
        })
    }
}

pub fn read_metadata(file: &Path) -> anyhow::Result<HashMap<String, Value>> {
    let mut buf = Vec::with_capacity(HEADER_BYTES);
    std::fs::File::open(file)?.take(HEADER_BYTES as u64).read_to_end(&mut buf)?;
    ensure!(buf.starts_with(b"GGUF"), "Not a GGUF file");
    let mut r = Reader { buf: &buf, pos: 4 };
    r.u32()?;
    r.u64()?;
    let count = r.u64()?;
    let mut meta = HashMap::new();
    for _ in 0..count {
        let key = r.string()?;
        let kind = r.u32()?;
        let value = r.value(kind)?;
        let stop = key.starts_with("tokenizer.") && architecture_complete(&meta);
        meta.insert(key, value);
        if stop {
            break;
        }
    }
    Ok(meta)
}

/// The chat template stored in the model file. It follows the tokenizer tables, so more of the file is read.
pub fn chat_template(file: &Path) -> Option<String> {
    const TEMPLATE_BYTES: u64 = 48 * 1024 * 1024;
    let mut buf = Vec::new();
    std::fs::File::open(file).ok()?.take(TEMPLATE_BYTES).read_to_end(&mut buf).ok()?;
    if !buf.starts_with(b"GGUF") {
        return None;
    }
    let mut r = Reader { buf: &buf, pos: 4 };
    r.u32().ok()?;
    r.u64().ok()?;
    let count = r.u64().ok()?;
    for _ in 0..count {
        let key = r.string().ok()?;
        let kind = r.u32().ok()?;
        let value = r.value(kind).ok()?;
        if key == "tokenizer.chat_template" {
            return match value {
                Value::Str(s) => Some(s),
                _ => None,
            };
        }
    }
    None
}

fn architecture_complete(meta: &HashMap<String, Value>) -> bool {
    match meta.get("general.architecture") {
        Some(Value::Str(arch)) => meta.contains_key(&format!("{arch}.block_count")),
        _ => false,
    }
}

/// Attention cache per token: attention layers x KV heads x (key + value length) x 2 bytes. Hybrid models
/// such as Qwen3.5 have attention in only one layer of every `full_attention_interval`.
pub fn kv_bytes_per_token(file: &Path) -> u64 {
    static CACHE: Mutex<Option<HashMap<std::path::PathBuf, (std::time::SystemTime, u64)>>> = Mutex::new(None);
    let modified = std::fs::metadata(file).and_then(|m| m.modified()).ok();
    if let (Some(m), Some(map)) = (modified, CACHE.lock().unwrap().as_ref())
        && let Some((when, bytes)) = map.get(file)
        && *when == m
    {
        return *bytes;
    }
    let bytes = compute_kv(file).unwrap_or(FALLBACK_KV_BYTES_PER_TOKEN);
    if let Some(m) = modified {
        CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(file.to_path_buf(), (m, bytes));
    }
    bytes
}

fn compute_kv(file: &Path) -> Option<u64> {
    let meta = read_metadata(file).ok()?;
    let Some(Value::Str(arch)) = meta.get("general.architecture") else { return None };
    let get = |k: &str| meta.get(&format!("{arch}.{k}")).and_then(Value::as_u64);
    let layers = get("block_count")?;
    let interval = get("full_attention_interval").filter(|&n| n > 0).unwrap_or(1);
    let heads = get("attention.head_count")?;
    let kv_heads = get("attention.head_count_kv").unwrap_or(heads);
    let head_dim = get("embedding_length")? / heads.max(1);
    let k = get("attention.key_length").unwrap_or(head_dim);
    let v = get("attention.value_length").unwrap_or(head_dim);
    let per_token = layers.div_ceil(interval) * kv_heads * (k + v) * 2;
    (per_token > 0).then_some(per_token)
}
