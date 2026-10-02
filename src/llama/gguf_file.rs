//! Reads a GGUF model file's header as it is stored and writes changed copies of it. Unchanged metadata is copied
//! byte for byte, tensors can be scaled by multiplying the scale stored in each of their blocks, and tensor data
//! is streamed, so a model larger than memory can be rewritten.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, bail, ensure};
use half::{bf16, f16};

use crate::util::Cancelled;

const STRING: u32 = 8;
const ARRAY: u32 = 9;
const DEFAULT_ALIGNMENT: u64 = 32;
/// About this many bytes are copied per read, rounded down to whole blocks of the tensor being copied.
const CHUNK: u64 = 16 * 1024 * 1024;

/// One metadata entry, with its value kept in the file's own encoding so an unchanged entry is copied exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct Kv {
    pub key: String,
    pub kind: u32,
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub kind: u32,
    /// From the start of the tensor data.
    pub offset: u64,
}

pub struct GgufFile {
    pub version: u32,
    pub kvs: Vec<Kv>,
    pub tensors: Vec<Tensor>,
    pub alignment: u64,
    pub data_start: u64,
    pub file_len: u64,
}

/// A tensor of the copy: where its bytes are in the source file, and the factor its weights are multiplied by.
#[derive(Debug, Clone)]
pub struct OutTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub kind: u32,
    pub source: u64,
    pub len: u64,
    pub scale: Option<f32>,
}

impl Kv {
    pub fn string(key: &str, value: &str) -> Kv {
        let mut raw = (value.len() as u64).to_le_bytes().to_vec();
        raw.extend_from_slice(value.as_bytes());
        Kv { key: key.into(), kind: STRING, raw }
    }

    pub fn as_u64(&self) -> Option<u64> {
        let r = &self.raw;
        Some(match self.kind {
            0 => r[0] as u64,
            1 => u64::try_from(r[0] as i8).ok()?,
            2 => u16::from_le_bytes(r[..2].try_into().ok()?) as u64,
            3 => u64::try_from(i16::from_le_bytes(r[..2].try_into().ok()?)).ok()?,
            4 => u32::from_le_bytes(r[..4].try_into().ok()?) as u64,
            5 => u64::try_from(i32::from_le_bytes(r[..4].try_into().ok()?)).ok()?,
            10 => u64::from_le_bytes(r[..8].try_into().ok()?),
            11 => u64::try_from(i64::from_le_bytes(r[..8].try_into().ok()?)).ok()?,
            _ => return None,
        })
    }

    /// The same entry with a new whole-number value, in the entry's own width.
    pub fn with_u64(&self, value: u64) -> Option<Kv> {
        let raw = match self.kind {
            0 => vec![u8::try_from(value).ok()?],
            1 => vec![i8::try_from(value).ok()? as u8],
            2 => u16::try_from(value).ok()?.to_le_bytes().to_vec(),
            3 => i16::try_from(value).ok()?.to_le_bytes().to_vec(),
            4 => u32::try_from(value).ok()?.to_le_bytes().to_vec(),
            5 => i32::try_from(value).ok()?.to_le_bytes().to_vec(),
            10 => value.to_le_bytes().to_vec(),
            11 => i64::try_from(value).ok()?.to_le_bytes().to_vec(),
            _ => return None,
        };
        Some(Kv { raw, ..self.clone() })
    }

    pub fn as_str(&self) -> Option<&str> {
        (self.kind == STRING).then(|| std::str::from_utf8(self.raw.get(8..)?).ok()).flatten()
    }

    /// An array's item type and each item's encoded bytes. Arrays of arrays are not split.
    pub fn items(&self) -> Option<(u32, Vec<Vec<u8>>)> {
        if self.kind != ARRAY {
            return None;
        }
        let item = u32::from_le_bytes(self.raw.get(..4)?.try_into().ok()?);
        let n = u64::from_le_bytes(self.raw.get(4..12)?.try_into().ok()?) as usize;
        let mut rest = &self.raw[12..];
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let len = match fixed_width(item) {
                Some(w) => w,
                None if item == STRING => 8 + u64::from_le_bytes(rest.get(..8)?.try_into().ok()?) as usize,
                None => return None,
            };
            out.push(rest.get(..len)?.to_vec());
            rest = &rest[len..];
        }
        Some((item, out))
    }

    pub fn with_items(&self, item: u32, items: &[Vec<u8>]) -> Kv {
        let mut raw = item.to_le_bytes().to_vec();
        raw.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            raw.extend_from_slice(i);
        }
        Kv { raw, ..self.clone() }
    }
}

fn fixed_width(kind: u32) -> Option<usize> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Elements per block, bytes per block, and the byte offsets inside a block of the 16-bit scales that every weight
/// in the block is multiplied by. Taken from the block structs in llama.cpp's ggml-common.h. A type with no offsets
/// can be copied but not scaled.
pub fn block_layout(kind: u32) -> Option<(u64, u64, &'static [usize])> {
    Some(match kind {
        0 => (1, 4, &[]),
        1 => (1, 2, &[0]),
        30 => (1, 2, &[]),
        2 => (32, 18, &[0]),
        3 => (32, 20, &[0, 2]),
        6 => (32, 22, &[0]),
        7 => (32, 24, &[0, 2]),
        8 => (32, 34, &[0]),
        10 => (256, 84, &[80, 82]),
        11 => (256, 110, &[108]),
        12 => (256, 144, &[0, 2]),
        13 => (256, 176, &[0, 2]),
        14 => (256, 210, &[208]),
        16 => (256, 66, &[0]),
        17 => (256, 74, &[0]),
        18 => (256, 98, &[0]),
        19 => (256, 50, &[0]),
        20 => (32, 18, &[0]),
        21 => (256, 110, &[0]),
        22 => (256, 82, &[0]),
        23 => (256, 136, &[0]),
        29 => (256, 56, &[]),
        24 => (1, 1, &[]),
        25 => (1, 2, &[]),
        26 => (1, 4, &[]),
        27 | 28 => (1, 8, &[]),
        _ => return None,
    })
}

/// Whether a tensor of this type can be multiplied by a factor in place.
pub fn scalable(kind: u32) -> bool {
    kind == 0 || kind == 30 || block_layout(kind).is_some_and(|(_, _, offsets)| !offsets.is_empty())
}

/// The name llama.cpp prints for a tensor type.
pub fn type_name(kind: u32) -> String {
    let name = match kind {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        30 => "BF16",
        _ => return format!("type {kind}"),
    };
    name.into()
}

impl Tensor {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> anyhow::Result<GgufFile> {
        let file = File::open(path).with_context(|| format!("Could not open {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 20, file);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        ensure!(&magic == b"GGUF", "{} is not a GGUF file", path.display());
        let version = read_u32(&mut r)?;
        ensure!(version >= 2, "GGUF version {version} is too old");
        let n_tensors = read_u64(&mut r)?;
        let n_kv = read_u64(&mut r)?;
        ensure!(n_tensors < 1 << 24 && n_kv < 1 << 20, "{} has an implausible header", path.display());
        let mut kvs = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_string(&mut r)?;
            let kind = read_u32(&mut r)?;
            let mut raw = Vec::new();
            read_raw(&mut r, kind, &mut raw)?;
            kvs.push(Kv { key, kind, raw });
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            ensure!(n_dims <= 8, "Tensor {name} has {n_dims} dimensions");
            let dims = (0..n_dims).map(|_| read_u64(&mut r)).collect::<anyhow::Result<Vec<u64>>>()?;
            let kind = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensors.push(Tensor { name, dims, kind, offset });
        }
        let alignment = kvs.iter().find(|k| k.key == "general.alignment").and_then(Kv::as_u64).filter(|&a| a > 0).unwrap_or(DEFAULT_ALIGNMENT);
        let header_end = r.stream_position()?;
        let data_start = header_end.div_ceil(alignment) * alignment;
        Ok(GgufFile { version, kvs, tensors, alignment, data_start, file_len })
    }

    pub fn kv(&self, key: &str) -> Option<&Kv> {
        self.kvs.iter().find(|k| k.key == key)
    }

    pub fn architecture(&self) -> Option<String> {
        self.kv("general.architecture").and_then(Kv::as_str).map(String::from)
    }

    /// An architecture key, such as `block_count` for `qwen35.block_count`.
    pub fn arch_u64(&self, key: &str) -> Option<u64> {
        self.kv(&format!("{}.{key}", self.architecture()?)).and_then(Kv::as_u64)
    }

    /// The bytes tensor `i` takes in the file, without the padding after it.
    pub fn tensor_len(&self, i: usize) -> u64 {
        let t = &self.tensors[i];
        if let Some((per, bytes, _)) = block_layout(t.kind)
            && t.elements() % per == 0
        {
            return t.elements() / per * bytes;
        }
        // An unknown type runs up to the next tensor, or to the end of the file.
        let next = self.tensors.iter().map(|o| o.offset).filter(|&o| o > t.offset).min();
        next.unwrap_or(self.file_len - self.data_start) - t.offset
    }

    /// The copy of tensor `i` as it is, ready to rename or scale.
    pub fn out(&self, i: usize) -> OutTensor {
        let t = &self.tensors[i];
        OutTensor { name: t.name.clone(), dims: t.dims.clone(), kind: t.kind, source: self.data_start + t.offset, len: self.tensor_len(i), scale: None }
    }
}

/// Writes a new GGUF file from `source`'s tensor data. It is written under a temporary name and renamed at the end,
/// so a stopped or failed write never leaves a broken model where the model list would find it.
pub fn write(
    source: &Path,
    out: &Path,
    version: u32,
    kvs: &[Kv],
    tensors: &[OutTensor],
    alignment: u64,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> anyhow::Result<()> {
    for t in tensors {
        if t.scale.is_some() && !scalable(t.kind) {
            bail!("{} is stored as {}, which cannot be scaled", t.name, type_name(t.kind));
        }
    }
    let partial = out.with_extension("gguf.partial");
    let result = write_to(source, &partial, version, kvs, tensors, alignment, cancel, &mut progress);
    match result {
        Ok(()) => std::fs::rename(&partial, out).with_context(|| format!("Could not name {}", out.display())),
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_to(
    source: &Path,
    out: &Path,
    version: u32,
    kvs: &[Kv],
    tensors: &[OutTensor],
    alignment: u64,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(u64, u64),
) -> anyhow::Result<()> {
    let mut src = File::open(source).with_context(|| format!("Could not open {}", source.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, File::create(out).with_context(|| format!("Could not create {}", out.display()))?);
    w.write_all(b"GGUF")?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(kvs.len() as u64).to_le_bytes())?;
    for kv in kvs {
        write_string(&mut w, &kv.key)?;
        w.write_all(&kv.kind.to_le_bytes())?;
        w.write_all(&kv.raw)?;
    }
    let mut offset = 0u64;
    for t in tensors {
        write_string(&mut w, &t.name)?;
        w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for d in &t.dims {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&t.kind.to_le_bytes())?;
        w.write_all(&offset.to_le_bytes())?;
        offset = (offset + t.len).div_ceil(alignment) * alignment;
    }
    let mut written = header_len(kvs, tensors);
    pad(&mut w, &mut written, alignment)?;
    let total: u64 = tensors.iter().map(|t| t.len).sum();
    let mut done = 0u64;
    let mut buf = vec![0u8; CHUNK as usize + 256];
    for t in tensors {
        src.seek(SeekFrom::Start(t.source))?;
        // Scaling works on whole blocks, so a read never ends inside one.
        let block = block_layout(t.kind).map(|(_, b, _)| b).unwrap_or(1);
        let step = (CHUNK / block).max(1) * block;
        let mut left = t.len;
        while left > 0 {
            if cancel.load(Ordering::SeqCst) {
                return Err(Cancelled.into());
            }
            let n = left.min(step) as usize;
            src.read_exact(&mut buf[..n]).with_context(|| format!("Could not read {} from {}", t.name, source.display()))?;
            if let Some(factor) = t.scale {
                scale(t.kind, &mut buf[..n], factor)?;
            }
            w.write_all(&buf[..n])?;
            left -= n as u64;
            written += n as u64;
            done += n as u64;
            progress(done, total);
        }
        pad(&mut w, &mut written, alignment)?;
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
}

fn header_len(kvs: &[Kv], tensors: &[OutTensor]) -> u64 {
    let mut n = 4 + 4 + 8 + 8;
    for kv in kvs {
        n += 8 + kv.key.len() as u64 + 4 + kv.raw.len() as u64;
    }
    for t in tensors {
        n += 8 + t.name.len() as u64 + 4 + 8 * t.dims.len() as u64 + 4 + 8;
    }
    n
}

fn pad(w: &mut impl Write, written: &mut u64, alignment: u64) -> anyhow::Result<()> {
    let target = written.div_ceil(alignment) * alignment;
    w.write_all(&vec![0u8; (target - *written) as usize])?;
    *written = target;
    Ok(())
}

/// Multiplies every weight in `data`, whole blocks of type `kind`, by `factor`.
pub fn scale(kind: u32, data: &mut [u8], factor: f32) -> anyhow::Result<()> {
    match kind {
        0 => {
            for c in data.chunks_exact_mut(4) {
                let v = f32::from_le_bytes(c.try_into().unwrap()) * factor;
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        30 => {
            for c in data.chunks_exact_mut(2) {
                let v = bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32() * factor;
                c.copy_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
        }
        _ => {
            let Some((_, bytes, offsets)) = block_layout(kind).filter(|(_, _, o)| !o.is_empty()) else {
                bail!("{} cannot be scaled", type_name(kind));
            };
            ensure!(data.len() as u64 % bytes == 0, "A {} chunk ends inside a block", type_name(kind));
            for block in data.chunks_exact_mut(bytes as usize) {
                for &o in offsets {
                    let v = f16::from_bits(u16::from_le_bytes([block[o], block[o + 1]])).to_f32() * factor;
                    block[o..o + 2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
                }
            }
        }
    }
    Ok(())
}

fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> anyhow::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> anyhow::Result<String> {
    let n = read_u64(r)?;
    ensure!(n < 1 << 30, "A GGUF string is implausibly long");
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn write_string(w: &mut impl Write, s: &str) -> anyhow::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

/// Reads a value of type `kind` and appends its encoded bytes to `out`.
fn read_raw(r: &mut impl Read, kind: u32, out: &mut Vec<u8>) -> anyhow::Result<()> {
    if let Some(w) = fixed_width(kind) {
        let mut b = [0u8; 8];
        r.read_exact(&mut b[..w])?;
        out.extend_from_slice(&b[..w]);
        return Ok(());
    }
    match kind {
        STRING => {
            let n = read_u64(r)?;
            ensure!(n < 1 << 30, "A GGUF string is implausibly long");
            out.extend_from_slice(&n.to_le_bytes());
            let start = out.len();
            out.resize(start + n as usize, 0);
            r.read_exact(&mut out[start..])?;
        }
        ARRAY => {
            let item = read_u32(r)?;
            let n = read_u64(r)?;
            out.extend_from_slice(&item.to_le_bytes());
            out.extend_from_slice(&n.to_le_bytes());
            if let Some(w) = fixed_width(item) {
                let start = out.len();
                out.resize(start + n as usize * w, 0);
                r.read_exact(&mut out[start..])?;
            } else {
                for _ in 0..n {
                    read_raw(r, item, out)?;
                }
            }
        }
        other => bail!("Unknown GGUF value type {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small model file: two metadata entries, an F32 tensor, and a Q8_0 tensor of two blocks.
    fn sample(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("sample.gguf");
        let kvs = vec![Kv::string("general.architecture", "test"), Kv { key: "test.block_count".into(), kind: 4, raw: 2u32.to_le_bytes().to_vec() }];
        let mut data = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let f32_len = data.len() as u64;
        for _ in 0..2 {
            data.extend_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
            data.extend(std::iter::repeat_n(3u8, 32));
        }
        let source = dir.join("raw.bin");
        std::fs::write(&source, &data).unwrap();
        let tensors = vec![
            OutTensor { name: "blk.0.a".into(), dims: vec![4], kind: 0, source: 0, len: f32_len, scale: None },
            OutTensor { name: "blk.1.b".into(), dims: vec![64], kind: 8, source: f32_len, len: 68, scale: None },
        ];
        write(&source, &path, 3, &kvs, &tensors, 32, &AtomicBool::new(false), |_, _| {}).unwrap();
        path
    }

    fn temp() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("scoobert-gguf-{}", crate::util::random_hex(4)));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_written_file_reads_back_the_same() {
        let dir = temp();
        let g = GgufFile::open(&sample(&dir)).unwrap();
        assert_eq!(g.architecture().as_deref(), Some("test"));
        assert_eq!(g.arch_u64("block_count"), Some(2));
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.tensors[1].offset, 32);
        assert_eq!(g.tensor_len(1), 68);
        assert_eq!(g.data_start % 32, 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn scaling_multiplies_block_scales_and_floats() {
        let dir = temp();
        let path = sample(&dir);
        let g = GgufFile::open(&path).unwrap();
        let mut tensors: Vec<OutTensor> = (0..2).map(|i| g.out(i)).collect();
        for t in &mut tensors {
            t.scale = Some(2.0);
        }
        let out = dir.join("scaled.gguf");
        write(&path, &out, g.version, &g.kvs, &tensors, g.alignment, &AtomicBool::new(false), |_, _| {}).unwrap();
        let s = GgufFile::open(&out).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let a = s.data_start as usize;
        assert_eq!(f32::from_le_bytes(bytes[a + 4..a + 8].try_into().unwrap()), 4.0);
        let b = a + s.tensors[1].offset as usize;
        assert_eq!(f16::from_bits(u16::from_le_bytes([bytes[b], bytes[b + 1]])).to_f32(), 1.0);
        assert_eq!(bytes[b + 2], 3);
        assert_eq!(f16::from_bits(u16::from_le_bytes([bytes[b + 34], bytes[b + 35]])).to_f32(), 1.0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn whole_numbers_and_arrays_keep_their_encoding() {
        let kv = Kv { key: "k".into(), kind: 4, raw: 7u32.to_le_bytes().to_vec() };
        assert_eq!(kv.with_u64(9).unwrap().raw, 9u32.to_le_bytes());
        let mut raw = 5u32.to_le_bytes().to_vec();
        raw.extend_from_slice(&3u64.to_le_bytes());
        for v in [10i32, 20, 30] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let arr = Kv { key: "a".into(), kind: ARRAY, raw };
        let (item, items) = arr.items().unwrap();
        assert_eq!((item, items.len()), (5, 3));
        let picked = arr.with_items(item, &[items[2].clone(), items[0].clone()]);
        assert_eq!(picked.items().unwrap().1, vec![30i32.to_le_bytes().to_vec(), 10i32.to_le_bytes().to_vec()]);
    }
}
