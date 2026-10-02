//! Runs a model lab edit on a real model file and checks the copy's structure, without loading the model.
//! Usage: cargo run --release --example lab_check -- <model.gguf> <out folder> <scale|remove|repeat> <from> <to>

use std::path::Path;
use std::sync::atomic::AtomicBool;

use scoobert::llama::gguf_file::GgufFile;
use scoobert::llama::lab::{self, LayerEdit, Part, Progress};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (src, dir, op) = (Path::new(&args[1]), Path::new(&args[2]), args[3].as_str());
    let (from, to): (u32, u32) = (args[4].parse()?, args[5].parse()?);
    let picked: Vec<u32> = (from..=to).collect();
    let out = dir.join(format!("{op}.gguf"));
    let cancel = AtomicBool::new(false);
    let last = std::cell::Cell::new(0u64);
    let report = |p: Progress| {
        let pct = p.done * 100 / p.total.max(1);
        if pct >= last.get() + 10 {
            last.set(pct);
            println!("{}: {pct}%", p.status);
        }
    };
    let start = std::time::Instant::now();
    match op {
        "scale" => lab::scale_layers(src, &out, "check scaled", Part::Both, &picked, 150, &cancel, &report)?,
        "remove" => lab::edit_layers(src, &out, "check removed", LayerEdit::Remove, &picked, &cancel, &report)?,
        "repeat" => lab::edit_layers(src, &out, "check repeated", LayerEdit::Repeat, &picked, &cancel, &report)?,
        other => anyhow::bail!("unknown operation {other}"),
    }
    println!("wrote {} in {:.0?}", out.display(), start.elapsed());
    let a = GgufFile::open(src)?;
    let b = GgufFile::open(&out)?;
    println!("layers {:?} -> {:?}", lab::layers(&a), lab::layers(&b));
    println!("tensors {} -> {}, metadata {} -> {}", a.tensors.len(), b.tensors.len(), a.kvs.len(), b.kvs.len());
    println!("name {:?}", b.kv("general.name").and_then(|k| k.as_str()));
    // Every copied tensor's bytes must match its source unless it was scaled.
    let read = |path: &Path, offset: u64, len: u64| -> anyhow::Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len.min(4096) as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    };
    let (mut same, mut differ) = (0, 0);
    for (j, t) in b.tensors.iter().enumerate() {
        let source_name = match op {
            "scale" => t.name.clone(),
            _ => t.name.clone(),
        };
        let Some(i) = a.tensors.iter().position(|s| s.name == source_name && s.dims == t.dims && s.kind == t.kind) else { continue };
        let x = read(src, a.data_start + a.tensors[i].offset, a.tensor_len(i))?;
        let y = read(&out, b.data_start + t.offset, b.tensor_len(j))?;
        if x == y { same += 1 } else { differ += 1 }
    }
    println!("tensors whose first bytes match a same-named source tensor: {same}, differ: {differ}");
    Ok(())
}
