//! A zip of one conversation with its project, the model server's logs, and a description of this computer, to send
//! to someone who looks at what went wrong.

use std::io::Write;
use std::path::Path;

use anyhow::Context;

/// Folders that hold downloaded packages, build output, or version history rather than the project's own files.
const SKIPPED_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "dist", "build", "out", ".next", ".nuxt", "__pycache__", ".venv", "venv", ".cache", ".turbo", ".gradle",
    ".vs", ".idea",
];
const SKIPPED_EXTENSIONS: &[&str] = &["gguf", "bin", "exe", "dll", "so", "dylib", "zip", "7z", "gz", "tar", "mp4", "mov", "iso"];
/// Larger files are left out and listed, so one asset cannot swell the package.
const MAX_FILE: u64 = 2_000_000;
const MAX_TOTAL: u64 = 200_000_000;
/// The end of each server log that goes in. A log grows by a line for every few tokens the model writes.
const LOG_TAIL: usize = 3_000_000;

pub struct Packed {
    pub files: usize,
    pub bytes: u64,
}

/// Writes the zip to `out`: the conversation file, the project's files under `project/`, the model server's logs under
/// `logs/`, and `report.txt` with `report` and the files left out.
pub fn review_package(conversation: &Path, project: &Path, out: &Path, report: &str) -> anyhow::Result<Packed> {
    let file = std::fs::File::create(out).with_context(|| format!("Could not create {}", out.display()))?;
    let mut zip = Zip::new(std::io::BufWriter::new(file));
    let name = conversation.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "conversation.jsonl".into());
    zip.add(&format!("conversation/{name}"), &std::fs::read(conversation).context("Could not read the conversation")?)?;
    let mut skipped = Vec::new();
    let mut total = 0;
    if project.is_dir() {
        walk(project, project, &mut zip, &mut skipped, &mut total)?;
    }
    for log in [crate::paths::get().server_log(), crate::paths::get().server_log().with_extension("previous.log")] {
        if let Ok(bytes) = std::fs::read(&log) {
            let tail = &bytes[bytes.len().saturating_sub(LOG_TAIL)..];
            zip.add(&format!("logs/{}", log.file_name().unwrap_or_default().to_string_lossy()), tail)?;
        }
    }
    let mut text = report.to_string();
    if !skipped.is_empty() {
        text.push_str("\n\nLeft out of project/:\n");
        for s in &skipped {
            text.push_str(&format!("  {s}\n"));
        }
    }
    zip.add("report.txt", text.as_bytes())?;
    let files = zip.entries.len();
    let bytes = zip.finish()?;
    Ok(Packed { files, bytes })
}

fn walk(root: &Path, dir: &Path, zip: &mut Zip<impl Write>, skipped: &mut Vec<String>, total: &mut u64) -> anyhow::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
        let Ok(kind) = entry.file_type() else { continue };
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            if SKIPPED_DIRS.iter().any(|d| entry.file_name().eq_ignore_ascii_case(d)) {
                skipped.push(format!("{rel}/ (packages or build output)"));
            } else {
                walk(root, &path, zip, skipped, total)?;
            }
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if SKIPPED_EXTENSIONS.contains(&ext.as_str()) || size > MAX_FILE {
            skipped.push(format!("{rel} ({} KB)", size / 1000));
        } else if *total + size > MAX_TOTAL {
            skipped.push(format!("{rel} (the package was full)"));
        } else if let Ok(bytes) = std::fs::read(&path) {
            *total += size;
            zip.add(&format!("project/{rel}"), &bytes)?;
        }
    }
    Ok(())
}

/// A zip writer with deflate compression and no extensions, so the package stays under 4 GB and 65535 files.
struct Zip<W: Write> {
    out: W,
    entries: Vec<(String, u32, u32, u32, u32)>,
    offset: u64,
    stamp: (u16, u16),
}

impl<W: Write> Zip<W> {
    fn new(out: W) -> Self {
        use chrono::{Datelike, Timelike};
        let now = chrono::Local::now();
        let time = ((now.hour() << 11) | (now.minute() << 5) | (now.second() / 2)) as u16;
        let date = (((now.year() - 1980).max(0) as u32) << 9 | now.month() << 5 | now.day()) as u16;
        Zip { out, entries: Vec::new(), offset: 0, stamp: (time, date) }
    }

    fn add(&mut self, name: &str, data: &[u8]) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.len() < 65_000, "The project has too many files to pack.");
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data)?;
        let packed = enc.finish()?;
        let crc = crc32fast::hash(data);
        let offset = u32::try_from(self.offset).context("The package grew past 4 GB.")?;
        let mut head = Vec::with_capacity(30 + name.len());
        head.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        head.extend_from_slice(&20u16.to_le_bytes());
        // Bit 11 marks the names as UTF-8.
        head.extend_from_slice(&0x0800u16.to_le_bytes());
        head.extend_from_slice(&8u16.to_le_bytes());
        head.extend_from_slice(&self.stamp.0.to_le_bytes());
        head.extend_from_slice(&self.stamp.1.to_le_bytes());
        head.extend_from_slice(&crc.to_le_bytes());
        head.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        head.extend_from_slice(&(data.len() as u32).to_le_bytes());
        head.extend_from_slice(&(name.len() as u16).to_le_bytes());
        head.extend_from_slice(&0u16.to_le_bytes());
        head.extend_from_slice(name.as_bytes());
        self.out.write_all(&head)?;
        self.out.write_all(&packed)?;
        self.offset += (head.len() + packed.len()) as u64;
        self.entries.push((name.to_string(), crc, packed.len() as u32, data.len() as u32, offset));
        Ok(())
    }

    /// Writes the central directory and returns the package's size.
    fn finish(mut self) -> anyhow::Result<u64> {
        let start = u32::try_from(self.offset).context("The package grew past 4 GB.")?;
        let mut dir = Vec::new();
        for (name, crc, packed, size, offset) in &self.entries {
            dir.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            dir.extend_from_slice(&20u16.to_le_bytes());
            dir.extend_from_slice(&20u16.to_le_bytes());
            dir.extend_from_slice(&0x0800u16.to_le_bytes());
            dir.extend_from_slice(&8u16.to_le_bytes());
            dir.extend_from_slice(&self.stamp.0.to_le_bytes());
            dir.extend_from_slice(&self.stamp.1.to_le_bytes());
            dir.extend_from_slice(&crc.to_le_bytes());
            dir.extend_from_slice(&packed.to_le_bytes());
            dir.extend_from_slice(&size.to_le_bytes());
            dir.extend_from_slice(&(name.len() as u16).to_le_bytes());
            dir.extend_from_slice(&[0u8; 12]);
            dir.extend_from_slice(&offset.to_le_bytes());
            dir.extend_from_slice(name.as_bytes());
        }
        let count = self.entries.len() as u16;
        let mut end = Vec::new();
        end.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        end.extend_from_slice(&[0u8; 4]);
        end.extend_from_slice(&count.to_le_bytes());
        end.extend_from_slice(&count.to_le_bytes());
        end.extend_from_slice(&(dir.len() as u32).to_le_bytes());
        end.extend_from_slice(&start.to_le_bytes());
        end.extend_from_slice(&0u16.to_le_bytes());
        self.out.write_all(&dir)?;
        self.out.write_all(&end)?;
        self.out.flush()?;
        Ok(self.offset + (dir.len() + end.len()) as u64)
    }
}

/// The package's file name: the conversation's title with unsafe characters replaced, and the date.
pub fn file_name(title: &str) -> String {
    let safe: String = title.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { ' ' }).collect();
    let words: Vec<&str> = safe.split_whitespace().take(8).collect();
    let stem = if words.is_empty() { "conversation".to_string() } else { words.join("-") };
    format!("{stem}-{}.zip", chrono::Local::now().format("%Y%m%d-%H%M"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_a_project_that_a_zip_reader_opens() {
        let dir = std::env::temp_dir().join(format!("scoobert-export-{}", crate::util::random_hex(4)));
        let project = dir.join("proj");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::create_dir_all(project.join("node_modules/x")).unwrap();
        std::fs::write(project.join("src/main.ts"), "console.log(1);\n".repeat(50)).unwrap();
        std::fs::write(project.join("node_modules/x/index.js"), "skip").unwrap();
        let conv = dir.join("c.jsonl");
        std::fs::write(&conv, "{}\n").unwrap();
        let out = dir.join("out.zip");
        let packed = review_package(&conv, &project, &out, "Scoobert test").unwrap();
        assert_eq!(packed.bytes, std::fs::metadata(&out).unwrap().len());
        let bytes = std::fs::read(&out).unwrap();
        // The end record names every entry, and the project's packages are left out.
        let count = u16::from_le_bytes([bytes[bytes.len() - 12], bytes[bytes.len() - 11]]) as usize;
        assert_eq!(count, packed.files);
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("project/src/main.ts") && text.contains("report.txt") && !text.contains("node_modules/x/index.js"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn names_the_file_after_the_title() {
        assert!(file_name("Fix the map: engine!").starts_with("Fix-the-map-engine-"));
        assert!(file_name("").starts_with("conversation-"));
    }
}
