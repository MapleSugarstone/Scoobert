//! A project's notes folder: plain Markdown files that link to each other with [[wikilinks]].

pub mod links;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, bail};

use links::{extract_links, extract_tags, note_name, rename_links, resolve_link};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Path relative to the notes folder, with forward slashes.
    pub path: String,
    pub is_dir: bool,
    pub modified: SystemTime,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub path: String,
    pub name_hit: bool,
    pub lines: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub struct Backlink {
    pub path: String,
    pub context: String,
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// Note paths, then unresolved link targets.
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub name: String,
    pub ghost: bool,
}

#[derive(Debug, Clone)]
pub struct Vault {
    pub root: PathBuf,
}

impl Vault {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Vault { root: root.into() }
    }

    pub fn exists(&self) -> bool {
        self.root.is_dir()
    }

    /// Resolves a relative path and rejects anything that would leave the notes folder.
    pub fn abs(&self, rel: &str) -> anyhow::Result<PathBuf> {
        let rel = Path::new(rel.trim_start_matches(['/', '\\']));
        let mut out = self.root.clone();
        for part in rel.components() {
            match part {
                Component::Normal(p) => out.push(p),
                Component::CurDir => {}
                _ => bail!("That path is outside the notes folder."),
            }
        }
        Ok(out)
    }

    pub fn rel(&self, abs: &Path) -> String {
        abs.strip_prefix(&self.root).unwrap_or(abs).to_string_lossy().replace('\\', "/")
    }

    pub fn files(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        self.walk(&self.root, &mut out);
        out
    }

    fn walk(&self, dir: &Path, out: &mut Vec<Entry>) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = read.flatten().collect();
        entries.sort_by_key(|e| e.file_name().to_string_lossy().to_lowercase());
        for e in entries {
            let name = e.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(meta) = e.metadata() else { continue };
            let path = e.path();
            out.push(Entry { path: self.rel(&path), is_dir: meta.is_dir(), modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH) });
            if meta.is_dir() {
                self.walk(&path, out);
            }
        }
    }

    pub fn notes(&self) -> Vec<Entry> {
        self.files().into_iter().filter(|f| !f.is_dir && f.path.to_lowercase().ends_with(".md")).collect()
    }

    pub fn read(&self, rel: &str) -> anyhow::Result<String> {
        let p = self.abs(rel)?;
        std::fs::read_to_string(&p).with_context(|| format!("Could not read {rel}"))
    }

    pub fn write(&self, rel: &str, content: &str) -> anyhow::Result<String> {
        let p = self.abs(rel)?;
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&p, content).with_context(|| format!("Could not write {rel}"))?;
        Ok(self.rel(&p))
    }

    fn unique(&self, dir: &str, base: &str) -> anyhow::Result<String> {
        std::fs::create_dir_all(self.abs(dir)?)?;
        for i in 0.. {
            let name = if i == 0 { format!("{base}.md") } else { format!("{base} {i}.md") };
            let rel = if dir.is_empty() { name } else { format!("{dir}/{name}") };
            if !self.abs(&rel)?.exists() {
                return Ok(rel);
            }
        }
        unreachable!()
    }

    pub fn create(&self, dir: &str, base: &str, content: &str) -> anyhow::Result<String> {
        let rel = self.unique(dir, &sanitize_name(base))?;
        self.write(&rel, content)
    }

    pub fn mkdir(&self, rel: &str) -> anyhow::Result<()> {
        std::fs::create_dir_all(self.abs(rel)?)?;
        Ok(())
    }

    /// Renames a note or folder and rewrites links that pointed at a renamed note.
    pub fn rename(&self, from: &str, to: &str) -> anyhow::Result<String> {
        let src = self.abs(from)?;
        let dst = self.abs(to)?;
        if dst.exists() {
            bail!("{to} already exists.");
        }
        if let Some(dir) = dst.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::rename(&src, &dst)?;
        let is_note = |p: &str| p.to_lowercase().ends_with(".md");
        if is_note(from) && is_note(to) {
            let (old, new) = (note_name(from), note_name(to));
            if old != new {
                for n in self.notes() {
                    let text = self.read(&n.path)?;
                    let next = rename_links(&text, &old, &new);
                    if next != text {
                        self.write(&n.path, &next)?;
                    }
                }
            }
        }
        Ok(self.rel(&dst))
    }

    /// Moves a note or folder to the system trash.
    pub fn delete(&self, rel: &str) -> anyhow::Result<()> {
        let p = self.abs(rel)?;
        if p == self.root {
            bail!("The notes folder itself cannot be deleted here.");
        }
        trash::delete(&p).map_err(|err| anyhow::anyhow!("Could not move {rel} to the trash: {err}"))
    }

    pub fn search(&self, query: &str) -> Vec<SearchHit> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for n in self.notes() {
            let text = self.read(&n.path).unwrap_or_default();
            let lines: Vec<(usize, String)> = text
                .lines()
                .enumerate()
                .filter(|(_, l)| l.to_lowercase().contains(&q))
                .take(3)
                .map(|(i, l)| (i + 1, clip(l.trim(), 200)))
                .collect();
            let name_hit = n.path.to_lowercase().contains(&q);
            if name_hit || !lines.is_empty() {
                hits.push(SearchHit { path: n.path, name_hit, lines });
            }
        }
        hits.sort_by_key(|h| !h.name_hit);
        hits.truncate(100);
        hits
    }

    /// Every note's outgoing links, resolved against every file in the folder.
    fn link_index(&self) -> (Vec<String>, BTreeMap<String, Vec<(String, Option<String>)>>) {
        let files = self.files();
        let paths: Vec<String> = files.iter().filter(|f| !f.is_dir).map(|f| f.path.clone()).collect();
        let notes: Vec<String> = paths.iter().filter(|p| p.to_lowercase().ends_with(".md")).cloned().collect();
        let mut links = BTreeMap::new();
        for n in &notes {
            let text = self.read(n).unwrap_or_default();
            let targets = extract_links(&text).into_iter().map(|t| {
                let resolved = resolve_link(&t, &paths).cloned();
                (t, resolved)
            });
            links.insert(n.clone(), targets.collect());
        }
        (notes, links)
    }

    pub fn tags(&self) -> BTreeMap<String, Vec<String>> {
        let mut tags: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for n in self.notes() {
            for tag in extract_tags(&self.read(&n.path).unwrap_or_default()) {
                tags.entry(tag).or_default().push(n.path.clone());
            }
        }
        tags
    }

    pub fn backlinks(&self, rel: &str) -> Vec<Backlink> {
        let (_, links) = self.link_index();
        let name = format!("[[{}", note_name(rel).to_lowercase());
        let mut out = Vec::new();
        for (from, targets) in links {
            if from == rel || !targets.iter().any(|(_, r)| r.as_deref() == Some(rel)) {
                continue;
            }
            let text = self.read(&from).unwrap_or_default();
            let context = text.lines().find(|l| l.to_lowercase().contains(&name)).unwrap_or("");
            out.push(Backlink { path: from, context: clip(context.trim(), 200) });
        }
        out
    }

    pub fn graph(&self) -> Graph {
        let (notes, links) = self.link_index();
        let mut nodes: Vec<GraphNode> =
            notes.iter().map(|p| GraphNode { id: p.clone(), name: note_name(p), ghost: false }).collect();
        let mut index: BTreeMap<String, usize> = notes.iter().enumerate().map(|(i, p)| (p.clone(), i)).collect();
        let mut edges = BTreeSet::new();
        for (from, targets) in &links {
            let a = index[from];
            for (target, resolved) in targets {
                let b = match resolved {
                    Some(r) if r.to_lowercase().ends_with(".md") => match index.get(r) {
                        Some(&b) => b,
                        None => continue,
                    },
                    Some(_) => continue,
                    None => {
                        let id = format!("ghost:{}", target.to_lowercase());
                        *index.entry(id.clone()).or_insert_with(|| {
                            nodes.push(GraphNode { id, name: target.clone(), ghost: true });
                            nodes.len() - 1
                        })
                    }
                };
                if a != b {
                    edges.insert((a.min(b), a.max(b)));
                }
            }
        }
        Graph { nodes, edges: edges.into_iter().collect() }
    }

    /// Today's daily note, created with a heading when missing.
    pub fn daily(&self, date: chrono::NaiveDate) -> anyhow::Result<String> {
        let stamp = date.format("%Y-%m-%d").to_string();
        let rel = format!("Daily/{stamp}.md");
        if !self.abs(&rel)?.exists() {
            self.write(&rel, &format!("# {stamp}\n\n"))?;
        }
        Ok(rel)
    }
}

pub fn sanitize_name(name: &str) -> String {
    let cleaned: String = name.chars().map(|c| if "\\/:*?\"<>|#^[]".contains(c) { ' ' } else { c }).collect();
    let joined = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed: String = joined.chars().take(120).collect();
    // Windows drops trailing dots from file names, and "Name..md" reads badly everywhere.
    let trimmed = trimmed.trim_end_matches(['.', ' ']).to_string();
    if trimmed.is_empty() { "Untitled".into() } else { trimmed }
}

pub fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max.saturating_sub(3)).collect();
    format!("{}...", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_outside_the_folder() {
        let v = Vault::new("/tmp/notes");
        assert!(v.abs("../secret").is_err());
        assert!(v.abs("Daily/2026-01-01.md").is_ok());
    }

    #[test]
    fn sanitizes_names() {
        assert_eq!(sanitize_name("  a/b: c?  "), "a b c");
        assert_eq!(sanitize_name("///"), "Untitled");
        assert_eq!(sanitize_name("Add a comment."), "Add a comment");
    }
}
