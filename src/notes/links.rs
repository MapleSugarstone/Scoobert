//! Wikilink parsing and resolution for the notes folder.

use std::sync::LazyLock;

use regex::Regex;

pub static WIKILINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(!?)\[\[([^\[\]\n]+?)\]\]").unwrap());
static KNOWN_EXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\.(md|png|jpe?g|gif|svg|webp|bmp|pdf|canvas|txt)$").unwrap());
static FENCED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```").unwrap());
static INLINE_CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`[^`\n]*`").unwrap());
static TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\s)#([\p{L}\p{N}_/-]*\p{L}[\p{L}\p{N}_/-]*)").unwrap());

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub target: String,
    pub alias: Option<String>,
    pub heading: Option<String>,
}

pub fn parse_link(raw: &str) -> Link {
    let (target, alias) = match raw.find('|') {
        Some(i) => (&raw[..i], Some(raw[i + 1..].trim().to_string()).filter(|a| !a.is_empty())),
        None => (raw, None),
    };
    let (target, heading) = match target.find('#') {
        Some(i) => (&target[..i], Some(target[i + 1..].trim().to_string())),
        None => (target, None),
    };
    Link { target: target.trim().to_string(), alias, heading }
}

/// The note's name: its file name without the .md extension.
pub fn note_name(rel_path: &str) -> String {
    let base = rel_path.rsplit('/').next().unwrap_or(rel_path);
    match base.len().checked_sub(3) {
        Some(n) if base[n..].eq_ignore_ascii_case(".md") => base[..n].to_string(),
        _ => base.to_string(),
    }
}

/// Resolves a link target: an exact path first, then the shortest path with that file name.
pub fn resolve_link<'a>(target: &str, paths: &'a [String]) -> Option<&'a String> {
    if target.is_empty() {
        return None;
    }
    let t = target.replace('\\', "/").trim_start_matches('/').to_lowercase();
    let wanted = if KNOWN_EXT.is_match(&t) { t } else { format!("{t}.md") };
    let suffix = format!("/{wanted}");
    let mut best: Option<&String> = None;
    for p in paths {
        let lp = p.to_lowercase();
        if lp == wanted {
            return Some(p);
        }
        if lp.ends_with(&suffix) && best.is_none_or(|b| p.len() < b.len()) {
            best = Some(p);
        }
    }
    best
}

pub fn strip_code(content: &str) -> String {
    let without_blocks = FENCED.replace_all(content, "");
    INLINE_CODE.replace_all(&without_blocks, "").into_owned()
}

pub fn extract_links(content: &str) -> Vec<String> {
    WIKILINK
        .captures_iter(&strip_code(content))
        .map(|c| parse_link(&c[2]).target)
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn extract_tags(content: &str) -> Vec<String> {
    let mut out: Vec<String> = TAG.captures_iter(&strip_code(content)).map(|c| c[1].to_string()).collect();
    out.sort();
    out.dedup();
    out
}

/// Rewrites every link to `old_name` so it points at `new_name`, keeping headings and aliases.
pub fn rename_links(content: &str, old_name: &str, new_name: &str) -> String {
    let lower = old_name.to_lowercase();
    WIKILINK
        .replace_all(content, |c: &regex::Captures| {
            let inner = &c[2];
            let target = parse_link(inner).target;
            let bare = target.strip_suffix(".md").or_else(|| target.strip_suffix(".MD")).unwrap_or(&target);
            let name = bare.rsplit('/').next().unwrap_or(bare);
            if bare.to_lowercase() != lower && name.to_lowercase() != lower {
                return c[0].to_string();
            }
            let rest = inner.find(&target).map(|i| &inner[i + target.len()..]).unwrap_or("");
            format!("{}[[{new_name}{rest}]]", &c[1])
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_by_name_and_path() {
        let paths = vec!["Daily/2026-01-01.md".to_string(), "Auth design.md".to_string(), "old/Auth design.md".to_string()];
        assert_eq!(resolve_link("auth design", &paths).map(String::as_str), Some("Auth design.md"));
        assert_eq!(resolve_link("2026-01-01", &paths).map(String::as_str), Some("Daily/2026-01-01.md"));
        assert_eq!(resolve_link("missing", &paths), None);
    }

    #[test]
    fn renames_links_keeping_aliases() {
        let text = "See [[Auth design#Tokens|tokens]] and [[auth design]] but not `[[Auth design]]` or [[Other]].";
        let out = rename_links(text, "Auth design", "Login design");
        assert_eq!(out, "See [[Login design#Tokens|tokens]] and [[Login design]] but not `[[Login design]]` or [[Other]].");
    }

    #[test]
    fn extracts_tags_outside_code() {
        let tags = extract_tags("#todo some text #area/auth `#notatag` #123");
        assert_eq!(tags, vec!["area/auth".to_string(), "todo".to_string()]);
    }
}
