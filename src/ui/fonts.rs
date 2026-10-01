//! Picks installed interface and code fonts, so the app looks native on each platform without bundling fonts.

use std::path::Path;
use std::sync::OnceLock;

use iced::Font;
use iced::font::Weight;

struct Fonts {
    ui: &'static str,
    mono: &'static str,
}

fn pick() -> &'static Fonts {
    static FONTS: OnceLock<Fonts> = OnceLock::new();
    FONTS.get_or_init(|| {
        if cfg!(windows) {
            let dir = std::env::var_os("WINDIR").map(std::path::PathBuf::from).unwrap_or_else(|| "C:\\Windows".into()).join("Fonts");
            let mono = if dir.join("CascadiaMono.ttf").exists() { "Cascadia Mono" } else { "Consolas" };
            return Fonts { ui: "Segoe UI", mono };
        }
        let files = font_files();
        let has = |needles: &[&str]| needles.iter().any(|n| files.iter().any(|f| f.starts_with(&n.to_lowercase())));
        let ui = [
            ("Inter", &["inter"][..]),
            ("Noto Sans", &["notosans-regular", "notosans["][..]),
            ("Adwaita Sans", &["adwaitasans"][..]),
            ("Cantarell", &["cantarell"][..]),
            ("Ubuntu", &["ubuntu-r", "ubuntu["][..]),
            ("DejaVu Sans", &["dejavusans.ttf"][..]),
            ("Liberation Sans", &["liberationsans-regular"][..]),
        ]
        .into_iter()
        .find(|(_, files)| has(files))
        .map(|(name, _)| name)
        .unwrap_or("Open Sans");
        let mono = [
            ("JetBrains Mono", &["jetbrainsmono"][..]),
            ("Noto Sans Mono", &["notosansmono"][..]),
            ("Adwaita Mono", &["adwaitamono"][..]),
            ("DejaVu Sans Mono", &["dejavusansmono"][..]),
            ("Ubuntu Mono", &["ubuntumono"][..]),
            ("Liberation Mono", &["liberationmono"][..]),
        ]
        .into_iter()
        .find(|(_, files)| has(files))
        .map(|(name, _)| name)
        .unwrap_or("Noto Sans Mono");
        Fonts { ui, mono }
    })
}

/// A font made for the interface language's script. Japanese, Korean, and Chinese need one, because a Latin UI
/// font leaves their characters to fallback fonts that mix weights and pick Han glyph shapes by the system locale.
fn script_font() -> Option<&'static str> {
    cjk_family(crate::i18n::current().code)
}

/// The interface font for `text`, which may be in another language than the interface. Text with Japanese, Korean,
/// or Chinese characters gets a font made for its script, for the same reason as `script_font`.
pub fn for_text(text: &str) -> Font {
    if script_font().is_some() {
        return ui();
    }
    let code = if text.chars().any(|c| matches!(c, '\u{3040}'..='\u{30ff}')) {
        "ja"
    } else if text.chars().any(|c| matches!(c, '\u{1100}'..='\u{11ff}' | '\u{ac00}'..='\u{d7af}')) {
        "ko"
    } else if text.chars().any(|c| matches!(c, '\u{4e00}'..='\u{9fff}')) {
        // Han characters alone do not say which language they are, so the system's language decides.
        match crate::i18n::system() {
            Some(code @ ("ko" | "zh-Hans")) => code,
            _ => "ja",
        }
    } else {
        return ui();
    };
    cjk_family(code).map(Font::with_name).unwrap_or_else(ui)
}

fn cjk_family(code: &str) -> Option<&'static str> {
    if cfg!(windows) {
        return match code {
            "ja" => Some("Yu Gothic UI"),
            "ko" => Some("Malgun Gothic"),
            "zh-Hans" => Some("Microsoft YaHei UI"),
            _ => None,
        };
    }
    let (family, needles): (&'static str, [&str; 2]) = match code {
        "ja" => ("Noto Sans CJK JP", ["notosanscjk", "notosansjp"]),
        "ko" => ("Noto Sans CJK KR", ["notosanscjk", "notosanskr"]),
        "zh-Hans" => ("Noto Sans CJK SC", ["notosanscjk", "notosanssc"]),
        _ => return None,
    };
    static FILES: OnceLock<Vec<String>> = OnceLock::new();
    FILES.get_or_init(font_files).iter().any(|f| needles.iter().any(|n| f.starts_with(n))).then_some(family)
}

/// Lower-case font file names in the usual Linux font folders, including the host's inside a Flatpak.
fn font_files() -> Vec<String> {
    let home = crate::paths::get().home.clone();
    let roots = [
        Path::new("/usr/share/fonts").to_path_buf(),
        Path::new("/usr/local/share/fonts").to_path_buf(),
        Path::new("/run/host/fonts").to_path_buf(),
        home.join(".local/share/fonts"),
        home.join(".fonts"),
    ];
    let mut out = Vec::new();
    for root in roots {
        collect(&root, 0, &mut out);
    }
    out
}

fn collect(dir: &Path, depth: u32, out: &mut Vec<String>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for e in read.flatten() {
        let path = e.path();
        if path.is_dir() {
            if depth < 4 {
                collect(&path, depth + 1, out);
            }
        } else if let Some(name) = path.file_name() {
            out.push(name.to_string_lossy().to_lowercase());
        }
    }
}

pub fn ui() -> Font {
    Font::with_name(script_font().unwrap_or(pick().ui))
}

pub fn ui_semibold() -> Font {
    Font { weight: Weight::Semibold, ..ui() }
}

pub fn ui_bold() -> Font {
    Font { weight: Weight::Bold, ..ui() }
}

pub fn mono() -> Font {
    Font::with_name(pick().mono)
}

pub fn mono_bold() -> Font {
    Font { weight: Weight::Bold, ..mono() }
}
