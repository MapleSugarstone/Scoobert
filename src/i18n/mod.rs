//! The interface language: the languages Scoobert offers, the one in use, and the text translated into each.
//!
//! Interface text is written in English in the code and passed through `tr` or `trf`. Each other language has a
//! JSON file in this folder that maps the English text to its translation, and text without an entry stays in
//! English. Text the model reads stays in English, because small models follow English instructions best.

use std::collections::HashMap;
use std::fmt::Display;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Language {
    /// The code saved in settings, such as "pt-BR".
    pub code: &'static str,
    /// The language's name in itself, as the language picker shows it.
    pub name: &'static str,
    /// The language's name in English, as the model is told it.
    pub english: &'static str,
    /// The character between groups of three digits.
    pub group: char,
    /// The character before the fraction of a number.
    pub decimal: char,
    /// A chrono format for a date with a time.
    pub date_time: &'static str,
    table: &'static str,
}

#[rustfmt::skip]
pub const LANGUAGES: &[Language] = &[
    Language { code: "en", name: "English", english: "English", group: ',', decimal: '.', date_time: "%b %-d, %Y, %-I:%M %p", table: "{}" },
    Language { code: "es", name: "Español", english: "Spanish", group: '.', decimal: ',', date_time: "%-d/%m/%Y, %H:%M", table: include_str!("es.json") },
    Language { code: "fr", name: "Français", english: "French", group: '\u{a0}', decimal: ',', date_time: "%d/%m/%Y %H:%M", table: include_str!("fr.json") },
    Language { code: "de", name: "Deutsch", english: "German", group: '.', decimal: ',', date_time: "%d.%m.%Y, %H:%M", table: include_str!("de.json") },
    Language { code: "pt-BR", name: "Português (Brasil)", english: "Brazilian Portuguese", group: '.', decimal: ',', date_time: "%d/%m/%Y, %H:%M", table: include_str!("pt-BR.json") },
    Language { code: "it", name: "Italiano", english: "Italian", group: '.', decimal: ',', date_time: "%d/%m/%Y, %H:%M", table: include_str!("it.json") },
    Language { code: "ru", name: "Русский", english: "Russian", group: '\u{a0}', decimal: ',', date_time: "%d.%m.%Y, %H:%M", table: include_str!("ru.json") },
    Language { code: "ja", name: "日本語", english: "Japanese", group: ',', decimal: '.', date_time: "%Y/%m/%d %H:%M", table: include_str!("ja.json") },
    Language { code: "ko", name: "한국어", english: "Korean", group: ',', decimal: '.', date_time: "%Y. %-m. %-d. %H:%M", table: include_str!("ko.json") },
    Language { code: "zh-Hans", name: "简体中文", english: "Simplified Chinese", group: ',', decimal: '.', date_time: "%Y/%m/%d %H:%M", table: include_str!("zh-Hans.json") },
];

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static TABLES: [OnceLock<HashMap<String, String>>; LANGUAGES.len()] = [const { OnceLock::new() }; LANGUAGES.len()];

impl PartialEq for Language {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl std::fmt::Debug for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code)
    }
}

impl Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

pub fn find(code: &str) -> Option<&'static Language> {
    LANGUAGES.iter().find(|l| l.code.eq_ignore_ascii_case(code))
}

/// Switches the interface to `code`, or to English when Scoobert does not offer it.
pub fn set(code: &str) {
    let i = LANGUAGES.iter().position(|l| l.code.eq_ignore_ascii_case(code)).unwrap_or(0);
    CURRENT.store(i, Ordering::Relaxed);
}

pub fn current() -> &'static Language {
    &LANGUAGES[CURRENT.load(Ordering::Relaxed)]
}

/// Interface text in the current language.
pub fn tr(english: &'static str) -> &'static str {
    let i = CURRENT.load(Ordering::Relaxed);
    if i == 0 {
        return english;
    }
    table(i).get(english).map(String::as_str).unwrap_or(english)
}

/// Interface text with named parts in the current language. Each `{name}` is replaced by its value.
pub fn trf(english: &'static str, args: &[(&str, &dyn Display)]) -> String {
    let mut out = tr(english).to_string();
    for (name, value) in args {
        out = out.replace(&format!("{{{name}}}"), &value.to_string());
    }
    out
}

/// Marks text that is kept in data and translated where it is shown, so the list of text to translate includes it.
pub const fn key(english: &'static str) -> &'static str {
    english
}

fn table(i: usize) -> &'static HashMap<String, String> {
    TABLES[i].get_or_init(|| serde_json::from_str(LANGUAGES[i].table).unwrap_or_default())
}

/// The offered language that matches the computer's own, if any.
pub fn system() -> Option<&'static str> {
    static FOUND: OnceLock<Option<&'static str>> = OnceLock::new();
    *FOUND.get_or_init(|| sys_locale::get_locales().find_map(|l| from_locale(&l)))
}

/// The offered language for a locale such as "es-MX", "pt_BR.UTF-8", or "zh-Hans-CN".
pub fn from_locale(locale: &str) -> Option<&'static str> {
    let locale = locale.split('.').next().unwrap_or_default().replace('_', "-").to_lowercase();
    let primary = locale.split('-').next().unwrap_or_default();
    match primary {
        // Brazilian Portuguese is the only Portuguese offered, and the only Chinese is Simplified.
        "pt" => Some("pt-BR"),
        "zh" => Some("zh-Hans"),
        _ => LANGUAGES.iter().find(|l| l.code == primary).map(|l| l.code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placeholders(text: &str) -> Vec<String> {
        let mut found: Vec<String> = regex::Regex::new(r"\{[a-z_]+\}").unwrap().find_iter(text).map(|m| m.as_str().to_string()).collect();
        found.sort();
        found
    }

    #[test]
    fn every_translation_parses_and_keeps_its_placeholders() {
        for (i, language) in LANGUAGES.iter().enumerate() {
            let parsed: HashMap<String, String> = serde_json::from_str(language.table).unwrap_or_else(|e| panic!("{}: {e}", language.code));
            for (english, translated) in &parsed {
                assert_eq!(placeholders(english), placeholders(translated), "{} changes the placeholders of {english:?}", language.code);
                assert!(!translated.trim().is_empty(), "{} leaves {english:?} empty", language.code);
            }
            assert_eq!(table(i).len(), parsed.len());
        }
    }

    #[test]
    fn locales_map_to_offered_languages() {
        assert_eq!(from_locale("es-MX"), Some("es"));
        assert_eq!(from_locale("pt_PT.UTF-8"), Some("pt-BR"));
        assert_eq!(from_locale("zh-Hant-TW"), Some("zh-Hans"));
        assert_eq!(from_locale("ja-JP"), Some("ja"));
        assert_eq!(from_locale("en-GB"), Some("en"));
        assert_eq!(from_locale("nl-NL"), None);
    }

    #[test]
    fn named_parts_are_filled_in() {
        assert_eq!(trf("Loading {model}: {secs} s", &[("model", &"Qwen"), ("secs", &3)]), "Loading Qwen: 3 s");
        assert_eq!(tr("Text without a translation"), "Text without a translation");
    }
}
