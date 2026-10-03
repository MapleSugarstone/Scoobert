//! Learning from ratings: the replies the user marked good or bad, and the steering direction from the bad replies
//! toward the good ones, which a model runs with while learning is on for it. It shifts tone, style, and persona,
//! the way the model lab's steering does, and does not teach facts or skills.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

use super::LocalModel;
use super::lab::{Progress, run_tool, tool_path};
use crate::i18n::{tr, trf};
use crate::paths;

/// Ratings of each kind the steering tool needs before a direction means anything.
pub const MIN_EACH: usize = 3;
/// Prompt pairs the tool reads at most, the newest ratings first, which keeps a 27B under half an hour on a CPU.
const MAX_PAIRS: usize = 40;
/// The strongest learning the settings offer, below where steering breaks the text.
pub const MAX_STRENGTH: f32 = 3.0;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Rating {
    pub good: bool,
    /// The user's message the reply answered.
    pub asked: String,
    pub reply: String,
    pub time: i64,
}

fn safe(model: &str) -> String {
    model.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect()
}

fn ratings_file(model: &str) -> PathBuf {
    paths::get().data.join("ratings").join(format!("{}.jsonl", safe(model)))
}

/// The learned direction, inside the models folder because llama-server cuts its control vector argument at ":", so a
/// Windows path with a drive letter cannot go in. The extension keeps the model scan from listing it as a model.
pub fn vector_relative(model: &str) -> String {
    format!(".learned/{}.vector", safe(model))
}

pub fn record(model: &str, rating: Rating) -> anyhow::Result<()> {
    let file = ratings_file(model);
    std::fs::create_dir_all(file.parent().context("The ratings folder has no parent")?)?;
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&file).with_context(|| trf("Could not save the rating in {path}", &[("path", &paths::display(&file))]))?;
    writeln!(out, "{}", serde_json::to_string(&rating)?)?;
    Ok(())
}

pub fn ratings(model: &str) -> Vec<Rating> {
    let Ok(text) = std::fs::read_to_string(ratings_file(model)) else { return Vec::new() };
    text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Summary {
    pub good: usize,
    pub bad: usize,
    /// Ratings made since the direction was last built.
    pub new: usize,
    /// A direction was built from earlier ratings.
    pub learned: bool,
}

impl Summary {
    pub fn can_learn(&self) -> bool {
        self.good >= MIN_EACH && self.bad >= MIN_EACH && (self.new > 0 || !self.learned)
    }
}

pub fn summary(models_dir: &Path, model: &str) -> Summary {
    let all = ratings(model);
    let built = std::fs::metadata(models_dir.join(vector_relative(model))).and_then(|m| m.modified()).ok();
    let built_ms = built.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as i64);
    Summary {
        good: all.iter().filter(|r| r.good).count(),
        bad: all.iter().filter(|r| !r.good).count(),
        new: all.iter().filter(|r| built_ms.is_none_or(|b| r.time > b)).count(),
        learned: built.is_some(),
    }
}

/// The arguments that make llama-server add the learned direction across the middle half of the layers, at
/// `strength` spread over them as the model lab does.
pub fn args(models_dir: &Path, model: &LocalModel, strength: f32) -> Vec<String> {
    let relative = vector_relative(&model.name);
    if strength == 0.0 || !models_dir.join(&relative).is_file() {
        return Vec::new();
    }
    let layers = main_layers(&model.path).unwrap_or(0);
    if layers < 4 {
        return Vec::new();
    }
    let (first, last) = ((layers / 4).max(1), layers * 3 / 4);
    let per_layer = strength.min(MAX_STRENGTH) / (last - first + 1) as f32;
    vec!["--control-vector-scaled".into(), format!("{relative}:{per_layer}"), "--control-vector-layer-range".into(), first.to_string(), last.to_string()]
}

/// Layers that make text, without the extra ones some models keep for predicting several tokens at once.
fn main_layers(file: &Path) -> Option<u64> {
    let meta = super::gguf::read_metadata(file).ok()?;
    let Some(super::gguf::Value::Str(arch)) = meta.get("general.architecture") else { return None };
    let int = |key: &str| match meta.get(&format!("{arch}.{key}")) {
        Some(super::gguf::Value::Int(n)) => Some(*n as u64),
        _ => None,
    };
    Some(int("block_count")?.saturating_sub(int("nextn_predict_layers").unwrap_or(0)))
}

/// Builds the direction from the bad replies toward the good ones with llama-cvector-generator, which loads the model
/// on its own, so the caller stops the server first.
pub async fn build(tools: &Path, model: &LocalModel, models_dir: &Path, cancel: &Arc<AtomicBool>, progress: impl Fn(Progress) + Send + Sync + 'static) -> anyhow::Result<()> {
    let all = ratings(&model.name);
    let good: Vec<&Rating> = all.iter().rev().filter(|r| r.good).collect();
    let bad: Vec<&Rating> = all.iter().rev().filter(|r| !r.good).collect();
    ensure!(
        good.len() >= MIN_EACH && bad.len() >= MIN_EACH,
        trf("Learning needs at least {count} good and {count} bad ratings.", &[("count", &MIN_EACH)])
    );
    // Each side gets the same number of lines, which the tool pairs up, with the smaller side repeating.
    let pairs = good.len().max(bad.len()).min(MAX_PAIRS);
    let chatml = super::gguf::chat_template(&model.path).is_some_and(|t| t.contains("<|im_start|>"));
    let line = |r: &Rating| {
        let asked: String = r.asked.chars().rev().take(500).collect::<Vec<_>>().into_iter().rev().collect();
        let reply: String = r.reply.chars().take(1500).collect();
        let text = if chatml { format!("<|im_start|>user\n{asked}<|im_end|>\n<|im_start|>assistant\n{reply}") } else { format!("{asked}\n\n{reply}") };
        text.replace('\\', "\\\\").replace('\n', "\\n")
    };
    let side = |list: &[&Rating]| (0..pairs).map(|i| line(list[i % list.len()])).collect::<Vec<_>>().join("\n");
    let work = paths::get().cache.join("learning");
    std::fs::create_dir_all(&work)?;
    let (pos, neg) = (work.join("good.txt"), work.join("bad.txt"));
    std::fs::write(&pos, side(&good))?;
    std::fs::write(&neg, side(&bad))?;
    let out = models_dir.join(vector_relative(&model.name));
    std::fs::create_dir_all(out.parent().context("The learned folder has no parent")?)?;
    let partial = out.with_extension("partial");
    let mut cmd = tokio::process::Command::new(tool_path(tools, "llama-cvector-generator")?);
    cmd.arg("-m").arg(&model.path).arg("--positive-file").arg(&pos).arg("--negative-file").arg(&neg).arg("-o").arg(&partial);
    cmd.args(["--method", "mean", "-c", "4096", "-b", "4096", "-ub", "512"]);
    let re = regex::Regex::new(r"Evaluating prompt\[(\d+)/(\d+)\]").unwrap();
    let total = pairs as u64;
    let result = run_tool(cmd, tools, cancel, move |l| {
        if let Some(c) = re.captures(l) {
            let done = c[1].parse::<u64>().unwrap_or(0).saturating_sub(1);
            progress(Progress { status: tr("Comparing the good replies with the bad ones").into(), done, total });
        }
    })
    .await;
    let _ = std::fs::remove_dir_all(&work);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    ensure!(partial.is_file(), tr("The steering tool finished without writing a steering vector."));
    std::fs::rename(&partial, &out).context("Could not save the learned direction")?;
    Ok(())
}

/// Forgets a model's ratings and what it learned from them.
pub fn reset(models_dir: &Path, model: &str) {
    let _ = std::fs::remove_file(ratings_file(model));
    let _ = std::fs::remove_file(models_dir.join(vector_relative(model)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learning_waits_for_enough_ratings() {
        let s = Summary { good: 3, bad: 2, new: 5, learned: false };
        assert!(!s.can_learn());
        let s = Summary { good: 3, bad: 3, new: 0, learned: true };
        assert!(!s.can_learn());
        let s = Summary { good: 4, bad: 3, new: 1, learned: true };
        assert!(s.can_learn());
    }
}
