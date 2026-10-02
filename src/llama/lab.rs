//! The model lab: a model file's details, and variants made from a model. A variant lives in its own folder in the
//! models folder, with a `variant.json` that names it and says how it runs: on a model file of its own, or on
//! another model's file with steering vectors or adapters that llama-server applies when it loads the model.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use super::LocalModel;
use super::gguf_file::{self, GgufFile, Kv, OutTensor};
use crate::i18n::{tr, trf};
use crate::util::{Cancelled, gb};

pub const VARIANT_FILE: &str = "variant.json";
/// Prompt pairs the steering tool reads. Every pair asks the same question, so the two personas are the only
/// difference between them.
const QUESTIONS: [&str; 24] = [
    "Tell me about your day.",
    "What do you think of the ocean?",
    "Describe a city at night.",
    "How should I start learning to cook?",
    "What makes a good friend?",
    "Explain how a bicycle works.",
    "What is your favorite season and why?",
    "Write two sentences about a lost key.",
    "How do you deal with a hard problem?",
    "What would you pack for a long trip?",
    "Describe the sound of rain.",
    "Give me advice about a job interview.",
    "What is the best way to spend a free afternoon?",
    "Tell me a short story about a robot.",
    "How do plants grow?",
    "What do you think about old houses?",
    "Describe your ideal breakfast.",
    "How should a team settle a disagreement?",
    "What is interesting about the moon?",
    "Explain why people keep diaries.",
    "What would you name a new island?",
    "Describe a quiet morning in the mountains.",
    "How do you explain a computer to a child?",
    "What makes music sad or happy?",
];

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Variant {
    pub name: String,
    /// The model it was made from.
    pub made_from: String,
    /// The catalog model it descends from, which sets its default context size.
    pub family: String,
    /// Each change, oldest first, in words for the details page.
    pub changes: Vec<String>,
    /// A model file in the variant's folder, written by an edit.
    pub own_file: Option<String>,
    /// Otherwise the model file it runs on, relative to the models folder.
    pub base_file: Option<String>,
    /// The image projector it can use, relative to the models folder.
    pub projector: Option<String>,
    pub steering: Vec<Steering>,
    pub adapters: Vec<Adapter>,
}

/// A steering vector, relative to the models folder, added between `first_layer` and `last_layer`.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Steering {
    pub file: String,
    pub strength: f32,
    pub first_layer: u32,
    pub last_layer: u32,
    pub toward: String,
    pub away: String,
}

/// An adapter, relative to the models folder.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Adapter {
    pub file: String,
    pub strength: f32,
    pub source: String,
}

/// Main layers, the extra prediction layers after them, and the size of the repeating layer pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layers {
    pub main: u32,
    pub extra: u32,
    pub group: u32,
}

#[derive(Debug, Clone)]
pub struct TensorRow {
    pub name: String,
    pub shape: String,
    pub kind: String,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Details {
    pub architecture: String,
    pub layers: Layers,
    pub embedding: u64,
    pub context: u64,
    pub file_size: u64,
    pub parameters: u64,
    /// Each number format with the bytes and tensors stored in it, largest first.
    pub formats: Vec<(String, u64, usize)>,
    pub metadata: Vec<(String, String)>,
    pub tensors: Vec<TensorRow>,
    /// Whether every layer output this lab scales is stored in a format it can scale.
    pub scalable: bool,
    pub variant: Option<Variant>,
}

/// The part of each layer a strength change applies to: the output of its attention, or of its feed-forward
/// block. Recurrent layers count their memory output as attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Attention,
    FeedForward,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerEdit {
    Remove,
    Repeat,
}

/// Formats a model can be converted to, with their approximate bits per weight.
pub const CONVERSIONS: [(&str, f64); 7] = [("Q8_0", 8.5), ("Q6_K", 6.6), ("Q5_K_M", 5.7), ("Q4_K_M", 4.9), ("IQ4_XS", 4.3), ("Q3_K_M", 3.9), ("Q2_K", 3.0)];

#[derive(Debug, Clone)]
pub enum Job {
    Steer { toward: String, away: String, strength: f32, first: u32, last: u32 },
    Adapter { source: String, strength: f32 },
    Strength { part: Part, layers: Vec<u32>, percent: u32 },
    Layers { edit: LayerEdit, layers: Vec<u32> },
    Convert { target: &'static str },
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub status: String,
    pub done: u64,
    pub total: u64,
}

impl Variant {
    pub fn load(folder: &Path) -> Option<Variant> {
        serde_json::from_str(&std::fs::read_to_string(folder.join(VARIANT_FILE)).ok()?).ok()
    }

    fn save(&self, folder: &Path) -> anyhow::Result<()> {
        std::fs::write(folder.join(VARIANT_FILE), serde_json::to_string_pretty(self)?).context("Could not save the variant")
    }
}

/// The model a variant folder describes, or `None` when its files are missing.
pub fn model_from(models_dir: &Path, folder: &Path) -> Option<LocalModel> {
    let v = Variant::load(folder)?;
    let path = match (&v.own_file, &v.base_file) {
        (Some(own), _) => folder.join(own),
        (None, Some(base)) => models_dir.join(base),
        _ => return None,
    };
    if !path.is_file() || v.name.trim().is_empty() {
        return None;
    }
    let mut size = std::fs::metadata(&path).map(|m| m.len()).ok()?;
    let mut args = Vec::new();
    // llama-server splits each entry at ":", so a Windows path with a drive letter cannot go in. The server runs in
    // the models folder, and these paths are relative to it.
    if !v.steering.is_empty() {
        // The vector holds the whole measured difference at each layer, and it adds up over the layers it is applied
        // to, so the strength the user picks is spread across them. On the 9B, 2 across 17 layers gave a clear
        // persona and about 8 broke the text.
        let list: Vec<String> = v
            .steering
            .iter()
            .map(|s| format!("{}:{}", s.file, s.strength / (s.last_layer.saturating_sub(s.first_layer) + 1) as f32))
            .collect();
        let first = v.steering.iter().map(|s| s.first_layer).min().unwrap_or(1);
        let last = v.steering.iter().map(|s| s.last_layer).max().unwrap_or(first);
        args.extend(["--control-vector-scaled".into(), list.join(","), "--control-vector-layer-range".into(), first.to_string(), last.to_string()]);
    }
    if !v.adapters.is_empty() {
        size += v.adapters.iter().filter_map(|a| std::fs::metadata(models_dir.join(&a.file)).ok()).map(|m| m.len()).sum::<u64>();
        let list: Vec<String> = v.adapters.iter().map(|a| format!("{}:{}", a.file, a.strength)).collect();
        args.extend(["--lora-scaled".into(), list.join(",")]);
    }
    let mmproj = v.projector.as_ref().map(|p| models_dir.join(p)).filter(|p| p.is_file());
    let family = if v.family.is_empty() { v.made_from.clone() } else { v.family.clone() };
    Some(LocalModel { name: v.name, path, mmproj, size, args, family, variant: Some(folder.to_path_buf()) })
}

pub fn layers(g: &GgufFile) -> Layers {
    let total = g.arch_u64("block_count").unwrap_or(0) as u32;
    let extra = (g.arch_u64("nextn_predict_layers").unwrap_or(0) as u32).min(total);
    let group = g.arch_u64("full_attention_interval").filter(|&n| n > 1).unwrap_or(1) as u32;
    Layers { main: total - extra, extra, group }
}

pub fn details(model: &LocalModel) -> anyhow::Result<Details> {
    let g = GgufFile::open(&model.path)?;
    let mut formats: Vec<(String, u64, usize)> = Vec::new();
    let mut tensors = Vec::with_capacity(g.tensors.len());
    let mut parameters = 0u64;
    for (i, t) in g.tensors.iter().enumerate() {
        let bytes = g.tensor_len(i);
        parameters += t.elements();
        let kind = gguf_file::type_name(t.kind);
        match formats.iter_mut().find(|f| f.0 == kind) {
            Some(f) => {
                f.1 += bytes;
                f.2 += 1;
            }
            None => formats.push((kind.clone(), bytes, 1)),
        }
        let shape = t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" x ");
        tensors.push(TensorRow { name: t.name.clone(), shape, kind, bytes });
    }
    formats.sort_by(|a, b| b.1.cmp(&a.1));
    let metadata = g
        .kvs
        .iter()
        .map(|kv| {
            let shown = if let Some(s) = kv.as_str() {
                crate::util::clip(&s.replace('\n', " "), 120)
            } else if let Some(n) = kv.as_u64() {
                n.to_string()
            } else if let Some((_, items)) = kv.items() {
                trf("a list of {count}", &[("count", &items.len())])
            } else if kv.kind == 6 {
                f32::from_le_bytes(kv.raw[..4].try_into().unwrap_or_default()).to_string()
            } else if kv.kind == 7 {
                (kv.raw.first() == Some(&1)).to_string()
            } else {
                String::new()
            };
            (kv.key.clone(), shown)
        })
        .collect();
    let scalable = g.tensors.iter().filter(|t| part_of(&t.name).is_some()).all(|t| gguf_file::scalable(t.kind));
    Ok(Details {
        architecture: g.architecture().unwrap_or_default(),
        layers: layers(&g),
        embedding: g.arch_u64("embedding_length").unwrap_or(0),
        context: g.arch_u64("context_length").unwrap_or(0),
        file_size: std::fs::metadata(&model.path).map(|m| m.len()).unwrap_or(0),
        parameters,
        formats,
        metadata,
        tensors,
        scalable,
        variant: model.variant.as_ref().and_then(|f| Variant::load(f)),
    })
}

/// Checks that the picked layers are inside the model and, for a model with a repeating pattern of layer kinds,
/// make up whole groups, since a layer moved out of its place in the pattern no longer loads.
pub fn check_layers(l: &Layers, picked: &[u32]) -> Result<(), String> {
    if picked.is_empty() {
        return Err(tr("Add at least one layer to the list.").into());
    }
    if picked.iter().any(|&i| i >= l.main) {
        return Err(trf("Pick layers from 0 to {last}.", &[("last", &(l.main.saturating_sub(1)))]));
    }
    let g = l.group.max(1);
    if picked.iter().any(|&i| (i / g * g..i / g * g + g).any(|j| !picked.contains(&j))) {
        return Err(trf("This model's layers come in groups of {group}, so pick whole groups.", &[("group", &l.group)]));
    }
    Ok(())
}

/// The runs of consecutive layers in `picked`, as first and last layer.
pub fn blocks(picked: &[u32]) -> Vec<(u32, u32)> {
    let mut sorted = picked.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    runs(&sorted)
}

/// The runs of counting-up layers in `order`, kept in order.
pub fn runs(order: &[u32]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &i in order {
        match out.last_mut() {
            Some(b) if b.1 + 1 == i => b.1 = i,
            _ => out.push((i, i)),
        }
    }
    out
}

/// Runs of layers written as "12 to 15, 24 to 27".
pub fn describe(blocks: &[(u32, u32)]) -> String {
    let each = blocks.iter().map(|&(a, b)| if a == b { a.to_string() } else { trf("{first} to {last}", &[("first", &a), ("last", &b)]) });
    each.collect::<Vec<_>>().join(", ")
}

/// The source layer of each layer in the edited model, the extra prediction layers last. A repeated run of
/// consecutive layers runs twice in a row as one block.
pub fn layer_order(l: &Layers, edit: LayerEdit, picked: &[u32]) -> Vec<u32> {
    let mut order: Vec<u32> = match edit {
        LayerEdit::Remove => (0..l.main).filter(|i| !picked.contains(i)).collect(),
        LayerEdit::Repeat => {
            let runs = blocks(picked);
            let mut order = Vec::new();
            for i in 0..l.main {
                order.push(i);
                if let Some(&(first, _)) = runs.iter().find(|b| b.1 == i) {
                    order.extend(first..=i);
                }
            }
            order
        }
    };
    order.extend(l.main..l.main + l.extra);
    order
}

fn part_of(tensor: &str) -> Option<Part> {
    let rest = tensor.strip_prefix("blk.")?.split_once('.')?.1;
    match rest {
        "attn_output.weight" | "ssm_out.weight" => Some(Part::Attention),
        "ffn_down.weight" | "ffn_down_exps.weight" | "ffn_down_shexp.weight" => Some(Part::FeedForward),
        _ => None,
    }
}

fn layer_of(tensor: &str) -> Option<u32> {
    tensor.strip_prefix("blk.")?.split_once('.')?.0.parse().ok()
}

/// A folder name for a variant: its name without characters that file systems or llama-server's argument lists
/// cannot take.
pub fn folder_name(name: &str) -> String {
    let safe: String = name.trim().chars().map(|c| if c.is_alphanumeric() || " -_.()".contains(c) { c } else { '_' }).collect();
    safe.trim_matches(['.', ' ']).to_string()
}

/// Bytes the edit writes, to compare with free space.
pub fn estimated_size(model: &LocalModel, job: &Job) -> u64 {
    let Ok(g) = GgufFile::open(&model.path) else { return 0 };
    match job {
        Job::Steer { .. } => 50_000_000,
        Job::Adapter { .. } => 0,
        Job::Strength { .. } => std::fs::metadata(&model.path).map(|m| m.len()).unwrap_or(0),
        Job::Layers { edit, layers: picked } => {
            let l = layers(&g);
            let order = layer_order(&l, *edit, picked);
            (0..g.tensors.len())
                .map(|i| match layer_of(&g.tensors[i].name) {
                    Some(src) => order.iter().filter(|&&o| o == src).count() as u64 * g.tensor_len(i),
                    None => g.tensor_len(i),
                })
                .sum()
        }
        Job::Convert { target } => {
            let bits = CONVERSIONS.iter().find(|c| c.0 == *target).map(|c| c.1).unwrap_or(8.0);
            let params: u64 = g.tensors.iter().map(|t| t.elements()).sum();
            (params as f64 * bits / 8.0) as u64
        }
    }
}

/// Makes a variant of `model` named `name` in the models folder. Returns the new model's name.
pub async fn make(
    http: reqwest::Client,
    tools: PathBuf,
    models_dir: PathBuf,
    model: LocalModel,
    name: String,
    job: Job,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(Progress) + Send + Sync + 'static,
) -> anyhow::Result<String> {
    let name = name.trim().to_string();
    let safe = folder_name(&name);
    ensure!(!safe.is_empty(), tr("Give the new model a name."));
    ensure!(!name.contains([',', ':']), tr("A model name cannot contain commas or colons."));
    let folder = models_dir.join(&safe);
    ensure!(!folder.exists(), trf("A folder named {name} is already in the models folder. Pick another name.", &[("name", &safe)]));
    let need = estimated_size(&model, &job);
    if let Some(free) = crate::sys::free_space(&models_dir)
        && need > free
    {
        bail!(trf("This needs about {need} of disk space, and {free} is free.", &[("need", &gb(need)), ("free", &gb(free))]));
    }
    std::fs::create_dir_all(&folder).with_context(|| format!("Could not create {}", folder.display()))?;
    let progress = Arc::new(progress);
    let result = build(&http, &tools, &models_dir, &folder, &model, &name, &safe, job, &cancel, progress).await;
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&folder);
    }
    result.map(|_| name)
}

#[allow(clippy::too_many_arguments)]
async fn build(
    http: &reqwest::Client,
    tools: &Path,
    models_dir: &Path,
    folder: &Path,
    model: &LocalModel,
    name: &str,
    safe: &str,
    job: Job,
    cancel: &Arc<AtomicBool>,
    progress: Arc<impl Fn(Progress) + Send + Sync + 'static>,
) -> anyhow::Result<()> {
    let relative = |p: &Path| -> String { p.strip_prefix(models_dir).unwrap_or(p).to_string_lossy().replace('\\', "/") };
    let parent = model.variant.as_ref().and_then(|f| Variant::load(f)).unwrap_or_default();
    let mut v = Variant {
        name: name.to_string(),
        made_from: model.name.clone(),
        family: if model.family.is_empty() { model.name.clone() } else { model.family.clone() },
        changes: parent.changes.clone(),
        own_file: None,
        base_file: Some(relative(&model.path)),
        projector: model.mmproj.as_ref().map(|p| relative(p)),
        steering: parent.steering.clone(),
        adapters: parent.adapters.clone(),
    };
    let own = format!("{safe}.gguf");
    match job {
        Job::Steer { toward, away, strength, first, last } => {
            let file = folder.join("steering.gguf");
            steer(tools, model, folder, &file, &toward, &away, cancel, &progress).await?;
            v.steering.push(Steering { file: relative(&file), strength, first_layer: first, last_layer: last, toward: toward.clone(), away: away.clone() });
            v.changes.push(trf("Steered toward \"{toward}\" and away from \"{away}\" at strength {strength}, layers {first} to {last}.", &[
                ("toward", &toward),
                ("away", &away),
                ("strength", &strength),
                ("first", &first),
                ("last", &last),
            ]));
        }
        Job::Adapter { source, strength } => {
            let file = adapter(http, model, folder, &source, cancel, &progress).await?;
            v.adapters.push(Adapter { file: relative(&file), strength, source: source.clone() });
            v.changes.push(trf("Added the adapter {source} at strength {strength}.", &[("source", &source), ("strength", &strength)]));
        }
        Job::Strength { part, layers: picked, percent } => {
            let out = folder.join(&own);
            let (src, dst) = (model.path.clone(), out.clone());
            let (cancel, p, name) = (cancel.clone(), progress.clone(), name.to_string());
            let list = describe(&blocks(&picked));
            tokio::task::spawn_blocking(move || scale_layers(&src, &dst, &name, part, &picked, percent, &cancel, &*p)).await??;
            v.own_file = Some(own);
            v.base_file = None;
            let what = match part {
                Part::Attention => tr("attention output"),
                Part::FeedForward => tr("feed-forward output"),
                Part::Both => tr("attention and feed-forward output"),
            };
            v.changes.push(trf("Set the {part} of layers {layers} to {percent}%.", &[("part", &what), ("layers", &list), ("percent", &percent)]));
        }
        Job::Layers { edit, layers: picked } => {
            let out = folder.join(&own);
            let (src, dst) = (model.path.clone(), out.clone());
            let (cancel, p, name) = (cancel.clone(), progress.clone(), name.to_string());
            let list = describe(&blocks(&picked));
            tokio::task::spawn_blocking(move || edit_layers(&src, &dst, &name, edit, &picked, &cancel, &*p)).await??;
            v.own_file = Some(own);
            v.base_file = None;
            // Steering vectors and adapters are made per layer, so they no longer line up.
            v.steering.clear();
            v.adapters.clear();
            v.changes.push(match edit {
                LayerEdit::Remove => trf("Removed layers {layers}.", &[("layers", &list)]),
                LayerEdit::Repeat => trf("Repeated layers {layers}.", &[("layers", &list)]),
            });
        }
        Job::Convert { target } => {
            let out = folder.join(&own);
            convert(tools, model, &out, name, target, cancel, &progress).await?;
            v.own_file = Some(own);
            v.base_file = None;
            v.changes.push(trf("Converted to {format}.", &[("format", &target)]));
        }
    }
    v.save(folder)
}

/// Writes a copy of `src` with the chosen part of the `picked` layers multiplied by `percent` / 100.
#[allow(clippy::too_many_arguments)]
pub fn scale_layers(src: &Path, out: &Path, name: &str, part: Part, picked: &[u32], percent: u32, cancel: &AtomicBool, progress: &dyn Fn(Progress)) -> anyhow::Result<()> {
    let g = GgufFile::open(src)?;
    check_layers(&layers(&g), picked).map_err(anyhow::Error::msg)?;
    let factor = percent as f32 / 100.0;
    let mut tensors: Vec<OutTensor> = (0..g.tensors.len()).map(|i| g.out(i)).collect();
    let mut touched = 0;
    for t in &mut tensors {
        let (Some(p), Some(layer)) = (part_of(&t.name), layer_of(&t.name)) else { continue };
        if (part == Part::Both || p == part) && picked.contains(&layer) {
            ensure!(gguf_file::scalable(t.kind), trf("{tensor} is stored as {format}, which the lab cannot scale.", &[("tensor", &t.name), ("format", &gguf_file::type_name(t.kind))]));
            t.scale = Some(factor);
            touched += 1;
        }
    }
    ensure!(touched > 0, tr("Those layers have no part of that kind."));
    let kvs = renamed(&g.kvs, name);
    let status = tr("Writing the new model").to_string();
    gguf_file::write(src, out, g.version, &kvs, &tensors, g.alignment, cancel, |done, total| progress(Progress { status: status.clone(), done, total }))
}

/// Writes a copy of `src` with the `picked` layers removed or repeated once.
#[allow(clippy::too_many_arguments)]
pub fn edit_layers(src: &Path, out: &Path, name: &str, edit: LayerEdit, picked: &[u32], cancel: &AtomicBool, progress: &dyn Fn(Progress)) -> anyhow::Result<()> {
    let g = GgufFile::open(src)?;
    let l = layers(&g);
    check_layers(&l, picked).map_err(anyhow::Error::msg)?;
    let order = layer_order(&l, edit, picked);
    let main = order.len() as u32 - l.extra;
    ensure!(main >= l.group.max(1), tr("At least one group of layers has to stay."));
    let old_total = l.main + l.extra;
    let mut tensors: Vec<OutTensor> = Vec::new();
    for i in 0..g.tensors.len() {
        if layer_of(&g.tensors[i].name).is_none() {
            tensors.push(g.out(i));
        }
    }
    for (new, &source) in order.iter().enumerate() {
        let prefix = format!("blk.{source}.");
        for i in 0..g.tensors.len() {
            if let Some(rest) = g.tensors[i].name.strip_prefix(&prefix) {
                let mut t = g.out(i);
                t.name = format!("blk.{new}.{rest}");
                tensors.push(t);
            }
        }
    }
    let arch = g.architecture().unwrap_or_default();
    let mut kvs = renamed(&g.kvs, name);
    for kv in &mut kvs {
        if kv.key == format!("{arch}.block_count") {
            *kv = kv.with_u64(order.len() as u64).context("block_count has an unexpected type")?;
        } else if let Some((item, items)) = kv.items()
            && items.len() as u32 == old_total
            && item != 8
        {
            // A per-layer list, such as head counts that differ between layers, follows its layers.
            let picked: Vec<Vec<u8>> = order.iter().map(|&s| items[s as usize].clone()).collect();
            *kv = kv.with_items(item, &picked);
        }
    }
    let status = tr("Writing the new model").to_string();
    gguf_file::write(src, out, g.version, &kvs, &tensors, g.alignment, cancel, |done, total| progress(Progress { status: status.clone(), done, total }))
}

/// The metadata with the model's name set to `name`.
fn renamed(kvs: &[Kv], name: &str) -> Vec<Kv> {
    let mut out: Vec<Kv> = kvs.iter().filter(|k| k.key != "general.name").cloned().collect();
    let at = out.iter().position(|k| k.key == "general.architecture").map(|i| i + 1).unwrap_or(0);
    out.insert(at, Kv::string("general.name", name));
    out
}

/// The steering tool's prompts for one side: the persona as a system message, then each question.
fn prompt_lines(model: &Path, persona: &str) -> String {
    let chatml = super::gguf::chat_template(model).is_some_and(|t| t.contains("<|im_start|>"));
    let escape = |s: &str| s.replace('\\', "\\\\").replace('\n', "\\n");
    QUESTIONS
        .iter()
        .map(|q| {
            if chatml {
                escape(&format!("<|im_start|>system\n{persona}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"))
            } else {
                escape(&format!("{persona}\n\n{q}\n"))
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[allow(clippy::too_many_arguments)]
async fn steer(
    tools: &Path,
    model: &LocalModel,
    folder: &Path,
    out: &Path,
    toward: &str,
    away: &str,
    cancel: &Arc<AtomicBool>,
    progress: &Arc<impl Fn(Progress) + Send + Sync + 'static>,
) -> anyhow::Result<()> {
    ensure!(!toward.trim().is_empty(), tr("Describe what to steer toward."));
    let away = if away.trim().is_empty() { "You are a helpful assistant." } else { away.trim() };
    let tool = tool_path(tools, "llama-cvector-generator")?;
    let (pos, neg) = (folder.join("toward.txt"), folder.join("away.txt"));
    std::fs::write(&pos, prompt_lines(&model.path, toward.trim()))?;
    std::fs::write(&neg, prompt_lines(&model.path, away))?;
    let mut cmd = tokio::process::Command::new(tool);
    // A short context keeps the tool from reserving a cache for the model's full context length.
    cmd.arg("-m").arg(&model.path).arg("--positive-file").arg(&pos).arg("--negative-file").arg(&neg).arg("-o").arg(out);
    cmd.args(["--method", "mean", "-c", "1024", "-b", "1024", "-ub", "1024"]);
    let total = QUESTIONS.len() as u64;
    let re = regex::Regex::new(r"Evaluating prompt\[(\d+)/(\d+)\]").unwrap();
    let p = progress.clone();
    run_tool(cmd, tools, cancel, move |line| {
        if let Some(c) = re.captures(line) {
            let done = c[1].parse::<u64>().unwrap_or(0).saturating_sub(1);
            p(Progress { status: tr("Comparing the two personas").into(), done, total });
        }
    })
    .await?;
    ensure!(out.is_file(), tr("The steering tool finished without writing a steering vector."));
    Ok(())
}

async fn convert(tools: &Path, model: &LocalModel, out: &Path, name: &str, target: &str, cancel: &Arc<AtomicBool>, progress: &Arc<impl Fn(Progress) + Send + Sync + 'static>) -> anyhow::Result<()> {
    let tool = tool_path(tools, "llama-quantize")?;
    let partial = out.with_extension("gguf.partial");
    let mut cmd = tokio::process::Command::new(tool);
    cmd.arg("--allow-requantize").arg("--override-kv").arg(format!("general.name=str:{name}")).arg(&model.path).arg(&partial).arg(target);
    let re = regex::Regex::new(r"\[\s*(\d+)/\s*(\d+)\]").unwrap();
    let p = progress.clone();
    let result = run_tool(cmd, tools, cancel, move |line| {
        if let Some(c) = re.captures(line) {
            p(Progress { status: tr("Converting the model").into(), done: c[1].parse().unwrap_or(0), total: c[2].parse().unwrap_or(1) });
        }
    })
    .await;
    if let Err(e) = result {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    std::fs::rename(&partial, out).context("Could not name the converted model")
}

async fn adapter(http: &reqwest::Client, model: &LocalModel, folder: &Path, source: &str, cancel: &Arc<AtomicBool>, progress: &Arc<impl Fn(Progress) + Send + Sync + 'static>) -> anyhow::Result<PathBuf> {
    let source = source.trim();
    ensure!(!source.is_empty(), tr("Give an adapter file, or a Hugging Face repository as owner/repository."));
    let local = Path::new(source);
    let file = if local.is_file() {
        let name = local.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "adapter.gguf".into());
        let target = folder.join(format!("adapter-{}", folder_name(&name)));
        std::fs::copy(local, &target).with_context(|| format!("Could not copy {}", local.display()))?;
        target
    } else {
        let token = tokio_util::sync::CancellationToken::new();
        let watch = token.clone();
        let flag = cancel.clone();
        let watcher = tokio::spawn(async move {
            while !flag.load(Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
            watch.cancel();
        });
        let p = progress.clone();
        let result = super::download::download_adapter(http, source, folder, &token, move |d| p(Progress { status: d.status, done: d.done, total: d.total })).await;
        watcher.abort();
        result?
    };
    let a = GgufFile::open(&file)?;
    let base = GgufFile::open(&model.path)?;
    let kind = a.kv("general.type").and_then(Kv::as_str).unwrap_or_default().to_string();
    ensure!(kind == "adapter", tr("That file is not an adapter. Pick a LoRA adapter in GGUF format."));
    ensure!(
        a.architecture() == base.architecture(),
        trf("That adapter is for {theirs} models, and this model is {ours}.", &[("theirs", &a.architecture().unwrap_or_default()), ("ours", &base.architecture().unwrap_or_default())])
    );
    Ok(file)
}

fn tool_path(tools: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let file = tools.join(if cfg!(windows) { format!("{name}.exe") } else { name.to_string() });
    ensure!(file.is_file(), trf("{tool} is missing from {folder}. Reinstall Scoobert to get it.", &[("tool", &name), ("folder", &tools.display())]));
    Ok(file)
}

/// Runs a llama.cpp tool, passing each line it prints to `on_line`, and stops it when `cancel` is set.
async fn run_tool(mut cmd: tokio::process::Command, tools: &Path, cancel: &Arc<AtomicBool>, mut on_line: impl FnMut(&str) + Send + 'static) -> anyhow::Result<()> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    #[cfg(windows)]
    {
        cmd.creation_flags(0x0800_0000);
        let _ = tools;
    }
    // On Linux the tools load llama.cpp's shared libraries from their own folder, as llama-server does.
    #[cfg(not(windows))]
    {
        let mut lib = std::ffi::OsString::from(tools);
        if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
            lib.push(":");
            lib.push(old);
        }
        cmd.env("LD_LIBRARY_PATH", lib);
    }
    let mut child = cmd.spawn().context("Could not start the llama.cpp tool")?;
    let stdout = child.stdout.take().context("No output from the llama.cpp tool")?;
    let stderr = child.stderr.take().context("No output from the llama.cpp tool")?;
    let tail = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::<String>::new()));
    let keep = |tail: &Arc<std::sync::Mutex<std::collections::VecDeque<String>>>, line: &str| {
        let mut t = tail.lock().unwrap();
        t.push_back(line.to_string());
        if t.len() > 6 {
            t.pop_front();
        }
    };
    let (t1, t2) = (tail.clone(), tail.clone());
    let errors = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            keep(&t2, &line);
        }
    });
    let reader = tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            keep(&t1, &line);
            on_line(&line);
        }
    });
    let status = loop {
        tokio::select! {
            s = child.wait() => break s?,
            _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {
                if cancel.load(Ordering::SeqCst) {
                    let _ = child.kill().await;
                    return Err(Cancelled.into());
                }
            }
        }
    };
    let _ = reader.await;
    let _ = errors.await;
    if !status.success() {
        let last = tail.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
        bail!("{}\n{last}", tr("The llama.cpp tool stopped with an error:"));
    }
    Ok(())
}

/// Saves new strengths for a variant's steering vectors and adapters.
pub fn set_strengths(folder: &Path, steering: &[f32], adapters: &[f32]) -> anyhow::Result<()> {
    let mut v = Variant::load(folder).context("The variant's description is missing")?;
    for (s, &value) in v.steering.iter_mut().zip(steering) {
        s.strength = value;
    }
    for (a, &value) in v.adapters.iter_mut().zip(adapters) {
        a.strength = value;
    }
    v.save(folder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picked_layers_cover_whole_groups_of_a_mixed_model() {
        let l = Layers { main: 64, extra: 1, group: 4 };
        assert!(check_layers(&l, &[16, 17, 18, 19, 24, 25, 26, 27]).is_ok());
        assert!(check_layers(&l, &[16, 17, 18]).is_err());
        assert!(check_layers(&l, &[60, 61, 62, 63, 64]).is_err());
        assert!(check_layers(&l, &[]).is_err());
        let plain = Layers { main: 10, extra: 0, group: 1 };
        assert!(check_layers(&plain, &[3, 7]).is_ok());
    }

    #[test]
    fn layer_edits_keep_the_prediction_layer_last() {
        let l = Layers { main: 8, extra: 1, group: 4 };
        assert_eq!(layer_order(&l, LayerEdit::Remove, &[0, 1, 2, 3]), vec![4, 5, 6, 7, 8]);
        assert_eq!(layer_order(&l, LayerEdit::Repeat, &[4, 5, 6, 7]), vec![0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn separate_runs_repeat_separately_and_adjacent_ones_as_a_block() {
        let l = Layers { main: 6, extra: 0, group: 1 };
        assert_eq!(layer_order(&l, LayerEdit::Repeat, &[4, 1]), vec![0, 1, 1, 2, 3, 4, 4, 5]);
        assert_eq!(layer_order(&l, LayerEdit::Repeat, &[2, 3]), vec![0, 1, 2, 3, 2, 3, 4, 5]);
        assert_eq!(blocks(&[5, 1, 2, 3]), vec![(1, 3), (5, 5)]);
    }

    #[test]
    fn parts_and_layers_come_from_tensor_names() {
        assert_eq!(part_of("blk.3.attn_output.weight"), Some(Part::Attention));
        assert_eq!(part_of("blk.0.ssm_out.weight"), Some(Part::Attention));
        assert_eq!(part_of("blk.12.ffn_down.weight"), Some(Part::FeedForward));
        assert_eq!(part_of("blk.12.ffn_up.weight"), None);
        assert_eq!(layer_of("blk.12.ffn_up.weight"), Some(12));
        assert_eq!(layer_of("output.weight"), None);
        assert_eq!(folder_name("My: model, v2"), "My_ model_ v2");
    }
}
