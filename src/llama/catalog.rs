//! Models the first-run setup and Settings offer, smallest first. `revision` pins the Hugging Face commit
//! Scoobert was tested with, and `name` matches the file name the download produces. `kv_bytes_per_token`
//! estimates memory before the download; afterward Scoobert reads the real value from the file.

use crate::i18n::key;

pub struct CatalogModel {
    pub name: &'static str,
    pub label: &'static str,
    pub tier: &'static str,
    pub spec: &'static str,
    pub revision: &'static str,
    pub projector: bool,
    pub download_bytes: u64,
    pub weight_bytes: u64,
    pub projector_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub context_size: u32,
    pub summary: &'static str,
}

pub const CATALOG: &[CatalogModel] = &[
    CatalogModel {
        name: "Qwen3.5-9B-Q4_K_M",
        label: "Qwen3.5 9B",
        tier: key("Fast"),
        spec: "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        revision: "3885219b6810b007914f3a7950a8d1b469d598a5",
        projector: true,
        download_bytes: 6_600_000_000,
        weight_bytes: 5_680_000_000,
        projector_bytes: 920_000_000,
        kv_bytes_per_token: 32_768,
        context_size: 32_768,
        summary: key("Generates about four words per second on a laptop CPU and reads screenshots, but makes more mistakes than the larger models."),
    },
    CatalogModel {
        name: "Qwen3.8-27B-UD-IQ4_XS",
        label: "Qwen3.8 27B",
        tier: key("Smart"),
        spec: "unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS",
        revision: "4ca720788d1e01f1bff70c033e0d0028fd02e502",
        projector: false,
        download_bytes: 14_250_000_000,
        weight_bytes: 14_250_000_000,
        projector_bytes: 0,
        kv_bytes_per_token: 69_632,
        context_size: 16_384,
        summary: key("The newest Qwen model at 4-bit precision. A much better coder than the 9B, at about one word per second on a laptop CPU."),
    },
    CatalogModel {
        name: "Qwen3.8-27B-UD-Q6_K",
        label: key("Qwen3.8 27B, high precision"),
        tier: key("Smarter"),
        spec: "unsloth/Qwen3.8-27B-GGUF:UD-Q6_K",
        revision: "4ca720788d1e01f1bff70c033e0d0028fd02e502",
        projector: false,
        download_bytes: 22_000_000_000,
        weight_bytes: 22_000_000_000,
        projector_bytes: 0,
        kv_bytes_per_token: 69_632,
        context_size: 32_768,
        summary: key("The same model at 6-bit precision, close to full quality. It is slower than the 4-bit version and suits computers with 32 GB of RAM or more."),
    },
    CatalogModel {
        name: "Qwen3.8-Flash-Next-UD-IQ3_XXS",
        label: "Qwen3.8 Flash-Next 125B",
        tier: key("Best"),
        spec: "unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ3_XXS",
        revision: "38bb39ee97821de2c9009abb7e93950eec396e66",
        projector: true,
        download_bytes: 82_900_000_000,
        weight_bytes: 82_000_000_000,
        projector_bytes: 900_000_000,
        kv_bytes_per_token: 131_072,
        context_size: 32_768,
        summary: key("The largest Qwen model, for workstations with 96 GB of RAM or more. It uses only about 6 billion of its parameters per word, so it can generate faster than the 27B on a desktop with enough memory. Not tested with Scoobert yet."),
    },
];

pub fn find(name: &str) -> Option<&'static CatalogModel> {
    CATALOG.iter().find(|m| m.name == name)
}

/// Free memory a catalog model needs, using the same formula as `LlamaServer::memory_needed`.
pub fn memory_needed(m: &CatalogModel) -> u64 {
    memory_formula(m.weight_bytes + m.projector_bytes, m.context_size as u64 * m.kv_bytes_per_token)
}

/// Weights, image projector, attention cache for the full context, and working buffers. The weights are
/// memory-mapped and a little of them can stay on disk, hence 85%.
pub fn memory_formula(weights: u64, kv: u64) -> u64 {
    ((weights + kv + 750_000_000) as f64 * 0.85) as u64
}
