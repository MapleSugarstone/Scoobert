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
    /// First-run setup may choose it and make it the default. The models for particular hardware or for drafting are
    /// left for the user to pick.
    pub pick: bool,
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
        pick: true,
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
        context_size: 32_768,
        summary: key("The newest Qwen model at 4-bit precision. A much better coder than the 9B, at about one word per second on a laptop CPU."),
        pick: true,
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
        pick: true,
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
        pick: true,
    },
    CatalogModel {
        name: "Qwen3.8-27B-UD-IQ3_XXS",
        label: key("Qwen3.8 27B, compact"),
        tier: key("For graphics cards"),
        spec: "unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS",
        revision: "4ca720788d1e01f1bff70c033e0d0028fd02e502",
        projector: false,
        download_bytes: 10_930_000_000,
        weight_bytes: 10_930_000_000,
        projector_bytes: 0,
        kv_bytes_per_token: 69_632,
        context_size: 32_768,
        summary: key("The 27B at 3-bit precision, small enough to sit entirely on a graphics card with 12 to 16 GB of memory, where it writes several times faster than the 4-bit version split between the card and the processor. It makes somewhat more mistakes."),
        pick: false,
    },
    CatalogModel {
        name: "Qwen3.6-35B-A3B-UD-IQ4_XS",
        label: "Qwen3.6 35B-A3B",
        tier: key("Quick"),
        // This repository's files add the model's own prediction layers, which made it write 1.45 times as fast.
        spec: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ4_XS",
        revision: "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        projector: true,
        download_bytes: 19_110_000_000,
        weight_bytes: 18_210_000_000,
        projector_bytes: 900_000_000,
        kv_bytes_per_token: 20_480,
        context_size: 32_768,
        summary: key("A mixture-of-experts model that uses about 3 billion of its 35 billion parameters per word, so it writes several times faster than the 27B, even on a processor, and the graphics card helps it even when it does not fit. It reads images, and it has its own prediction layers for Predict ahead. In Scoobert's coding benchmark it passed as many tasks as the 27B in a third of the time."),
        pick: false,
    },
    CatalogModel {
        name: "Qwen3.6-35B-A3B-UD-IQ3_XXS",
        label: key("Qwen3.6 35B-A3B, compact"),
        tier: key("For 16 GB of memory"),
        spec: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-IQ3_XXS",
        revision: "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        projector: true,
        download_bytes: 14_970_000_000,
        weight_bytes: 14_070_000_000,
        projector_bytes: 900_000_000,
        kv_bytes_per_token: 20_480,
        context_size: 32_768,
        summary: key("The 35B-A3B at 3-bit precision, for a computer with 16 GB of memory. In Scoobert's coding benchmark it passed one task fewer than the 4-bit version at the same speed, and it sometimes reasons in code comments until its reply runs out of room."),
        pick: false,
    },
    CatalogModel {
        name: "Qwen3.5-0.8B-Q4_K_M",
        label: "Qwen3.5 0.8B",
        tier: key("Draft"),
        spec: "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
        revision: "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
        projector: false,
        download_bytes: 530_000_000,
        weight_bytes: 530_000_000,
        projector_bytes: 0,
        kv_bytes_per_token: 12_288,
        context_size: 32_768,
        summary: key("A tiny model that drafts words ahead for a larger Qwen model, which checks them, so the larger one writes faster. Pick it under Predict ahead on the larger model. It is too weak to chat with on its own."),
        pick: false,
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
