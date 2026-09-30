//! Hosted providers Scoobert can send conversations to with the user's API key.

use anyhow::bail;
use serde_json::Value;

use crate::store::{CustomProvider, HostedModel};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    /// The OpenAI chat completions format, which most providers and llama-server accept.
    OpenAi,
    Anthropic,
}

/// How a provider takes the thinking effort setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingStyle {
    None,
    /// `reasoning_effort: "low" | "medium" | "high"`.
    Effort,
    /// `reasoning: { effort }`.
    OpenRouter,
    /// `enable_thinking` plus `thinking_budget`.
    Qwen,
    /// `thinking: { type: "enabled", budget_tokens }`.
    Anthropic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub base_url: String,
    pub key_url: String,
    pub thinking: ThinkingStyle,
    /// OpenAI's reasoning models reject `max_tokens`.
    pub max_tokens_field: &'static str,
}

struct Preset {
    id: &'static str,
    name: &'static str,
    api: Api,
    base_url: &'static str,
    key_url: &'static str,
    thinking: ThinkingStyle,
    max_tokens_field: &'static str,
}

const PRESETS: &[Preset] = &[
    Preset { id: "openrouter", name: "OpenRouter", api: Api::OpenAi, base_url: "https://openrouter.ai/api/v1", key_url: "https://openrouter.ai/keys", thinking: ThinkingStyle::OpenRouter, max_tokens_field: "max_tokens" },
    Preset { id: "anthropic", name: "Anthropic", api: Api::Anthropic, base_url: "https://api.anthropic.com/v1", key_url: "https://console.anthropic.com/settings/keys", thinking: ThinkingStyle::Anthropic, max_tokens_field: "max_tokens" },
    Preset { id: "openai", name: "OpenAI", api: Api::OpenAi, base_url: "https://api.openai.com/v1", key_url: "https://platform.openai.com/api-keys", thinking: ThinkingStyle::Effort, max_tokens_field: "max_completion_tokens" },
    Preset { id: "google", name: "Google Gemini", api: Api::OpenAi, base_url: "https://generativelanguage.googleapis.com/v1beta/openai", key_url: "https://aistudio.google.com/apikey", thinking: ThinkingStyle::Effort, max_tokens_field: "max_tokens" },
    Preset { id: "qwen", name: "Qwen (Alibaba Cloud)", api: Api::OpenAi, base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", key_url: "https://modelstudio.console.alibabacloud.com/", thinking: ThinkingStyle::Qwen, max_tokens_field: "max_tokens" },
    Preset { id: "deepseek", name: "DeepSeek", api: Api::OpenAi, base_url: "https://api.deepseek.com/v1", key_url: "https://platform.deepseek.com/api_keys", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
    Preset { id: "mistral", name: "Mistral", api: Api::OpenAi, base_url: "https://api.mistral.ai/v1", key_url: "https://console.mistral.ai/api-keys", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
    Preset { id: "xai", name: "xAI", api: Api::OpenAi, base_url: "https://api.x.ai/v1", key_url: "https://console.x.ai/", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
    Preset { id: "groq", name: "Groq", api: Api::OpenAi, base_url: "https://api.groq.com/openai/v1", key_url: "https://console.groq.com/keys", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
    Preset { id: "together", name: "Together AI", api: Api::OpenAi, base_url: "https://api.together.xyz/v1", key_url: "https://api.together.ai/settings/api-keys", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
    Preset { id: "fireworks", name: "Fireworks AI", api: Api::OpenAi, base_url: "https://api.fireworks.ai/inference/v1", key_url: "https://fireworks.ai/account/api-keys", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
    Preset { id: "cerebras", name: "Cerebras", api: Api::OpenAi, base_url: "https://api.cerebras.ai/v1", key_url: "https://cloud.cerebras.ai/", thinking: ThinkingStyle::None, max_tokens_field: "max_tokens" },
];

/// Built-in providers followed by the ones the user added by address.
pub fn all(custom: &[CustomProvider]) -> Vec<Provider> {
    let presets = PRESETS.iter().map(|p| Provider {
        id: p.id.into(),
        name: p.name.into(),
        api: p.api,
        base_url: p.base_url.into(),
        key_url: p.key_url.into(),
        thinking: p.thinking,
        max_tokens_field: p.max_tokens_field,
    });
    let added = custom.iter().map(|c| Provider {
        id: c.id.clone(),
        name: c.name.clone(),
        api: Api::OpenAi,
        base_url: c.base_url.trim_end_matches('/').to_string(),
        key_url: String::new(),
        thinking: ThinkingStyle::None,
        max_tokens_field: "max_tokens",
    });
    presets.chain(added).collect()
}

pub fn find(id: &str, custom: &[CustomProvider]) -> Option<Provider> {
    all(custom).into_iter().find(|p| p.id == id)
}

const DEFAULT_CONTEXT: u32 = 128_000;
const REASONING: &[&str] = &["qwen3", "qwq", "deepseek-r", "reasoner", "thinking", "gpt-oss", "magistral", "o1", "o3", "o4", "gpt-5", "claude", "gemini-2.5", "gemini-3", "grok-3-mini", "grok-4"];

/// Lists the chat models a provider offers. The key is optional for providers that list models publicly.
pub async fn list_models(http: &reqwest::Client, provider: &Provider, key: Option<&str>) -> anyhow::Result<Vec<HostedModel>> {
    let url = format!("{}/models", provider.base_url);
    let mut req = http.get(&url).timeout(std::time::Duration::from_secs(30));
    if let Some(key) = key {
        req = match provider.api {
            Api::Anthropic => req.header("x-api-key", key).header("anthropic-version", "2023-06-01"),
            Api::OpenAi => req.bearer_auth(key),
        };
    }
    let res = req.send().await?;
    let status = res.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        bail!("{} rejected the API key.", provider.name);
    }
    if !status.is_success() {
        bail!("{} returned {status} for its model list.", provider.name);
    }
    let body: Value = res.json().await?;
    let items = body["data"].as_array().cloned().unwrap_or_default();
    let mut out: Vec<HostedModel> = items.iter().filter_map(|m| model_entry(provider, m)).collect();
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

fn model_entry(provider: &Provider, m: &Value) -> Option<HostedModel> {
    let raw_id = m["id"].as_str()?;
    // Gemini lists models as "models/<id>" but takes the bare id in requests.
    let id = raw_id.strip_prefix("models/").unwrap_or(raw_id).to_string();
    let lower = id.to_lowercase();
    let skip = ["embed", "tts", "whisper", "audio", "realtime", "transcribe", "image", "dall-e", "moderation", "rerank", "guard", "search", "davinci", "babbage", "sora", "veo", "imagen", "aqa", "learnlm"];
    if skip.iter().any(|s| lower.contains(s)) {
        return None;
    }
    if provider.id == "openrouter" {
        let params = m["supported_parameters"].as_array();
        if params.is_some_and(|p| !p.iter().any(|v| v == "tools")) {
            return None;
        }
    }
    if provider.id == "openai" && !(lower.starts_with("gpt-") || lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4") || lower.starts_with("chatgpt")) {
        return None;
    }
    let name = m["name"].as_str().or(m["display_name"].as_str()).unwrap_or(&id).to_string();
    let context = m["context_length"]
        .as_u64()
        .or(m["context_window"].as_u64())
        .or(m["max_context_length"].as_u64())
        .or(m["top_provider"]["context_length"].as_u64())
        .map(|c| c.min(u32::MAX as u64) as u32)
        .unwrap_or(if provider.api == Api::Anthropic { 200_000 } else { DEFAULT_CONTEXT });
    let modalities = m["architecture"]["input_modalities"].as_array();
    let vision = match modalities {
        Some(list) => list.iter().any(|v| v == "image"),
        None => provider.api == Api::Anthropic || lower.contains("vision") || lower.contains("gpt-4o") || lower.contains("gemini") || lower.contains("-vl"),
    };
    let reasoning = match m["supported_parameters"].as_array() {
        Some(p) => p.iter().any(|v| v == "reasoning"),
        None => REASONING.iter().any(|r| lower.contains(r)),
    };
    Some(HostedModel { provider: provider.id.clone(), id, name, context, vision, reasoning })
}
