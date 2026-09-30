//! Builds chat requests and reads streamed replies in the OpenAI and Anthropic formats.

use anyhow::{Context, bail};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use super::conversation::{AssistantMessage, Image, Message, StopReason, ToolCall, Usage};
use super::providers::{Api, ThinkingStyle};
use crate::store::Thinking;
use crate::util::{Cancelled, now_millis};

/// Where a request goes and how that server takes it.
#[derive(Clone, Debug)]
pub struct Endpoint {
    pub api: Api,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// llama-server on this computer, which takes Qwen's template switches and a thinking budget.
    pub local: bool,
    pub thinking_style: ThinkingStyle,
    pub reasoning: bool,
    pub max_tokens_field: &'static str,
    pub provider_name: String,
}

pub struct ChatRequest<'a> {
    pub system: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [Value],
    pub thinking: Thinking,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub enum Delta {
    Thinking(String),
    Text(String),
    /// A tool call started; the arguments are still streaming.
    ToolCall(String),
    /// Characters of tool-call arguments received so far, such as the content of a file being written.
    ToolInput(usize),
    /// Tokens the local model has generated so far for this reply.
    Generated(u64),
    /// How much of the prompt llama-server has read, and the whole prompt's size in tokens.
    Progress { done: u64, total: u64, prompt: u64 },
}

/// The request body exactly as Scoobert sends it, which the prompt cache also renders.
pub fn payload(ep: &Endpoint, req: &ChatRequest) -> Value {
    match ep.api {
        Api::OpenAi => openai_payload(ep, req),
        Api::Anthropic => anthropic_payload(ep, req),
    }
}

fn openai_payload(ep: &Endpoint, req: &ChatRequest) -> Value {
    let mut messages = vec![json!({ "role": "system", "content": req.system })];
    for m in req.messages {
        match m {
            Message::User(u) => messages.push(json!({ "role": "user", "content": openai_user_content(&u.full_text(), &u.images) })),
            Message::Assistant(a) => {
                let mut msg = Map::new();
                msg.insert("role".into(), "assistant".into());
                msg.insert("content".into(), if a.text.is_empty() { Value::Null } else { a.text.clone().into() });
                // Qwen's template keeps earlier reasoning in the prompt, so history re-renders byte for byte
                // and the cached prompt stays usable.
                if ep.local && !a.thinking.is_empty() {
                    msg.insert("reasoning_content".into(), a.thinking.clone().into());
                }
                if !a.tool_calls.is_empty() {
                    let calls: Vec<Value> = a
                        .tool_calls
                        .iter()
                        .map(|c| {
                            json!({ "id": c.id, "type": "function", "function": { "name": c.name, "arguments": arguments_text(&c.arguments) } })
                        })
                        .collect();
                    msg.insert("tool_calls".into(), calls.into());
                }
                if a.text.is_empty() && a.tool_calls.is_empty() {
                    continue;
                }
                messages.push(Value::Object(msg));
            }
            Message::Tool(t) => messages.push(json!({ "role": "tool", "tool_call_id": t.call_id, "content": t.output })),
        }
    }
    let mut body = Map::new();
    body.insert("model".into(), ep.model.clone().into());
    body.insert("messages".into(), messages.into());
    if !req.tools.is_empty() {
        body.insert("tools".into(), req.tools.to_vec().into());
    }
    body.insert("stream".into(), true.into());
    body.insert("stream_options".into(), json!({ "include_usage": true }));
    body.insert(ep.max_tokens_field.into(), req.max_tokens.into());
    let on = req.thinking != Thinking::Off;
    if ep.local {
        if ep.reasoning {
            body.insert("chat_template_kwargs".into(), json!({ "enable_thinking": on, "preserve_thinking": true }));
            if on {
                // Reasoning and the answer share max_tokens, so an uncapped reasoning phase could leave no answer.
                body.insert("thinking_budget_tokens".into(), req.thinking.budget().into());
            }
        }
        body.insert("cache_prompt".into(), true.into());
    } else if ep.reasoning && on {
        let effort = req.thinking.label().to_lowercase();
        match ep.thinking_style {
            ThinkingStyle::Effort => {
                body.insert("reasoning_effort".into(), effort.into());
            }
            ThinkingStyle::OpenRouter => {
                body.insert("reasoning".into(), json!({ "effort": effort }));
            }
            ThinkingStyle::Qwen => {
                body.insert("enable_thinking".into(), true.into());
                body.insert("thinking_budget".into(), hosted_budget(req.thinking).into());
            }
            ThinkingStyle::None | ThinkingStyle::Anthropic => {}
        }
    } else if ep.reasoning && ep.thinking_style == ThinkingStyle::Qwen {
        body.insert("enable_thinking".into(), false.into());
    }
    Value::Object(body)
}

fn openai_user_content(text: &str, images: &[Image]) -> Value {
    if images.is_empty() {
        return text.into();
    }
    let mut parts = vec![json!({ "type": "text", "text": text })];
    for img in images {
        parts.push(json!({ "type": "image_url", "image_url": { "url": format!("data:{};base64,{}", img.mime, img.data) } }));
    }
    parts.into()
}

fn arguments_text(args: &Value) -> String {
    match args {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Thinking tokens for hosted models, which generate far faster than a CPU.
fn hosted_budget(level: Thinking) -> u32 {
    match level {
        Thinking::Off => 0,
        Thinking::Low => 2048,
        Thinking::Medium => 8192,
        Thinking::High => 24_000,
    }
}

fn anthropic_payload(ep: &Endpoint, req: &ChatRequest) -> Value {
    let thinking_on = ep.reasoning && req.thinking != Thinking::Off;
    let mut messages: Vec<Value> = Vec::new();
    let mut push = |role: &str, blocks: Vec<Value>| {
        if let Some(last) = messages.last_mut()
            && last["role"] == role
        {
            last["content"].as_array_mut().unwrap().extend(blocks);
            return;
        }
        messages.push(json!({ "role": role, "content": blocks }));
    };
    for m in req.messages {
        match m {
            Message::User(u) => {
                let mut blocks = vec![json!({ "type": "text", "text": u.full_text() })];
                for img in &u.images {
                    blocks.push(json!({ "type": "image", "source": { "type": "base64", "media_type": img.mime, "data": img.data } }));
                }
                push("user", blocks);
            }
            Message::Assistant(a) => {
                let mut blocks = Vec::new();
                if thinking_on && !a.thinking.is_empty() && let Some(sig) = &a.signature {
                    blocks.push(json!({ "type": "thinking", "thinking": a.thinking, "signature": sig }));
                }
                if !a.text.is_empty() {
                    blocks.push(json!({ "type": "text", "text": a.text }));
                }
                for c in &a.tool_calls {
                    let input = if c.arguments.is_object() { c.arguments.clone() } else { json!({}) };
                    blocks.push(json!({ "type": "tool_use", "id": c.id, "name": c.name, "input": input }));
                }
                if !blocks.is_empty() {
                    push("assistant", blocks);
                }
            }
            Message::Tool(t) => push(
                "user",
                vec![json!({ "type": "tool_result", "tool_use_id": t.call_id, "content": t.output, "is_error": t.is_error })],
            ),
        }
    }
    // Caching the conversation up to the newest message makes the next request read it at a tenth of the price.
    if let Some(block) = messages.last_mut().and_then(|m| m["content"].as_array_mut()).and_then(|b| b.last_mut()) {
        block["cache_control"] = json!({ "type": "ephemeral" });
    }
    let mut tools: Vec<Value> = req
        .tools
        .iter()
        .map(|t| {
            let f = &t["function"];
            json!({ "name": f["name"], "description": f["description"], "input_schema": f["parameters"] })
        })
        .collect();
    if let Some(last) = tools.last_mut() {
        last["cache_control"] = json!({ "type": "ephemeral" });
    }
    let mut body = json!({
        "model": ep.model,
        "system": [{ "type": "text", "text": req.system, "cache_control": { "type": "ephemeral" } }],
        "messages": messages,
        "tools": tools,
        "stream": true,
        "max_tokens": req.max_tokens,
    });
    if thinking_on {
        let budget = hosted_budget(req.thinking);
        body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
        body["max_tokens"] = (budget + req.max_tokens).into();
    }
    body
}

/// Sends a request and reads the streamed reply, calling `on_delta` as text arrives. A stopped request
/// returns what arrived so far with `StopReason::Aborted`.
pub async fn send(
    http: &reqwest::Client,
    ep: &Endpoint,
    body: &Value,
    cancel: &CancellationToken,
    mut on_delta: impl FnMut(Delta),
) -> anyhow::Result<AssistantMessage> {
    let url = match ep.api {
        Api::OpenAi => format!("{}/chat/completions", ep.base_url),
        Api::Anthropic => format!("{}/messages", ep.base_url),
    };
    let mut req = http.post(&url).json(body);
    req = match ep.api {
        Api::OpenAi if !ep.api_key.is_empty() => req.bearer_auth(&ep.api_key),
        Api::OpenAi => req,
        Api::Anthropic => req.header("x-api-key", &ep.api_key).header("anthropic-version", "2023-06-01"),
    };
    if ep.base_url.contains("openrouter.ai") {
        req = req.header("X-Title", "Scoobert");
    }
    let res = tokio::select! {
        r = req.send() => r.with_context(|| format!("Could not reach {}", ep.provider_name))?,
        _ = cancel.cancelled() => return Err(Cancelled.into()),
    };
    let status = res.status();
    if !status.is_success() {
        let text = res.text().await.unwrap_or_default();
        bail!(http_error(ep, status, &text));
    }

    let mut acc = Accumulator::new(ep.model.clone());
    let mut stream = res.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let chunk = tokio::select! {
            c = stream.next() => c,
            _ = cancel.cancelled() => {
                acc.stop = Some(StopReason::Aborted);
                break;
            }
        };
        let Some(chunk) = chunk else { break };
        let chunk = match chunk {
            Ok(c) => c,
            Err(err) => {
                acc.error = Some(format!("The connection to {} dropped: {err}", ep.provider_name));
                break;
            }
        };
        buf.extend_from_slice(&chunk);
        while let Some(end) = find_event_end(&buf) {
            let event: Vec<u8> = buf.drain(..end.0 + end.1).collect();
            let text = String::from_utf8_lossy(&event[..end.0]);
            let data: String = text
                .lines()
                .filter_map(|l| l.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(&data) else { continue };
            match ep.api {
                Api::OpenAi => acc.openai_event(&value, &mut on_delta)?,
                Api::Anthropic => acc.anthropic_event(&value, &mut on_delta)?,
            }
        }
    }
    Ok(acc.finish())
}

/// Position and separator length of the first complete server-sent event in `buf`.
fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2));
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| (p, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

fn http_error(ep: &Endpoint, status: reqwest::StatusCode, body: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().or(v["error"].as_str()).or(v["message"].as_str()).map(String::from))
        .unwrap_or_else(|| crate::util::clip(body.trim(), 300));
    match status.as_u16() {
        401 | 403 if !ep.local => format!("{} rejected the API key. Check it under Hosted models in Settings. ({detail})", ep.provider_name),
        402 => format!("{} says the account is out of credit. ({detail})", ep.provider_name),
        429 => format!("{} is limiting requests right now. Wait a moment and try again. ({detail})", ep.provider_name),
        _ => format!("{} returned {status}: {detail}", ep.provider_name),
    }
}

struct PendingCall {
    id: String,
    name: String,
    args: String,
}

struct Accumulator {
    msg: AssistantMessage,
    calls: Vec<PendingCall>,
    /// Anthropic content block index to position in `calls`.
    blocks: Vec<(u64, usize)>,
    stop: Option<StopReason>,
    usage: Usage,
    error: Option<String>,
}

impl Accumulator {
    fn new(model: String) -> Self {
        Accumulator {
            msg: AssistantMessage { model, time: now_millis(), ..Default::default() },
            calls: Vec::new(),
            blocks: Vec::new(),
            stop: None,
            usage: Usage::default(),
            error: None,
        }
    }

    fn openai_event(&mut self, v: &Value, on_delta: &mut impl FnMut(Delta)) -> anyhow::Result<()> {
        if let Some(err) = v.get("error") {
            let text = err["message"].as_str().map(String::from).unwrap_or_else(|| err.to_string());
            bail!(text);
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage.input = u["prompt_tokens"].as_u64().unwrap_or(self.usage.input);
            self.usage.output = u["completion_tokens"].as_u64().unwrap_or(self.usage.output);
            self.usage.cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(self.usage.cached);
        }
        if let Some(p) = v.get("prompt_progress").filter(|p| p.is_object()) {
            let total = p["total"].as_u64().unwrap_or(0);
            let cached = p["cache"].as_u64().unwrap_or(0);
            let done = p["processed"].as_u64().unwrap_or(0);
            on_delta(Delta::Progress { done: done.saturating_sub(cached), total: total.saturating_sub(cached), prompt: total });
        }
        let Some(choice) = v["choices"].get(0) else { return Ok(()) };
        let d = &choice["delta"];
        let reasoning = ["reasoning_content", "reasoning", "reasoning_text"]
            .iter()
            .find_map(|f| d[*f].as_str().filter(|s| !s.is_empty()));
        if let Some(r) = reasoning {
            self.msg.thinking.push_str(r);
            on_delta(Delta::Thinking(r.to_string()));
        }
        if let Some(t) = d["content"].as_str().filter(|s| !s.is_empty()) {
            self.msg.text.push_str(t);
            on_delta(Delta::Text(t.to_string()));
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            let index = tc["index"].as_u64().unwrap_or(self.calls.len().saturating_sub(1) as u64) as usize;
            while self.calls.len() <= index {
                self.calls.push(PendingCall { id: String::new(), name: String::new(), args: String::new() });
            }
            let call = &mut self.calls[index];
            if let Some(id) = tc["id"].as_str().filter(|s| !s.is_empty()) {
                call.id = id.to_string();
            }
            if let Some(name) = tc["function"]["name"].as_str().filter(|s| !s.is_empty()) {
                call.name.push_str(name);
                on_delta(Delta::ToolCall(call.name.clone()));
            }
            if let Some(args) = tc["function"]["arguments"].as_str().filter(|a| !a.is_empty()) {
                call.args.push_str(args);
                on_delta(Delta::ToolInput(call.args.len()));
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.stop = Some(match reason {
                "tool_calls" | "function_call" => StopReason::ToolUse,
                "length" => StopReason::Length,
                _ => StopReason::Stop,
            });
        }
        Ok(())
    }

    fn anthropic_event(&mut self, v: &Value, on_delta: &mut impl FnMut(Delta)) -> anyhow::Result<()> {
        match v["type"].as_str().unwrap_or_default() {
            "error" => bail!(v["error"]["message"].as_str().unwrap_or("The provider reported an error.").to_string()),
            "message_start" => {
                let u = &v["message"]["usage"];
                self.usage.cached = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                self.usage.input = u["input_tokens"].as_u64().unwrap_or(0)
                    + self.usage.cached
                    + u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let block = &v["content_block"];
                if block["type"] == "tool_use" {
                    let name = block["name"].as_str().unwrap_or_default().to_string();
                    on_delta(Delta::ToolCall(name.clone()));
                    self.calls.push(PendingCall { id: block["id"].as_str().unwrap_or_default().into(), name, args: String::new() });
                    self.blocks.push((v["index"].as_u64().unwrap_or(0), self.calls.len() - 1));
                }
            }
            "content_block_delta" => {
                let d = &v["delta"];
                match d["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        let t = d["text"].as_str().unwrap_or_default();
                        self.msg.text.push_str(t);
                        on_delta(Delta::Text(t.to_string()));
                    }
                    "thinking_delta" => {
                        let t = d["thinking"].as_str().unwrap_or_default();
                        self.msg.thinking.push_str(t);
                        on_delta(Delta::Thinking(t.to_string()));
                    }
                    "signature_delta" => {
                        self.msg.signature.get_or_insert_with(String::new).push_str(d["signature"].as_str().unwrap_or_default());
                    }
                    "input_json_delta" => {
                        let index = v["index"].as_u64().unwrap_or(0);
                        if let Some(&(_, i)) = self.blocks.iter().find(|(b, _)| *b == index) {
                            self.calls[i].args.push_str(d["partial_json"].as_str().unwrap_or_default());
                            on_delta(Delta::ToolInput(self.calls[i].args.len()));
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(out) = v["usage"]["output_tokens"].as_u64() {
                    self.usage.output = out;
                }
                if let Some(reason) = v["delta"]["stop_reason"].as_str() {
                    self.stop = Some(match reason {
                        "tool_use" => StopReason::ToolUse,
                        "max_tokens" => StopReason::Length,
                        _ => StopReason::Stop,
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(mut self) -> AssistantMessage {
        for (i, c) in self.calls.into_iter().enumerate() {
            if c.name.is_empty() {
                continue;
            }
            let arguments = if c.args.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&c.args).unwrap_or(Value::String(c.args))
            };
            let id = if c.id.is_empty() { format!("call_{}_{i}", self.msg.time) } else { c.id };
            self.msg.tool_calls.push(ToolCall { id, name: c.name, arguments });
        }
        let aborted = self.stop == Some(StopReason::Aborted);
        self.msg.stop = match (&self.error, self.stop) {
            (Some(_), _) => StopReason::Error,
            (None, Some(StopReason::Aborted)) => StopReason::Aborted,
            _ if !self.msg.tool_calls.is_empty() => StopReason::ToolUse,
            (None, Some(s)) => s,
            (None, None) => StopReason::Stop,
        };
        if aborted {
            // Tool calls cut off mid-stream have partial arguments and must not run.
            self.msg.tool_calls.clear();
        }
        self.msg.error = self.error;
        if self.usage != Usage::default() {
            self.msg.usage = Some(self.usage);
        }
        self.msg
    }
}
