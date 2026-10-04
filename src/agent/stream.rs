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
    /// Overrides the thinking budget of `thinking` without changing how the prompt is rendered.
    pub thinking_budget: Option<u32>,
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
    // A tool message holds only text, so screenshots follow the tool results as a user message.
    let mut shots: Vec<Image> = Vec::new();
    let flush = |messages: &mut Vec<Value>, shots: &mut Vec<Image>| {
        if !shots.is_empty() {
            messages.push(json!({ "role": "user", "content": openai_user_content("The screenshots from the tool results above.", shots) }));
            shots.clear();
        }
    };
    for m in req.messages {
        if !matches!(m, Message::Tool(_)) {
            flush(&mut messages, &mut shots);
        }
        match m {
            Message::User(u) => messages.push(json!({ "role": "user", "content": openai_user_content(&u.full_text(), u.sent_images()) })),
            Message::Assistant(a) => {
                let mut msg = Map::new();
                msg.insert("role".into(), "assistant".into());
                // A reply stopped while thinking has reasoning and nothing else, and still needs a content field.
                let reasoning_only = ep.local && a.resend_thinking && !a.thinking.is_empty() && a.text.is_empty() && a.tool_calls.is_empty();
                msg.insert("content".into(), if a.text.is_empty() && !reasoning_only { Value::Null } else { a.text.clone().into() });
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
                if a.text.is_empty() && a.tool_calls.is_empty() && !reasoning_only {
                    continue;
                }
                messages.push(Value::Object(msg));
            }
            Message::Tool(t) => {
                messages.push(json!({ "role": "tool", "tool_call_id": t.call_id, "content": t.output }));
                shots.extend(t.sent_images().iter().cloned());
            }
        }
    }
    flush(&mut messages, &mut shots);
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
            // Qwen's template writes the effort level into the system prompt, so a request that changes it cannot
            // reuse the cached prompt. Side requests keep the conversation's level and shrink only the budget.
            let mut kwargs = json!({ "enable_thinking": on, "preserve_thinking": true });
            if on {
                kwargs["reasoning_effort"] = qwen_effort(req.thinking).into();
            }
            body.insert("chat_template_kwargs".into(), kwargs);
            if on {
                // Reasoning and the answer share max_tokens, so an uncapped reasoning phase could leave no answer.
                body.insert("thinking_budget_tokens".into(), req.thinking_budget.unwrap_or(req.thinking.budget()).into());
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

/// Whether a chat request carries an image, which a local model reads through its image projector.
pub fn has_images(body: &Value) -> bool {
    let parts = |m: &Value| m["content"].as_array().is_some_and(|p| p.iter().any(|p| p["type"] == "image_url"));
    body["messages"].as_array().is_some_and(|messages| messages.iter().any(parts))
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
        // A call cut off mid-stream keeps its raw text, which a server cannot render as arguments and rejects with
        // the whole request, so it goes as the fields that arrived.
        Value::String(s) if serde_json::from_str::<Value>(s).is_err() => Value::Object(partial_object(s).into_iter().collect()).to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A file write cut off by Stop, reduced to the complete lines it had written. Any other call, and a write without a
/// complete line yet, is dropped.
fn salvage_write(call: ToolCall) -> Option<ToolCall> {
    if call.name != "write" {
        return None;
    }
    let fields: Vec<(String, Value)> = match &call.arguments {
        Value::Object(o) => o.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        Value::String(raw) => partial_object(raw),
        _ => return None,
    };
    let path = fields.iter().find(|(k, _)| k == "path").and_then(|(_, v)| v.as_str())?.to_string();
    let content = fields.iter().find(|(k, _)| k == "content").and_then(|(_, v)| v.as_str())?;
    let end = content.rfind('\n')? + 1;
    let content = content[..end].to_string();
    // The fields keep the model's order, so the call renders as it was generated.
    let mut args = Map::new();
    for (key, value) in fields {
        let value = match key.as_str() {
            "content" => Value::String(content.clone()),
            "path" => Value::String(path.clone()),
            _ => value,
        };
        args.insert(key, value);
    }
    Some(ToolCall { arguments: Value::Object(args), ..call })
}

/// The top-level fields of a JSON object cut off mid-stream, in order. A string cut off partway keeps what arrived,
/// and anything after it is gone.
fn partial_object(raw: &str) -> Vec<(String, Value)> {
    let mut chars = raw.trim_start().chars().peekable();
    let mut fields = Vec::new();
    if chars.next() != Some('{') {
        return fields;
    }
    // Reads a string after its opening quote, and whether it was cut off before its closing quote.
    let read_string = |chars: &mut std::iter::Peekable<std::str::Chars>| -> (String, bool) {
        let mut out = String::new();
        while let Some(c) = chars.next() {
            match c {
                '"' => return (out, false),
                '\\' => match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                            Some(ch) if hex.len() == 4 => out.push(ch),
                            _ => return (out, true),
                        }
                    }
                    Some(other) => out.push(other),
                    None => return (out, true),
                },
                c => out.push(c),
            }
        }
        (out, true)
    };
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace() || *c == ',') {
            chars.next();
        }
        if chars.next() != Some('"') {
            return fields;
        }
        let (key, cut) = read_string(&mut chars);
        if cut {
            return fields;
        }
        while chars.peek().is_some_and(|c| c.is_whitespace() || *c == ':') {
            chars.next();
        }
        match chars.peek() {
            Some('"') => {
                chars.next();
                let (value, cut) = read_string(&mut chars);
                fields.push((key, Value::String(value)));
                if cut {
                    return fields;
                }
            }
            Some(_) => {
                let word: String = std::iter::from_fn(|| chars.next_if(|c| !matches!(c, ',' | '}') && !c.is_whitespace())).collect();
                match serde_json::from_str::<Value>(&word) {
                    Ok(value) if matches!(chars.peek(), Some(',' | '}') | Some(' ' | '\n' | '\r' | '\t')) => fields.push((key, value)),
                    _ => return fields,
                }
            }
            None => return fields,
        }
    }
}

/// The effort names Qwen's chat template accepts.
fn qwen_effort(level: Thinking) -> &'static str {
    match level {
        Thinking::Off | Thinking::Low => "low",
        Thinking::Medium => "medium",
        Thinking::High => "xhigh",
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
                for img in u.sent_images() {
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
                    let input = match &c.arguments {
                        Value::Object(_) => c.arguments.clone(),
                        Value::String(raw) => Value::Object(partial_object(raw).into_iter().collect()),
                        _ => json!({}),
                    };
                    blocks.push(json!({ "type": "tool_use", "id": c.id, "name": c.name, "input": input }));
                }
                if !blocks.is_empty() {
                    push("assistant", blocks);
                }
            }
            Message::Tool(t) => {
                let content = if t.sent_images().is_empty() {
                    Value::from(t.output.clone())
                } else {
                    let mut parts = vec![json!({ "type": "text", "text": t.output })];
                    for img in t.sent_images() {
                        parts.push(json!({ "type": "image", "source": { "type": "base64", "media_type": img.mime, "data": img.data } }));
                    }
                    parts.into()
                };
                push("user", vec![json!({ "type": "tool_result", "tool_use_id": t.call_id, "content": content, "is_error": t.is_error })]);
            }
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

impl PendingCall {
    fn new(id: String, name: String) -> Self {
        PendingCall { id, name, args: String::new() }
    }

    fn push_args(&mut self, more: &str, on_delta: &mut impl FnMut(Delta)) {
        self.args.push_str(more);
        on_delta(Delta::ToolInput(self.args.len()));
    }
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
                self.calls.push(PendingCall::new(String::new(), String::new()));
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
                call.push_args(args, on_delta);
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
                    self.calls.push(PendingCall::new(block["id"].as_str().unwrap_or_default().into(), name));
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
                            self.calls[i].push_args(d["partial_json"].as_str().unwrap_or_default(), on_delta);
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
        // A stop, or a reply that ran out of room, cuts off the call being written.
        let cut = matches!(self.stop, Some(StopReason::Aborted | StopReason::Length));
        self.msg.stop = match (&self.error, self.stop) {
            (Some(_), _) => StopReason::Error,
            (None, Some(s @ (StopReason::Aborted | StopReason::Length))) => s,
            _ if !self.msg.tool_calls.is_empty() => StopReason::ToolUse,
            (None, Some(s)) => s,
            (None, None) => StopReason::Stop,
        };
        if cut {
            // A call cut off mid-stream must not run as written. A file write keeps its complete lines, which Scoobert
            // saves so the model continues the file instead of writing it again.
            self.msg.tool_calls = std::mem::take(&mut self.msg.tool_calls).into_iter().filter_map(salvage_write).collect();
        }
        self.msg.error = self.error;
        if self.usage != Usage::default() {
            self.msg.usage = Some(self.usage);
        }
        self.msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(raw: &str) -> Option<Value> {
        salvage_write(ToolCall { id: "1".into(), name: "write".into(), arguments: Value::String(raw.into()) }).map(|c| c.arguments)
    }

    #[test]
    fn a_cut_off_write_keeps_its_complete_lines() {
        let args = cut(r#"{"path": "src/a.ts", "content": "line one\nline \"two\"\nline thr"#).unwrap();
        assert_eq!(args["path"], "src/a.ts");
        assert_eq!(args["content"], "line one\nline \"two\"\n");
        let keys: Vec<&String> = args.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["path", "content"]);
    }

    #[test]
    fn a_cut_off_write_keeps_append_and_its_order() {
        let args = cut(r#"{"append": true, "path": "a.ts", "content": "x\ny"#).unwrap();
        assert_eq!(args["append"], true);
        assert_eq!(args["content"], "x\n");
        assert_eq!(args.as_object().unwrap().keys().next().unwrap(), "append");
    }

    #[test]
    fn nothing_is_kept_without_a_path_or_a_whole_line() {
        assert_eq!(cut(r#"{"path": "src/a"#), None);
        assert_eq!(cut(r#"{"path": "a.ts", "content": "half a li"#), None);
        assert_eq!(cut(r#"{"content": "x\n", "pa"#), None);
        let read = ToolCall { id: "1".into(), name: "read".into(), arguments: Value::String(r#"{"path": "a"#.into()) };
        assert_eq!(salvage_write(read), None);
    }

    #[test]
    fn escapes_cut_in_half_end_the_text() {
        assert_eq!(cut(r#"{"path": "a.ts", "content": "one\ntwo\n\u00"#).unwrap()["content"], "one\ntwo\n");
        assert_eq!(cut("{\"path\": \"a.ts\", \"content\": \"one\\n\\").unwrap()["content"], "one\n");
    }

    #[test]
    fn a_reply_out_of_room_keeps_only_its_complete_lines() {
        let mut acc = Accumulator::new("m".into());
        let mut deltas = Vec::new();
        let chunk = json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "id": "c1", "function": { "name": "write", "arguments": "{\"path\": \"a.ts\", \"content\": \"one\\ntw" } }] } }] });
        acc.openai_event(&chunk, &mut |d| deltas.push(d)).unwrap();
        acc.openai_event(&json!({ "choices": [{ "delta": {}, "finish_reason": "length" }] }), &mut |d| deltas.push(d)).unwrap();
        let msg = acc.finish();
        assert_eq!(msg.stop, StopReason::Length);
        assert_eq!(msg.tool_calls[0].arguments["content"], "one\n");
    }

    #[test]
    fn a_cut_off_call_goes_back_as_valid_arguments() {
        let sent = arguments_text(&Value::String(r#"{"path": "a.ts", "content": "one\ntw"#.into()));
        let parsed: Value = serde_json::from_str(&sent).unwrap();
        assert_eq!(parsed["path"], "a.ts");
    }
}
