use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider, Usage};
use super::streaming::{self, SseFrame, StreamEvent};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::time::Instant;
use tokio::sync::mpsc::Sender;

/// Default Anthropic model
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-opus-4-8";

/// Anthropic Messages API version header value
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Maximum tokens to generate. The Messages API requires this parameter;
/// 4096 is supported by every Claude model.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// System content block with optional cache_control for Anthropic
#[derive(Debug, Clone, Serialize)]
struct SystemContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControlMarker>,
}

/// Cache control marker for Anthropic's prompt caching
#[derive(Debug, Clone, Serialize)]
struct CacheControlMarker {
    #[serde(rename = "type")]
    cache_type: String,
    /// Extended TTL ("1h"); omitted for the default 5-minute cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
    id: Option<String>,
    name: Option<String>,
    input: Option<Value>,
}

pub struct AnthropicClient {
    model: String,
}

impl AnthropicClient {
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }
}

impl Default for AnthropicClient {
    fn default() -> Self {
        Self::new_with_model(DEFAULT_ANTHROPIC_MODEL)
    }
}

#[async_trait]
impl ModelClient for AnthropicClient {
    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    fn api_endpoint(&self) -> String {
        let base = std::env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        format!("{}/v1/messages", base.trim_end_matches('/'))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn apply_auth(&self, request: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
        request
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
    }

    fn format_messages(&self, messages: &[Message]) -> Value {
        // The Messages API takes the system prompt as a top-level parameter,
        // not as a message role; system messages are extracted in
        // format_request_body and skipped here.
        let formatted_messages: Vec<Value> = messages
            .iter()
            .filter(|msg| msg.role != "system")
            .map(|msg| {
                let role = match msg.role.as_str() {
                    "user" | "assistant" => msg.role.as_str(),
                    // Default other roles to user
                    _ => "user",
                };
                json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect();

        json!(formatted_messages)
    }

    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
        // Check if any system message has cache_control
        let has_cache_control = messages.iter()
            .any(|msg| msg.role == "system" && msg.cache_control.is_some());

        // Format messages (excluding system)
        let formatted_messages = self.format_messages(messages);

        let mut request = json!({
            "model": self.model_name(),
            "messages": formatted_messages,
            "max_tokens": DEFAULT_MAX_TOKENS
        });

        // Handle system messages - use content blocks if cache_control is present
        let system_messages: Vec<&Message> = messages.iter()
            .filter(|msg| msg.role == "system")
            .collect();

        if !system_messages.is_empty() {
            if has_cache_control {
                // Use content block format for cache_control support
                let system_blocks: Vec<SystemContentBlock> = system_messages.iter()
                    .map(|msg| SystemContentBlock {
                        content_type: "text".to_string(),
                        text: msg.content.clone(),
                        cache_control: msg.cache_control.as_ref().map(|cc| CacheControlMarker {
                            cache_type: cc.cache_type.clone(),
                            ttl: cc.ttl.clone(),
                        }),
                    })
                    .collect();
                request["system"] = serde_json::to_value(system_blocks).unwrap_or(json!([]));
            } else {
                // Use simple string format for backward compatibility
                if let Some(system_msg) = system_messages.first() {
                    request["system"] = json!(system_msg.content);
                }
            }
        }

        // Structured outputs via forced tool use: works across all current
        // Claude models and the tool input is guaranteed to match the schema.
        if let Some(schema_str) = schema {
            if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
                request["tools"] = json!([{
                    "name": model_name.unwrap_or("response"),
                    "description": "Extract structured data according to the schema",
                    "input_schema": schema_value
                }]);
                request["tool_choice"] = json!({
                    "type": "tool",
                    "name": model_name.unwrap_or("response")
                });
            }
        }

        request
    }

    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        let response: AnthropicResponse = serde_json::from_str(response_text)?;

        // Check for tool use first (structured outputs)
        for content in &response.content {
            if content.content_type == "tool_use" {
                if let Some(input) = &content.input {
                    // Return the tool input as JSON string
                    return serde_json::to_string(input)
                        .map_err(|e| ModelClientError::ParseError(format!("Failed to serialize tool input: {e}")));
                }
            }
        }

        // Fall back to text content
        response
            .content
            .into_iter()
            .find(|content| content.content_type == "text")
            .and_then(|content| content.text)
            .ok_or_else(|| ModelClientError::ParseError("No text or tool_use content found".to_string()))
    }

    async fn send_request_structured_with_usage(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>,
    ) -> Result<(String, Option<Usage>), ModelClientError> {
        let api_key = self.get_api_key();
        let body = self.format_request_body(messages, schema, model_name);

        let mut request = self.apply_auth(client.post(self.api_endpoint()), &api_key);
        request = self.apply_cache_beta_headers(request, messages);

        let t0 = Instant::now();
        let response = request.json(&body).send().await?;
        let status = response.status();
        let text = response.text().await?;
        let latency_ms = t0.elapsed().as_millis() as i64;
        if status.is_success() {
            let parsed = self.parse_response(&text)?;
            let mut usage = self.parse_usage(&text).unwrap_or_default();
            usage.latency_ms = Some(latency_ms);
            Ok((parsed, Some(usage)))
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }

    fn parse_usage(&self, response_text: &str) -> Option<Usage> {
        // Read directly off the raw JSON body (not the typed AnthropicResponse
        // struct) so a missing/unexpected usage shape can never break the
        // main response parse path -- see the trait doc on `parse_usage`.
        //
        // Anthropic's own `input_tokens` EXCLUDES cache reads/writes, unlike
        // every other provider here (where the prompt-token count already
        // includes cached tokens). Normalize by folding both cache fields
        // into `input_tokens` so `cached_tokens <= input_tokens` uniformly
        // and the cost formula in Python stays provider-agnostic.
        //
        // Known limitation (documented, not fixed here): cache *writes* are
        // billed at a ~1.25x premium over the base input rate, but this
        // normalization folds `cache_creation_input_tokens` into
        // `input_tokens` at the base rate -- `cost_usd` slightly undercounts
        // when a request creates new cache entries.
        let v: Value = serde_json::from_str(response_text).ok()?;
        let usage = v.get("usage")?;
        let base_input = usage.get("input_tokens").and_then(Value::as_i64).unwrap_or(0);
        let cache_read = usage.get("cache_read_input_tokens").and_then(Value::as_i64).unwrap_or(0);
        let cache_creation = usage.get("cache_creation_input_tokens").and_then(Value::as_i64).unwrap_or(0);
        let output_tokens = usage.get("output_tokens").and_then(Value::as_i64);

        Some(Usage {
            input_tokens: Some(base_input + cache_read + cache_creation),
            output_tokens,
            cached_tokens: Some(cache_read),
            latency_ms: None,
        })
    }

    async fn send_request_streaming(
        &self,
        client: &Client,
        messages: &[Message],
        row: usize,
        tx: &Sender<(usize, StreamEvent)>,
    ) -> Result<(), ModelClientError> {
        let api_key = self.get_api_key();
        let mut body = self.format_request_body(messages, None, None);
        body["stream"] = json!(true);

        let mut request = self.apply_auth(client.post(self.api_endpoint()), &api_key);
        request = self.apply_cache_beta_headers(request, messages);

        let response = request.json(&body).send().await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(ModelClientError::Http(status.as_u16(), text));
        }

        streaming::drive_sse_stream(response, row, tx, |frame: &SseFrame| {
            streaming::anthropic_frame_to_event(frame)
        })
        .await;
        Ok(())
    }
}

impl AnthropicClient {
    /// Attach the prompt-caching beta header(s) when any message carries a
    /// `cache_control` marker; extended (1h) TTL needs the extra beta flag.
    /// Shared by the structured and streaming request paths.
    fn apply_cache_beta_headers(
        &self,
        request: reqwest::RequestBuilder,
        messages: &[Message],
    ) -> reqwest::RequestBuilder {
        if messages.iter().any(|msg| msg.cache_control.is_some()) {
            let needs_extended_ttl = messages.iter().any(|msg| {
                msg.cache_control.as_ref().and_then(|cc| cc.ttl.as_deref()) == Some("1h")
            });
            let beta = if needs_extended_ttl {
                "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"
            } else {
                "prompt-caching-2024-07-31"
            };
            request.header("anthropic-beta", beta)
        } else {
            request
        }
    }
}
