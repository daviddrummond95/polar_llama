use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::{Deserialize, Serialize};
use reqwest::Client;

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

    async fn send_request_structured(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>,
    ) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = self.format_request_body(messages, schema, model_name);

        let mut request = self.apply_auth(client.post(self.api_endpoint()), &api_key);

        // Prompt-caching beta header(s); extended (1h) TTL needs the extra beta.
        if messages.iter().any(|msg| msg.cache_control.is_some()) {
            let needs_extended_ttl = messages.iter().any(|msg| {
                msg.cache_control.as_ref().and_then(|cc| cc.ttl.as_deref()) == Some("1h")
            });
            let beta = if needs_extended_ttl {
                "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"
            } else {
                "prompt-caching-2024-07-31"
            };
            request = request.header("anthropic-beta", beta);
        }

        let response = request.json(&body).send().await?;
        let status = response.status();
        let text = response.text().await?;
        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }
}
