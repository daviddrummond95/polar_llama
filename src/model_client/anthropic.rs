use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::{Deserialize, Serialize};
use reqwest::Client;

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
    // Kept for backward compatibility but marked as deprecated
    #[deprecated(since = "0.2.0", note = "Use new_with_model instead")]
    pub fn new() -> Self {
        Self {
            model: "claude-3-opus-20240229".to_string(),
        }
    }
    
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }
    
    // Renamed to new_with_model, kept for backwards compatibility
    #[deprecated(since = "0.2.0", note = "Use new_with_model instead")]
    pub fn with_model(model: &str) -> Self {
        Self::new_with_model(model)
    }
}

impl Default for AnthropicClient {
    fn default() -> Self {
        Self::new_with_model("claude-3-opus-20240229")
    }
}

#[async_trait]
impl ModelClient for AnthropicClient {
    fn provider(&self) -> Provider {
        Provider::Anthropic
    }
    
    fn api_endpoint(&self) -> String {
        "https://api.anthropic.com/v1/messages".to_string()
    }
    
    fn model_name(&self) -> &str {
        &self.model
    }
    
    fn format_messages(&self, messages: &[Message]) -> Value {
        // Anthropic doesn't support system messages directly in the messages array
        // We need to find a system message and extract it for the system parameter
        let mut system_prompt = None;
        let mut formatted_messages = Vec::new();
        
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // Store the first system message encountered 
                    if system_prompt.is_none() {
                        system_prompt = Some(msg.content.clone());
                    }
                    // Don't add system messages to the regular messages array
                },
                "user" | "assistant" => {
                    // Add user and assistant messages to the messages array
                    formatted_messages.push(json!({
                        "role": msg.role,
                        "content": msg.content
                    }));
                },
                _ => {
                    // Default other roles to user
                    formatted_messages.push(json!({
                        "role": "user",
                        "content": msg.content
                    }));
                }
            }
        }
        
        // Return the array of messages, system message is handled separately in format_request_body
        json!(formatted_messages)
    }
    
    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
        // Check if any system message has cache_control
        let has_cache_control = messages.iter()
            .any(|msg| msg.role == "system" && msg.cache_control.is_some());

        // Format messages (excluding system)
        let formatted_messages = self.format_messages(messages);

        // Build request with or without system parameter
        let mut request = json!({
            "model": self.model_name(),
            "messages": formatted_messages,
            "max_tokens": 4096
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

        // Add structured output support using tools if schema is provided
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
        match serde_json::from_str::<AnthropicResponse>(response_text) {
            Ok(response) => {
                // Check for tool use first (structured outputs)
                for content in &response.content {
                    if content.content_type == "tool_use" {
                        if let Some(input) = &content.input {
                            // Return the tool input as JSON string
                            return serde_json::to_string(input)
                                .map_err(|e| ModelClientError::ParseError(format!("Failed to serialize tool input: {}", e)));
                        }
                    }
                }

                // Fall back to text content
                for content in &response.content {
                    if content.content_type == "text" {
                        if let Some(text) = &content.text {
                            return Ok(text.clone());
                        }
                    }
                }
                Err(ModelClientError::ParseError("No text or tool_use content found".to_string()))
            },
            Err(err) => {
                Err(ModelClientError::Serialization(err))
            }
        }
    }
    
    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = serde_json::to_string(&self.format_request_body(messages, None, None))?;

        // Check if any message has cache_control to add the beta header
        let has_cache_control = messages.iter()
            .any(|msg| msg.cache_control.is_some());

        let mut request = client.post(self.api_endpoint())
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        // Add prompt caching beta header if cache_control is present
        if has_cache_control {
            request = request.header("anthropic-beta", "prompt-caching-2024-07-31");
        }

        let response = request.body(body).send().await?;

        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }

    async fn send_request_structured(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>
    ) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = serde_json::to_string(&self.format_request_body(messages, schema, model_name))?;

        // Check if any message has cache_control to add the beta header
        let has_cache_control = messages.iter()
            .any(|msg| msg.cache_control.is_some());

        let mut request = client.post(self.api_endpoint())
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        // Add prompt caching beta header if cache_control is present
        if has_cache_control {
            request = request.header("anthropic-beta", "prompt-caching-2024-07-31");
        }

        let response = request.body(body).send().await?;

        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }
} 