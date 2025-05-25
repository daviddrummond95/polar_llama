use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::Deserialize;
use reqwest::Client;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
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
    
    fn format_request_body(&self, messages: &[Message]) -> Value {
        // Extract system message if present
        let system = messages.iter()
            .find(|msg| msg.role == "system")
            .map(|msg| msg.content.clone());
        
        // Format messages (excluding system)
        let formatted_messages = self.format_messages(messages);
        
        // Build request with or without system parameter
        let mut request = json!({
            "model": self.model_name(),
            "messages": formatted_messages,
            "max_tokens": 1024
        });
        
        // Add system parameter if we found a system message
        if let Some(system_content) = system {
            request["system"] = json!(system_content);
        }
        
        request
    }
    
    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        match serde_json::from_str::<AnthropicResponse>(response_text) {
            Ok(response) => {
                // Find the first text content
                for content in &response.content {
                    if content.content_type == "text" {
                        if let Some(text) = &content.text {
                            return Ok(text.clone());
                        }
                    }
                }
                Err(ModelClientError::ParseError("No text content found".to_string()))
            },
            Err(err) => {
                Err(ModelClientError::Serialization(err))
            }
        }
    }
    
    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = serde_json::to_string(&self.format_request_body(messages))?;
        
        let response = client.post(self.api_endpoint())
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?;
            
        let status = response.status();
        let text = response.text().await?;
        
        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }
} 