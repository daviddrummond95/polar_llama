use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::Deserialize;
use reqwest::Client;

#[derive(Debug, Deserialize)]
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
        json!(
            messages.iter().map(|msg| {
                json!({
                    "role": if msg.role == "user" { "user" } else { "assistant" },
                    "content": msg.content
                })
            }).collect::<Vec<_>>()
        )
    }
    
    fn format_request_body(&self, messages: &[Message]) -> Value {
        json!({
            "model": self.model_name(),
            "messages": self.format_messages(messages),
            "max_tokens": 1024
        })
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