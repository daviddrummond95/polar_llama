use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::Deserialize;
use reqwest::Client;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAICompletion {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIChoice {
    index: i32,
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

pub struct OpenAIClient {
    model: String,
}

impl OpenAIClient {
    // Kept for backward compatibility but marked as deprecated
    #[deprecated(since = "0.2.0", note = "Use new_with_model instead")]
    pub fn new() -> Self {
        Self {
            model: "gpt-4-turbo".to_string(),
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

impl Default for OpenAIClient {
    fn default() -> Self {
        Self::new_with_model("gpt-4o-mini")
    }
}

#[async_trait]
impl ModelClient for OpenAIClient {
    fn provider(&self) -> Provider {
        Provider::OpenAI
    }
    
    fn api_endpoint(&self) -> String {
        "https://api.openai.com/v1/chat/completions".to_string()
    }
    
    fn model_name(&self) -> &str {
        &self.model
    }
    
    fn format_messages(&self, messages: &[Message]) -> Value {
        // OpenAI supports standard system, user, and assistant roles
        let formatted_messages = messages.iter().map(|msg| {
            let role = match msg.role.as_str() {
                "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
                "function" => "function", // Support function role for function calling
                // Default unknown roles to user
                _ => "user",
            };
            
            json!({
                "role": role,
                "content": msg.content
            })
        }).collect::<Vec<_>>();
        
        json!(formatted_messages)
    }
    
    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
        // Default OpenAI request parameters
        let mut body = json!({
            "model": self.model_name(),
            "messages": self.format_messages(messages),
            "temperature": 0.7,
            "max_tokens": 1024
        });

        // Add structured output support if schema is provided
        if let Some(schema_str) = schema {
            if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_name.unwrap_or("response"),
                        "strict": true,
                        "schema": schema_value
                    }
                });
            }
        }

        body
    }

    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        match serde_json::from_str::<OpenAICompletion>(response_text) {
            Ok(completion) => {
                if let Some(choice) = completion.choices.first() {
                    Ok(choice.message.content.clone())
                } else {
                    Err(ModelClientError::ParseError("No response content".to_string()))
                }
            },
            Err(err) => {
                Err(ModelClientError::Serialization(err))
            }
        }
    }

    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = serde_json::to_string(&self.format_request_body(messages, None, None))?;

        let response = client.post(self.api_endpoint())
            .bearer_auth(api_key)
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