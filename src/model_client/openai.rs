use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct OpenAICompletion {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    index: i32,
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
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
        json!(
            messages.iter().map(|msg| {
                json!({
                    "role": msg.role,
                    "content": msg.content
                })
            }).collect::<Vec<_>>()
        )
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
} 