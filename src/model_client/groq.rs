use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::Deserialize;

/// Default Groq model (llama3-70b-8192 was decommissioned by Groq)
pub const DEFAULT_GROQ_MODEL: &str = "llama-3.3-70b-versatile";

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GroqCompletion {
    id: String,
    model: String,
    choices: Vec<GroqChoice>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GroqChoice {
    index: i32,
    message: GroqMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GroqMessage {
    role: String,
    content: Option<String>,
}

pub struct GroqClient {
    model: String,
}

impl GroqClient {
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }
}

impl Default for GroqClient {
    fn default() -> Self {
        Self::new_with_model(DEFAULT_GROQ_MODEL)
    }
}

#[async_trait]
impl ModelClient for GroqClient {
    fn provider(&self) -> Provider {
        Provider::Groq
    }

    fn api_endpoint(&self) -> String {
        "https://api.groq.com/openai/v1/chat/completions".to_string()
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn format_messages(&self, messages: &[Message]) -> Value {
        // Groq uses an OpenAI-compatible API, so all standard roles work
        let formatted_messages = messages.iter().map(|msg| {
            let role = match msg.role.as_str() {
                "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
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
        let mut body = json!({
            "model": self.model_name(),
            "messages": self.format_messages(messages),
        });

        // Add structured output support if schema is provided (Groq supports OpenAI format)
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
        let completion: GroqCompletion = serde_json::from_str(response_text)?;
        completion
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| ModelClientError::ParseError("No response content".to_string()))
    }
}
