use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider, Usage};
use super::streaming::{self, StreamEvent};
use serde::Deserialize;
use reqwest::Client;
use tokio::sync::mpsc::Sender;

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
        let base = std::env::var("GROQ_BASE_URL")
            .unwrap_or_else(|_| "https://api.groq.com".to_string());
        format!("{}/openai/v1/chat/completions", base.trim_end_matches('/'))
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

    fn parse_usage(&self, response_text: &str) -> Option<Usage> {
        // Groq is OpenAI-response-shape-compatible; `prompt_tokens_details`
        // is usually absent (Groq rarely reports a cache hit), which reads
        // as `cached_tokens: None` -- never an error.
        let v: Value = serde_json::from_str(response_text).ok()?;
        let usage = v.get("usage")?;
        Some(Usage {
            input_tokens: usage.get("prompt_tokens").and_then(Value::as_i64),
            output_tokens: usage.get("completion_tokens").and_then(Value::as_i64),
            cached_tokens: usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_i64),
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
        // Groq is OpenAI-SSE-compatible.
        streaming::run_openai_sse(self, client, messages, row, tx).await
    }
}
