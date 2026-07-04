use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider, EmbeddingClient};
use serde::{Deserialize, Serialize};
use reqwest::Client;

/// Default OpenAI chat model
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAICompletion {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIChoice {
    index: i32,
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
}

pub struct OpenAIClient {
    model: String,
}

impl OpenAIClient {
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }
}

impl Default for OpenAIClient {
    fn default() -> Self {
        Self::new_with_model(DEFAULT_OPENAI_MODEL)
    }
}

#[async_trait]
impl ModelClient for OpenAIClient {
    fn provider(&self) -> Provider {
        Provider::OpenAI
    }

    fn api_endpoint(&self) -> String {
        let base = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        format!("{}/v1/chat/completions", base.trim_end_matches('/'))
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
                "tool" => "tool",
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
        // Note: no hardcoded temperature/max_tokens — newer OpenAI models
        // (o-series, gpt-5 family) reject those parameters, so we rely on
        // provider defaults for maximum compatibility.
        let mut body = json!({
            "model": self.model_name(),
            "messages": self.format_messages(messages),
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
        let completion: OpenAICompletion = serde_json::from_str(response_text)?;
        completion
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| ModelClientError::ParseError("No response content".to_string()))
    }
}

// ============================================================================
// Embedding Client Implementation
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    input: &'a [String],
    model: &'a str,
    encoding_format: &'a str,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIEmbeddingResponse {
    object: String,
    data: Vec<OpenAIEmbeddingData>,
    model: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIEmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f64>,
}

pub struct OpenAIEmbeddingClient {
    model: String,
    dimensions: usize,
}

impl OpenAIEmbeddingClient {
    pub fn new() -> Self {
        Self::new_with_model("text-embedding-3-small")
    }

    pub fn new_with_model(model: &str) -> Self {
        // Determine dimensions based on model
        let dimensions = match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // Default
        };

        Self {
            model: model.to_string(),
            dimensions,
        }
    }
}

impl Default for OpenAIEmbeddingClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingClient for OpenAIEmbeddingClient {
    fn provider(&self) -> Provider {
        Provider::OpenAI
    }

    fn embedding_endpoint(&self) -> String {
        "https://api.openai.com/v1/embeddings".to_string()
    }

    fn embedding_model(&self) -> &str {
        &self.model
    }

    fn embedding_dimensions(&self) -> usize {
        self.dimensions
    }

    async fn generate_embeddings(
        &self,
        client: &Client,
        texts: &[String],
    ) -> Result<Vec<Vec<f64>>, ModelClientError> {
        let api_key = self.get_api_key();

        let request_body = OpenAIEmbeddingRequest {
            input: texts,
            model: &self.model,
            encoding_format: "float",
        };

        let response = client
            .post(self.embedding_endpoint())
            .bearer_auth(api_key)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            let embedding_response: OpenAIEmbeddingResponse =
                serde_json::from_str(&text)?;

            // Sort by index to ensure correct order
            let mut sorted_data = embedding_response.data;
            sorted_data.sort_by_key(|d| d.index);

            Ok(sorted_data.into_iter().map(|d| d.embedding).collect())
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }
}
