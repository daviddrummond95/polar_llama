use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider, Usage};
use serde::Deserialize;

/// Default Gemini model
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    role: String,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: String,
}

pub struct GeminiClient {
    model: String,
    api_key: Option<String>,
}

impl GeminiClient {
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }
}

impl Default for GeminiClient {
    fn default() -> Self {
        Self::new_with_model(DEFAULT_GEMINI_MODEL)
    }
}

#[async_trait]
impl ModelClient for GeminiClient {
    fn provider(&self) -> Provider {
        Provider::Gemini
    }

    fn api_endpoint(&self) -> String {
        format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn apply_auth(&self, request: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
        // Gemini authenticates with the x-goog-api-key header. Using the
        // header (rather than a ?key= query parameter) keeps the key out of
        // URLs and logs, and works for both plain and structured requests.
        request.header("x-goog-api-key", api_key)
    }

    fn format_messages(&self, messages: &[Message]) -> Value {
        // System messages are passed via the top-level system_instruction
        // field (handled in format_request_body) and skipped here.
        let formatted_messages: Vec<Value> = messages
            .iter()
            .filter(|msg| msg.role != "system")
            .map(|msg| {
                let role = match msg.role.as_str() {
                    "assistant" => "model", // Gemini uses "model" for assistant messages
                    _ => "user",
                };
                json!({
                    "role": role,
                    "parts": [{ "text": msg.content }]
                })
            })
            .collect();

        json!(formatted_messages)
    }

    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, _model_name: Option<&str>) -> Value {
        let mut body = json!({
            "contents": self.format_messages(messages),
        });

        // Use the native system_instruction field for system prompts
        if let Some(system) = messages.iter().find(|msg| msg.role == "system") {
            body["system_instruction"] = json!({
                "parts": [{ "text": system.content }]
            });
        }

        // Native structured output: constrain the response to the JSON schema.
        // Responses are additionally validated post-hoc by the caller.
        if let Some(schema_str) = schema {
            let mut generation_config = json!({
                "response_mime_type": "application/json"
            });
            if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
                generation_config["response_json_schema"] = schema_value;
            }
            body["generationConfig"] = generation_config;
        }

        body
    }

    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        let response: GeminiResponse = serde_json::from_str(response_text)?;
        response
            .candidates
            .into_iter()
            .next()
            .and_then(|candidate| candidate.content.parts.into_iter().next())
            .map(|part| part.text)
            .ok_or_else(|| ModelClientError::ParseError("No response content".to_string()))
    }

    fn get_api_key(&self) -> String {
        self.api_key.clone().unwrap_or_else(|| {
            std::env::var("GEMINI_API_KEY").unwrap_or_default()
        })
    }

    fn parse_usage(&self, response_text: &str) -> Option<Usage> {
        // Off the raw JSON body, not the typed GeminiResponse struct -- see
        // the trait doc on `parse_usage`. `promptTokenCount` already
        // includes any cached subset, so no normalization is needed.
        let v: Value = serde_json::from_str(response_text).ok()?;
        let usage = v.get("usageMetadata")?;
        Some(Usage {
            input_tokens: usage.get("promptTokenCount").and_then(Value::as_i64),
            output_tokens: usage.get("candidatesTokenCount").and_then(Value::as_i64),
            cached_tokens: usage.get("cachedContentTokenCount").and_then(Value::as_i64),
            latency_ms: None,
        })
    }
}
