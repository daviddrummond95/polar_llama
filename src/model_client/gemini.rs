use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use serde::Deserialize;
use reqwest::Client;

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
    // Kept for backward compatibility but marked as deprecated
    #[deprecated(since = "0.2.0", note = "Use new_with_model instead")]
    pub fn new() -> Self {
        Self {
            model: "gemini-1.5-pro".to_string(),
            api_key: None,
        }
    }
    
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
            api_key: None,
        }
    }
    
    // Renamed to new_with_model, kept for backwards compatibility
    #[deprecated(since = "0.2.0", note = "Use new_with_model instead")]
    pub fn with_model(model: &str) -> Self {
        Self::new_with_model(model)
    }
    
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }
}

impl Default for GeminiClient {
    fn default() -> Self {
        Self::new_with_model("gemini-1.5-pro")
    }
}

#[async_trait]
impl ModelClient for GeminiClient {
    fn provider(&self) -> Provider {
        Provider::Gemini
    }
    
    fn api_endpoint(&self) -> String {
        format!("https://generativelanguage.googleapis.com/v1beta/models/{}/generateContent", self.model)
    }
    
    fn model_name(&self) -> &str {
        &self.model
    }
    
    fn format_messages(&self, messages: &[Message]) -> Value {
        // Check if we need to handle system message specially
        let system_message = messages.iter().find(|msg| msg.role == "system");
        
        // Create formatted messages array
        let mut formatted_messages = Vec::new();
        
        // If system message exists, add it first with special handling
        if let Some(system) = system_message {
            formatted_messages.push(json!({
                "role": "user",
                "parts": [
                    {
                        "text": format!("System instruction: {}", system.content)
                    }
                ]
            }));
        }
        
        // Add remaining non-system messages
        for msg in messages.iter().filter(|msg| msg.role != "system") {
            let role = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "model", // Gemini uses "model" for assistant messages
                _ => "user", // Default to user for unknown roles
            };
            
            formatted_messages.push(json!({
                "role": role,
                "parts": [
                    {
                        "text": msg.content
                    }
                ]
            }));
        }
        
        json!(formatted_messages)
    }
    
    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, _model_name: Option<&str>) -> Value {
        let mut body = json!({
            "contents": self.format_messages(messages),
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024
            }
        });

        // Gemini structured output support is limited, we'll use JSON mode and post-validate
        if schema.is_some() {
            body["generationConfig"]["response_mime_type"] = json!("application/json");
        }

        body
    }

    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        match serde_json::from_str::<GeminiResponse>(response_text) {
            Ok(response) => {
                if let Some(candidate) = response.candidates.first() {
                    if let Some(part) = candidate.content.parts.first() {
                        return Ok(part.text.clone());
                    }
                }
                Err(ModelClientError::ParseError("No response content".to_string()))
            },
            Err(err) => {
                Err(ModelClientError::Serialization(err))
            }
        }
    }

    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = serde_json::to_string(&self.format_request_body(messages, None, None))?;

        let url = format!("{}?key={}", self.api_endpoint(), api_key);

        let response = client.post(url)
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
    
    fn get_api_key(&self) -> String {
        self.api_key.clone().unwrap_or_else(|| {
            std::env::var("GEMINI_API_KEY").unwrap_or_default()
        })
    }
} 