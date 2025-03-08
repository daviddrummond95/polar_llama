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
        json!(
            messages.iter().map(|msg| {
                json!({
                    "role": if msg.role == "user" { "user" } else { "model" },
                    "parts": [
                        {
                            "text": msg.content
                        }
                    ]
                })
            }).collect::<Vec<_>>()
        )
    }
    
    fn format_request_body(&self, messages: &[Message]) -> Value {
        json!({
            "contents": self.format_messages(messages),
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024
            }
        })
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
        let body = serde_json::to_string(&self.format_request_body(messages))?;
        
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