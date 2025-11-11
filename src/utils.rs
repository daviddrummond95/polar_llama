use polars::prelude::*;
use serde_json::json;
use crate::model_client::{self, Provider, create_client, Message, ModelClientError};

// Remove duplicate error type - use ModelClientError from model_client instead
pub type FetchError = ModelClientError;

// This function is useful for writing functions which
// accept pairs of List columns. Delete if unneded.
#[allow(dead_code)]
pub(crate) fn binary_amortized_elementwise<'a, T, K, F>(
    ca: &'a ListChunked,
    weights: &'a ListChunked,
    mut f: F,
) -> ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: ArrayFromIter<Option<K>>,
    F: FnMut(&Series, &Series) -> Option<K> + Copy,
{
    ca.amortized_iter()
        .zip(weights.amortized_iter())
        .map(|(lhs, rhs)| match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => f(lhs.as_ref(), rhs.as_ref()),
            _ => None,
        })
        .collect_ca(ca.name().clone())
}

pub async fn fetch_data(messages: &[String]) -> Vec<Option<String>> {
    // Default to OpenAI with gpt-4-turbo
    let client = create_client(Provider::OpenAI, "gpt-4-turbo");
    model_client::fetch_data_generic(&*client, messages).await
}

pub async fn fetch_data_with_provider(messages: &[String], provider: Provider, model: &str) -> Vec<Option<String>> {
    let client = create_client(provider, model);
    model_client::fetch_data_generic(&*client, messages).await
}

// New function to support message arrays with OpenAI default
pub async fn fetch_data_message_arrays(message_arrays: &[Vec<Message>]) -> Vec<Option<String>> {
    // Default to OpenAI with gpt-4-turbo
    let client = create_client(Provider::OpenAI, "gpt-4-turbo");
    model_client::fetch_data_generic_enhanced(&*client, message_arrays).await
}

// New function to support message arrays with specific provider
pub async fn fetch_data_message_arrays_with_provider(
    message_arrays: &[Vec<Message>], 
    provider: Provider, 
    model: &str
) -> Vec<Option<String>> {
    let client = create_client(provider, model);
    model_client::fetch_data_generic_enhanced(&*client, message_arrays).await
}

// Function to parse a string as a JSON array of messages
pub fn parse_message_json(json_str: &str) -> Result<Vec<Message>, serde_json::Error> {
    // Try parsing as a single message first
    let single_message: Result<Message, serde_json::Error> = serde_json::from_str(json_str);
    if let Ok(message) = single_message {
        return Ok(vec![message]);
    }
    
    // If that fails, try parsing as an array of messages
    serde_json::from_str(json_str)
}

// Simplified sync function that uses the model_client error types (OpenAI only, deprecated)
pub fn fetch_api_response_sync(msg: &str, model: &str) -> Result<String, FetchError> {
    // Default to OpenAI for backward compatibility
    fetch_api_response_sync_with_provider(msg, model, Provider::OpenAI)
}

// New sync function with provider support
pub fn fetch_api_response_sync_with_provider(msg: &str, model: &str, provider: Provider) -> Result<String, FetchError> {
    let agent = ureq::agent();
    let message_obj = json!({"role": "user", "content": msg});

    match provider {
        Provider::OpenAI => {
            let body = json!({
                "model": model,
                "messages": [message_obj],
                "temperature": 0.7,
                "max_tokens": 1024
            }).to_string();

            let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            let auth = format!("Bearer {api_key}");

            let response = agent.post("https://api.openai.com/v1/chat/completions")
                .set("Authorization", auth.as_str())
                .set("Content-Type", "application/json")
                .send_string(&body);

            let status = response.status();
            if response.ok() {
                let response_text = response.into_string()
                    .map_err(|e| ModelClientError::ParseError(format!("Failed to read response body: {e}")))?;
                parse_openai_response(&response_text)
            } else {
                let error_text = response.into_string()
                    .unwrap_or_else(|_| "Unknown error".to_string());
                Err(ModelClientError::Http(status, error_text))
            }
        },
        Provider::Anthropic => {
            let body = json!({
                "model": model,
                "messages": [message_obj],
                "max_tokens": 1024
            }).to_string();

            let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();

            let response = agent.post("https://api.anthropic.com/v1/messages")
                .set("x-api-key", api_key.as_str())
                .set("anthropic-version", "2023-06-01")
                .set("Content-Type", "application/json")
                .send_string(&body);

            let status = response.status();
            if response.ok() {
                let response_text = response.into_string()
                    .map_err(|e| ModelClientError::ParseError(format!("Failed to read response body: {e}")))?;
                parse_anthropic_response(&response_text)
            } else {
                let error_text = response.into_string()
                    .unwrap_or_else(|_| "Unknown error".to_string());
                Err(ModelClientError::Http(status, error_text))
            }
        },
        Provider::Gemini => {
            let body = json!({
                "contents": [{
                    "role": "user",
                    "parts": [{"text": msg}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024
                }
            }).to_string();

            let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
            let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{model}/generateContent?key={api_key}");

            let response = agent.post(&url)
                .set("Content-Type", "application/json")
                .send_string(&body);

            let status = response.status();
            if response.ok() {
                let response_text = response.into_string()
                    .map_err(|e| ModelClientError::ParseError(format!("Failed to read response body: {e}")))?;
                parse_gemini_response(&response_text)
            } else {
                let error_text = response.into_string()
                    .unwrap_or_else(|_| "Unknown error".to_string());
                Err(ModelClientError::Http(status, error_text))
            }
        },
        Provider::Groq => {
            let body = json!({
                "model": model,
                "messages": [message_obj],
                "temperature": 0.7,
                "max_tokens": 1024
            }).to_string();

            let api_key = std::env::var("GROQ_API_KEY").unwrap_or_default();
            let auth = format!("Bearer {api_key}");

            let response = agent.post("https://api.groq.com/openai/v1/chat/completions")
                .set("Authorization", auth.as_str())
                .set("Content-Type", "application/json")
                .send_string(&body);

            let status = response.status();
            if response.ok() {
                let response_text = response.into_string()
                    .map_err(|e| ModelClientError::ParseError(format!("Failed to read response body: {e}")))?;
                parse_openai_response(&response_text) // Groq uses OpenAI-compatible format
            } else {
                let error_text = response.into_string()
                    .unwrap_or_else(|_| "Unknown error".to_string());
                Err(ModelClientError::Http(status, error_text))
            }
        },
        Provider::Bedrock => {
            // Bedrock requires AWS SDK and cannot be used synchronously via HTTP
            Err(ModelClientError::ParseError(
                "Bedrock provider is not supported in synchronous mode. Please use inference_async instead.".to_string()
            ))
        }
    }
}

// Simplified OpenAI response parsing using model_client error types
fn parse_openai_response(response_text: &str) -> Result<String, ModelClientError> {
    // Use a simple JSON parsing approach since we only need the content
    let json: serde_json::Value = serde_json::from_str(response_text)?;

    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
        if let Some(first_choice) = choices.first() {
            if let Some(message) = first_choice.get("message") {
                if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                    return Ok(content.to_string());
                }
            }
        }
    }

    Err(ModelClientError::ParseError("No response content found".to_string()))
}

// Parse Anthropic API response
fn parse_anthropic_response(response_text: &str) -> Result<String, ModelClientError> {
    let json: serde_json::Value = serde_json::from_str(response_text)?;

    if let Some(content_array) = json.get("content").and_then(|c| c.as_array()) {
        for content_item in content_array {
            if let Some(content_type) = content_item.get("type").and_then(|t| t.as_str()) {
                if content_type == "text" {
                    if let Some(text) = content_item.get("text").and_then(|t| t.as_str()) {
                        return Ok(text.to_string());
                    }
                }
            }
        }
    }

    Err(ModelClientError::ParseError("No text content found in Anthropic response".to_string()))
}

// Parse Gemini API response
fn parse_gemini_response(response_text: &str) -> Result<String, ModelClientError> {
    let json: serde_json::Value = serde_json::from_str(response_text)?;

    if let Some(candidates) = json.get("candidates").and_then(|c| c.as_array()) {
        if let Some(first_candidate) = candidates.first() {
            if let Some(content) = first_candidate.get("content") {
                if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                    if let Some(first_part) = parts.first() {
                        if let Some(text) = first_part.get("text").and_then(|t| t.as_str()) {
                            return Ok(text.to_string());
                        }
                    }
                }
            }
        }
    }

    Err(ModelClientError::ParseError("No response content found in Gemini response".to_string()))
}

// New functions supporting structured outputs with validation

pub async fn fetch_data_with_provider_and_schema(
    messages: &[String],
    provider: Provider,
    model: &str,
    schema: Option<&str>,
    model_name: Option<&str>
) -> Vec<Option<String>> {
    let client = create_client(provider, model);
    model_client::fetch_data_generic_with_schema(&*client, messages, schema, model_name).await
}

pub async fn fetch_data_message_arrays_with_provider_and_schema(
    message_arrays: &[Vec<Message>],
    provider: Provider,
    model: &str,
    schema: Option<&str>,
    model_name: Option<&str>
) -> Vec<Option<String>> {
    let client = create_client(provider, model);
    model_client::fetch_data_generic_enhanced_with_schema(&*client, message_arrays, schema, model_name).await
}
