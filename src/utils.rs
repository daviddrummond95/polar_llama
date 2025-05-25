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

// Simplified sync function that uses the model_client error types
pub fn fetch_api_response_sync(msg: &str, model: &str) -> Result<String, FetchError> {
    let agent = ureq::agent();
    let body = json!({
        "messages": [{"role": "user", "content": msg}],
        "model": model
    }).to_string();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string());
    let auth = format!("Bearer {}", api_key);
    
    let response = agent.post("https://api.openai.com/v1/chat/completions")
        .set("Authorization", auth.as_str())
        .set("Content-Type", "application/json")
        .send_string(&body);

    let status = response.status();
    if response.ok() {
        let response_text = response.into_string()
            .map_err(|e| ModelClientError::ParseError(format!("Failed to read response body: {}", e)))?;
        parse_openai_response(&response_text)
    } else {
        let error_text = response.into_string()
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(ModelClientError::Http(status, error_text))
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
