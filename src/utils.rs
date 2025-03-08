use polars::prelude::*;
use std::error::Error;
use std::fmt;
use serde_json::{json, Value};
use serde::{Deserialize, Serialize};
use crate::model_client::{self, Provider, create_client};

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletion {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    index: i32,
    message: Message,
    logprobs: Option<Value>,
    finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    role: String,
    content: String,
    refusal: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
    #[serde(default)]
    prompt_tokens_details: Option<Value>,
    #[serde(default)]
    completion_tokens_details: Option<Value>,
}

#[derive(Debug)]
pub enum FetchError {
    Http(u16, String), // Status code and error message
    Serialization(serde_json::Error), // JSON parsing error
    // Reqwest(reqwest::Error), // May be needed in future
    ReadBody(std::io::Error), // Changed from ureq::Error to std::io::Error
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FetchError::Http(code, ref message) => write!(f, "HTTP Error {}: {}", code, message),
            FetchError::Serialization(ref err) => write!(f, "Serialization Error: {}", err),
            FetchError::ReadBody(ref err) => write!(f, "Error reading body: {}", err),
            // FetchError::Reqwest(ref err) => write!(f, "Request Error: {}", err),
        }
    }
}

impl Error for FetchError {}

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

// Parse OpenAI API response to extract message content
fn parse_openai_response(response_text: &str) -> Result<String, FetchError> {
    match serde_json::from_str::<ChatCompletion>(response_text) {
        Ok(completion) => {
            if let Some(choice) = completion.choices.first() {
                Ok(choice.message.content.clone())
            } else {
                Ok("No response content".to_string())
            }
        },
        Err(err) => {
            Err(FetchError::Serialization(err))
        }
    }
}

// Initialize a global runtime for all async operations

pub async fn fetch_data(messages: &[String]) -> Vec<Option<String>> {
    // Default to OpenAI with gpt-4-turbo
    let client = create_client(Provider::OpenAI, "gpt-4-turbo");
    model_client::fetch_data_generic(&*client, messages).await
}

pub async fn fetch_data_with_provider(messages: &[String], provider: Provider, model: &str) -> Vec<Option<String>> {
    let client = create_client(provider, model);
    model_client::fetch_data_generic(&*client, messages).await
}

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

    if response.ok() {
        let response_text = response.into_string().map_err(FetchError::ReadBody)?;
        parse_openai_response(&response_text)
    } else {
        Err(FetchError::Http(response.status(), response.into_string().unwrap_or_else(|_| "Unknown error".to_string())))
    }
}
