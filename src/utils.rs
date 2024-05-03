use polars::prelude::*;
use reqwest::Client;
use std::error::Error;
use std::fmt;
use futures::future::join_all;
use serde_json::json;

#[derive(Debug)]
pub enum FetchError {
    Http(u16, String), // Status code and error message
    Serialization(serde_json::Error),
    Reqwest(reqwest::Error),
    ReadBody(std::io::Error), // Changed from ureq::Error to std::io::Error
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FetchError::Http(code, ref message) => write!(f, "HTTP Error {}: {}", code, message),
            FetchError::Serialization(ref err) => write!(f, "Serialization Error: {}", err),
            FetchError::ReadBody(ref err) => write!(f, "Error reading body: {}", err),
            FetchError::Reqwest(ref err) => write!(f, "Request Error: {}", err),
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
    unsafe {
        ca.amortized_iter()
            .zip(weights.amortized_iter())
            .map(|(lhs, rhs)| match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => f(lhs.as_ref(), rhs.as_ref()),
                _ => None,
            })
            .collect_ca(ca.name())
    }
}

// Initialize a global runtime for all async operations

pub async fn fetch_data(messages: &[String]) -> Vec<Option<String>> {
    let client = Client::new();
    let fetch_tasks: Vec<_> = messages.iter().map(|message| {
        let client = &client;
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string());
        async move {
            let body = format!(
                            r#"{{"messages": [{}], "model": "gpt-4-turbo"}}"#,
                            message
                        );
            let response = client.post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(api_key)
                .header("Content-Type", "application/json")
                .body(body)
                .send()
                .await;

            match response {
                Ok(res) => {
                    if res.status().is_success() {
                        res.text().await.ok()
                    } else {
                        None
                    }
                },
                Err(_) => None,
            }
        }
    }).collect();

    join_all(fetch_tasks).await
}

pub fn fetch_api_response_sync(msg: &str, model: &str) -> Result<String, FetchError> {
    let agent = ureq::agent();
    let body = json!({
        "messages": [{"role": "user", "content": msg}],
        "model": model
    }).to_string();
    // Get enviorment variable "OPENAI_API_KEY"
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string());
    let auth = format!("Bearer {}", api_key);
    let response = agent.post("https://api.openai.com/v1/chat/completions")
        .set("Authorization", auth.as_str())
        .set("Content-Type", "application/json")
        .send_string(&body);

    if response.ok() {
        response.into_string().map_err(FetchError::ReadBody)
    } else {
        Err(FetchError::Http(response.status(), response.into_string().unwrap_or_else(|_| "Unknown error".to_string())))
    }
}
