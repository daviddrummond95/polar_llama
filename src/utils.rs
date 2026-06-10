use polars::prelude::*;
use std::sync::LazyLock;
use tokio::runtime::Runtime;
use crate::model_client::{self, Provider, create_client, create_embedding_client, Message, ModelClientError};

// Remove duplicate error type - use ModelClientError from model_client instead
pub type FetchError = ModelClientError;

/// Global Tokio runtime shared by all blocking entry points.
pub(crate) static RT: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

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
    fetch_data_with_provider(messages, Provider::OpenAI, crate::model_client::openai::DEFAULT_OPENAI_MODEL).await
}

pub async fn fetch_data_with_provider(messages: &[String], provider: Provider, model: &str) -> Vec<Option<String>> {
    let client = create_client(provider, model);
    model_client::fetch_data_generic(&*client, messages).await
}

// New function to support message arrays with OpenAI default
pub async fn fetch_data_message_arrays(message_arrays: &[Vec<Message>]) -> Vec<Option<String>> {
    fetch_data_message_arrays_with_provider(
        message_arrays,
        Provider::OpenAI,
        crate::model_client::openai::DEFAULT_OPENAI_MODEL,
    ).await
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

// Sync entry point: blocks on the shared async client, so the request logic
// (auth, formatting, parsing) lives in exactly one place per provider.
pub fn fetch_api_response_sync_with_provider(msg: &str, model: &str, provider: Provider) -> Result<String, FetchError> {
    let client = create_client(provider, model);
    let messages = vec![Message {
        role: "user".to_string(),
        content: msg.to_string(),
    }];

    RT.block_on(async move {
        client.send_request(model_client::http_client(), &messages).await
    })
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

// ============================================================================
// Embedding Functions
// ============================================================================

/// Fetch embeddings with default provider (OpenAI) and model
pub async fn fetch_embeddings(texts: &[String]) -> Vec<Option<Vec<f64>>> {
    fetch_embeddings_with_provider(texts, Provider::OpenAI, "text-embedding-3-small").await
}

/// Fetch embeddings with specific provider and model
pub async fn fetch_embeddings_with_provider(
    texts: &[String],
    provider: Provider,
    model: &str
) -> Vec<Option<Vec<f64>>> {
    let client = create_embedding_client(provider, model);
    model_client::fetch_embeddings_generic(&*client, texts).await
}
