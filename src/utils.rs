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
        cache_control: None,
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

// ============================================================================
// Cache-Aware Fetch Functions
// ============================================================================

/// Fetch with cache warming pattern:
/// 1. Execute first request and wait for completion (warms cache)
/// 2. Execute remaining requests in parallel (should hit cache)
///
/// This ordering is crucial for Anthropic/Bedrock where cache TTL is short
pub async fn fetch_with_cache_warming(
    message_arrays: &[Vec<Message>],
    provider: Provider,
    model: &str,
    response_schema: Option<&str>,
    response_model_name: Option<&str>,
) -> Vec<Option<String>> {
    if message_arrays.is_empty() {
        return vec![];
    }

    let client = create_client(provider, model);
    // Reuse the shared pooled client (rustls + OS cert store); no per-call TLS bypass.
    let reqwest_client = model_client::http_client().clone();

    // Step 1: Process first request to warm the cache
    let first_result = if let Some(schema) = response_schema {
        client.send_request_structured(
            &reqwest_client,
            &message_arrays[0],
            Some(schema),
            response_model_name,
        ).await
    } else {
        client.send_request(&reqwest_client, &message_arrays[0]).await
    };

    let first_result = match first_result {
        Ok(response) => {
            // Validate if schema is provided
            if let Some(schema_str) = response_schema {
                match model_client::validate_json_schema(&response, schema_str) {
                    Ok(_) => Some(response),
                    Err(validation_error) => {
                        Some(model_client::create_error_response(
                            "validation_failed",
                            &validation_error,
                            Some(&response),
                        ))
                    }
                }
            } else {
                Some(response)
            }
        }
        Err(e) => {
            eprintln!("Error fetching from {} (cache warming): {}", provider.as_str(), e);
            Some(model_client::create_error_response("api_error", &e.to_string(), None))
        }
    };

    let mut results = vec![first_result];

    // Step 2: Process remaining requests in parallel (should hit cache)
    if message_arrays.len() > 1 {
        let remaining_futures: Vec<_> = message_arrays[1..]
            .iter()
            .map(|msgs| {
                let client = create_client(provider, model);
                let reqwest_client = reqwest_client.clone();
                let messages = msgs.clone();
                let schema_owned = response_schema.map(|s| s.to_string());
                let model_name_owned = response_model_name.map(|s| s.to_string());

                async move {
                    let result = if let Some(schema) = schema_owned.as_deref() {
                        client.send_request_structured(
                            &reqwest_client,
                            &messages,
                            Some(schema),
                            model_name_owned.as_deref(),
                        ).await
                    } else {
                        client.send_request(&reqwest_client, &messages).await
                    };

                    match result {
                        Ok(response) => {
                            // Validate if schema is provided
                            if let Some(schema_str) = schema_owned.as_deref() {
                                match model_client::validate_json_schema(&response, schema_str) {
                                    Ok(_) => Some(response),
                                    Err(validation_error) => {
                                        Some(model_client::create_error_response(
                                            "validation_failed",
                                            &validation_error,
                                            Some(&response),
                                        ))
                                    }
                                }
                            } else {
                                Some(response)
                            }
                        }
                        Err(e) => {
                            eprintln!("Error fetching from {}: {}", provider.as_str(), e);
                            Some(model_client::create_error_response("api_error", &e.to_string(), None))
                        }
                    }
                }
            })
            .collect();

        let remaining_results = futures::future::join_all(remaining_futures).await;
        results.extend(remaining_results);
    }

    results
}
