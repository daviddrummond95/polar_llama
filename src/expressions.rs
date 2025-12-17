#![allow(clippy::unused_unit)]
use crate::utils::*;
use crate::model_client::{Provider, Message};
use crate::cost;
use once_cell::sync::Lazy;
use polars::prelude::*;
use polars_core::prelude::CompatLevel;
use polars_core::chunked_array::builder::ListPrimitiveChunkedBuilder;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::borrow::Cow;
use tokio::runtime::Runtime;
use std::str::FromStr;
use polars::datatypes::DataType;

// Initialize a global runtime for all async operations
static RT: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

// Helper function to run async operations in a way that allows true parallelization
// Instead of directly blocking on async work, we spawn it as a task and then block on the handle
// This allows multiple threads to spawn tasks that run concurrently on the runtime's thread pool
fn run_async<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let handle = RT.spawn(future);
    RT.block_on(handle).expect("Task panicked")
}

#[derive(Debug, Deserialize)]
pub struct InferenceKwargs {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    response_schema: Option<String>,
    #[serde(default)]
    response_model_name: Option<String>,
}

fn parse_provider(provider_str: &str) -> Option<Provider> {
    Provider::from_str(provider_str).ok()
}

/// Get default model for a given provider
fn get_default_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "gpt-4-turbo",
        Provider::Anthropic => "claude-3-opus-20240229",
        Provider::Gemini => "gemini-1.5-pro",
        Provider::Groq => "llama3-70b-8192",
        Provider::Bedrock => "anthropic.claude-3-haiku-20240307-v1:0",
    }
}

// This polars_expr annotation registers the function with Polars at build time
#[polars_expr(output_type=String)]
fn inference(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;

    // Determine provider and model
    let provider = match &kwargs.provider {
        Some(provider_str) => parse_provider(provider_str).unwrap_or(Provider::OpenAI),
        None => Provider::OpenAI,
    };

    let model = kwargs.model.unwrap_or_else(|| get_default_model(provider).to_string());

    let out = ca.apply(|opt_value| {
        opt_value.map(|value| {
            let response = crate::utils::fetch_api_response_sync_with_provider(value, &model, provider);
            Cow::Owned(response.unwrap_or_default())
        })
    });
    Ok(out.into_series())
}

// Register the asynchronous inference function with Polars
#[polars_expr(output_type=String)]
fn inference_async(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    let input_series = &inputs[0];

    // Handle empty series or null dtype - return empty String series
    if input_series.is_empty() || input_series.dtype() == &polars::datatypes::DataType::Null {
        return Ok(StringChunked::from_iter_options(
            input_series.name().clone(),
            std::iter::empty::<Option<String>>(),
        ).into_series());
    }

    let ca: &StringChunked = input_series.str()?;

    // Collect all messages, keeping track of their original indices
    let mut messages_with_indices: Vec<(usize, String)> = Vec::new();
    for (idx, opt_value) in ca.into_iter().enumerate() {
        if let Some(value) = opt_value {
            messages_with_indices.push((idx, value.to_owned()));
        }
    }

    // Extract just the messages for the API calls
    let messages: Vec<String> = messages_with_indices.iter().map(|(_, msg)| msg.clone()).collect();

    // Get results based on provider and model
    let api_results = if kwargs.response_schema.is_some() {
        // Use structured output with validation
        // Clone the strings so we can move them into async blocks
        let schema_owned = kwargs.response_schema.clone();
        let model_name_owned = kwargs.response_model_name.clone();

        match (&kwargs.provider, &kwargs.model) {
            (Some(provider_str), Some(model)) => {
                // Try to parse provider string to Provider enum
                if let Some(provider) = parse_provider(provider_str) {
                    let messages_owned = messages.clone();
                    let model_owned = model.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        fetch_data_with_provider_and_schema(&messages_owned, provider, &model_owned, schema.as_deref(), model_name.as_deref()).await
                    })
                } else {
                    let messages_owned = messages.clone();
                    let model_owned = model.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        fetch_data_with_provider_and_schema(&messages_owned, Provider::OpenAI, &model_owned, schema.as_deref(), model_name.as_deref()).await
                    })
                }
            },
            (Some(provider_str), None) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let default_model = get_default_model(provider);
                    let messages_owned = messages.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        fetch_data_with_provider_and_schema(&messages_owned, provider, default_model, schema.as_deref(), model_name.as_deref()).await
                    })
                } else {
                    let default_model = get_default_model(Provider::OpenAI);
                    let messages_owned = messages.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        fetch_data_with_provider_and_schema(&messages_owned, Provider::OpenAI, default_model, schema.as_deref(), model_name.as_deref()).await
                    })
                }
            },
            (None, Some(model)) => {
                let messages_owned = messages.clone();
                let model_owned = model.clone();
                let schema = schema_owned.clone();
                let model_name = model_name_owned.clone();
                run_async(async move {
                    fetch_data_with_provider_and_schema(&messages_owned, Provider::OpenAI, &model_owned, schema.as_deref(), model_name.as_deref()).await
                })
            },
            (None, None) => {
                let default_model = get_default_model(Provider::OpenAI);
                let messages_owned = messages.clone();
                let schema = schema_owned.clone();
                let model_name = model_name_owned.clone();
                run_async(async move {
                    fetch_data_with_provider_and_schema(&messages_owned, Provider::OpenAI, default_model, schema.as_deref(), model_name.as_deref()).await
                })
            },
        }
    } else {
        // Use regular inference without structured output
        match (&kwargs.provider, &kwargs.model) {
            (Some(provider_str), Some(model)) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let messages_owned = messages.clone();
                    let model_owned = model.clone();
                    run_async(async move {
                        fetch_data_with_provider(&messages_owned, provider, &model_owned).await
                    })
                } else {
                    let messages_owned = messages.clone();
                    let model_owned = model.clone();
                    run_async(async move {
                        fetch_data_with_provider(&messages_owned, Provider::OpenAI, &model_owned).await
                    })
                }
            },
            (Some(provider_str), None) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let default_model = get_default_model(provider);
                    let messages_owned = messages.clone();
                    run_async(async move {
                        fetch_data_with_provider(&messages_owned, provider, default_model).await
                    })
                } else {
                    let messages_owned = messages.clone();
                    run_async(async move {
                        fetch_data(&messages_owned).await
                    })
                }
            },
            (None, Some(model)) => {
                let messages_owned = messages.clone();
                let model_owned = model.clone();
                run_async(async move {
                    fetch_data_with_provider(&messages_owned, Provider::OpenAI, &model_owned).await
                })
            },
            (None, None) => {
                let messages_owned = messages.clone();
                run_async(async move {
                    fetch_data(&messages_owned).await
                })
            },
        }
    };

    // Map results back to original positions
    let mut results: Vec<Option<String>> = vec![None; ca.len()];
    for ((idx, _), result) in messages_with_indices.iter().zip(api_results.iter()) {
        results[*idx] = result.clone();
    }

    let out = StringChunked::from_iter_options(ca.name().clone(), results.into_iter());

    Ok(out.into_series())
}

#[derive(Deserialize)]
pub struct MessageKwargs {
    message_type: String,
}

// Register the string_to_message function with Polars
#[polars_expr(output_type=String)]
fn string_to_message(inputs: &[Series], kwargs: MessageKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let message_type = kwargs.message_type;

    let out: StringChunked = ca.apply(|opt_value| {
        opt_value.map(|value| {
            // Properly escape the content as a JSON string
            let escaped_value = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string());
            Cow::Owned(format!(
                "{{\"role\": \"{message_type}\", \"content\": {escaped_value}}}"
            ))
        })
    });
    Ok(out.into_series())
}

// New function to handle JSON arrays of messages
#[polars_expr(output_type=String)]
fn inference_messages(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    let input_series = &inputs[0];

    // Handle empty series or null dtype - return empty String series
    if input_series.is_empty() || input_series.dtype() == &polars::datatypes::DataType::Null {
        return Ok(StringChunked::from_iter_options(
            input_series.name().clone(),
            std::iter::empty::<Option<String>>(),
        ).into_series());
    }

    let ca: &StringChunked = input_series.str()?;

    // Collect message arrays, keeping track of their original indices
    let mut arrays_with_indices: Vec<(usize, Vec<Message>)> = Vec::new();
    for (idx, opt) in ca.into_iter().enumerate() {
        if let Some(s) = opt {
            // Parse the JSON string into a vector of Messages
            let messages = crate::utils::parse_message_json(s).unwrap_or_default();
            arrays_with_indices.push((idx, messages));
        }
    }

    // Extract just the message arrays for the API calls
    let message_arrays: Vec<Vec<Message>> = arrays_with_indices.iter().map(|(_, arr)| arr.clone()).collect();

    // Get results based on provider and model
    let api_results = if kwargs.response_schema.is_some() {
        // Use structured output with validation
        // Clone the strings so we can move them into async blocks
        let schema_owned = kwargs.response_schema.clone();
        let model_name_owned = kwargs.response_model_name.clone();

        match (&kwargs.provider, &kwargs.model) {
            (Some(provider_str), Some(model)) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let arrays_owned = message_arrays.clone();
                    let model_owned = model.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays_with_provider_and_schema(&arrays_owned, provider, &model_owned, schema.as_deref(), model_name.as_deref()).await
                    })
                } else {
                    let arrays_owned = message_arrays.clone();
                    let model_owned = model.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays_with_provider_and_schema(&arrays_owned, Provider::OpenAI, &model_owned, schema.as_deref(), model_name.as_deref()).await
                    })
                }
            },
            (Some(provider_str), None) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let default_model = get_default_model(provider);
                    let arrays_owned = message_arrays.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays_with_provider_and_schema(&arrays_owned, provider, default_model, schema.as_deref(), model_name.as_deref()).await
                    })
                } else {
                    let default_model = get_default_model(Provider::OpenAI);
                    let arrays_owned = message_arrays.clone();
                    let schema = schema_owned.clone();
                    let model_name = model_name_owned.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays_with_provider_and_schema(&arrays_owned, Provider::OpenAI, default_model, schema.as_deref(), model_name.as_deref()).await
                    })
                }
            },
            (None, Some(model)) => {
                let arrays_owned = message_arrays.clone();
                let model_owned = model.clone();
                let schema = schema_owned.clone();
                let model_name = model_name_owned.clone();
                run_async(async move {
                    crate::utils::fetch_data_message_arrays_with_provider_and_schema(&arrays_owned, Provider::OpenAI, &model_owned, schema.as_deref(), model_name.as_deref()).await
                })
            },
            (None, None) => {
                let default_model = get_default_model(Provider::OpenAI);
                let arrays_owned = message_arrays.clone();
                let schema = schema_owned.clone();
                let model_name = model_name_owned.clone();
                run_async(async move {
                    crate::utils::fetch_data_message_arrays_with_provider_and_schema(&arrays_owned, Provider::OpenAI, default_model, schema.as_deref(), model_name.as_deref()).await
                })
            },
        }
    } else {
        // Use regular inference without structured output
        match (&kwargs.provider, &kwargs.model) {
            (Some(provider_str), Some(model)) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let arrays_owned = message_arrays.clone();
                    let model_owned = model.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays_with_provider(&arrays_owned, provider, &model_owned).await
                    })
                } else {
                    let arrays_owned = message_arrays.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays(&arrays_owned).await
                    })
                }
            },
            (Some(provider_str), None) => {
                if let Some(provider) = parse_provider(provider_str) {
                    let default_model = get_default_model(provider);
                    let arrays_owned = message_arrays.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays_with_provider(&arrays_owned, provider, default_model).await
                    })
                } else {
                    let arrays_owned = message_arrays.clone();
                    run_async(async move {
                        crate::utils::fetch_data_message_arrays(&arrays_owned).await
                    })
                }
            },
            (None, Some(model)) => {
                let arrays_owned = message_arrays.clone();
                let model_owned = model.clone();
                run_async(async move {
                    crate::utils::fetch_data_message_arrays_with_provider(&arrays_owned, Provider::OpenAI, &model_owned).await
                })
            },
            (None, None) => {
                let arrays_owned = message_arrays.clone();
                run_async(async move {
                    crate::utils::fetch_data_message_arrays(&arrays_owned).await
                })
            },
        }
    };

    // Map results back to original positions
    let mut results: Vec<Option<String>> = vec![None; ca.len()];
    for ((idx, _), result) in arrays_with_indices.iter().zip(api_results.iter()) {
        results[*idx] = result.clone();
    }

    let out = StringChunked::from_iter_options(ca.name().clone(), results.into_iter());

    Ok(out.into_series())
}

// Function to combine multiple message expressions into a single JSON array
#[polars_expr(output_type=String)]
fn combine_messages(inputs: &[Series]) -> PolarsResult<Series> {
    // Ensure we have at least one input
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "combine_messages requires at least one input".into(),
        ));
    }

    // Get the first input to determine length and name
    let first_ca = inputs[0].str()?;
    let name = first_ca.name().clone();
    let len = first_ca.len();

    // Create a vector to store the results for each row
    let mut result_values = Vec::with_capacity(len);

    // Process each row
    for i in 0..len {
        let mut combined_messages = String::from("[");
        let mut first = true;

        // Process each input column for this row
        for input in inputs {
            let ca = input.str()?;
            if let Some(msg_str) = ca.get(i) {
                // Skip empty messages
                if msg_str.is_empty() {
                    continue;
                }
                
                // Add comma if not the first message
                if !first {
                    combined_messages.push(',');
                }
                
                // Determine if this is a single message or an array of messages
                if msg_str.starts_with("[") && msg_str.ends_with("]") {
                    // This is already an array, so remove the brackets
                    let inner = &msg_str[1..msg_str.len() - 1];
                    if !inner.is_empty() {
                        combined_messages.push_str(inner);
                        first = false;
                    }
                } else if msg_str.starts_with("{") && msg_str.ends_with("}") {
                    // This is a single message, just append it
                    combined_messages.push_str(msg_str);
                    first = false;
                } else {
                    // Try to parse as a message object or array
                    // For simplicity, we'll just wrap it as a user message if it doesn't parse
                    match serde_json::from_str::<serde_json::Value>(msg_str) {
                        Ok(_) => {
                            // It's valid JSON, append it directly
                            combined_messages.push_str(msg_str);
                            first = false;
                        },
                        Err(_) => {
                            // It's not valid JSON, wrap it as a user message
                            combined_messages.push_str(&format!(
                                "{{\"role\": \"user\", \"content\": \"{}\"}}",
                                msg_str.replace("\"", "\\\"")
                            ));
                            first = false;
                        }
                    }
                }
            }
        }
        
        // Close the array
        combined_messages.push(']');
        
        // Add to results
        result_values.push(Some(combined_messages));
    }

    // Create chunked array from the results
    let ca = StringChunked::from_iter_options(name, result_values.into_iter());
    Ok(ca.into_series())
}

// ============================================================================
// Embedding Expressions
// ============================================================================

#[derive(Debug, Deserialize, Default)]
pub struct EmbeddingKwargs {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

/// Get default embedding model for a given provider
fn get_default_embedding_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "text-embedding-3-small",
        Provider::Anthropic => "text-embedding-3-small", // Fallback to OpenAI
        Provider::Gemini => "text-embedding-004",
        Provider::Groq => "text-embedding-3-small", // Fallback to OpenAI
        Provider::Bedrock => "amazon.titan-embed-text-v1",
    }
}

// Register the asynchronous embedding function with Polars
#[polars_expr(output_type_func=embedding_output_type)]
fn embedding_async(inputs: &[Series], kwargs: EmbeddingKwargs) -> PolarsResult<Series> {
    let input_series = &inputs[0];

    // Handle empty series or null dtype
    if input_series.is_empty() || input_series.dtype() == &DataType::Null {
        let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            input_series.name().clone(),
            0,
            0,
            DataType::Float64,
        );
        return Ok(builder.finish().into_series());
    }

    let ca: &StringChunked = input_series.str()?;

    // Collect all texts, keeping track of their original indices
    let mut texts_with_indices: Vec<(usize, String)> = Vec::new();
    for (idx, opt_value) in ca.into_iter().enumerate() {
        if let Some(value) = opt_value {
            texts_with_indices.push((idx, value.to_owned()));
        }
    }

    // Extract just the texts for the API calls
    let texts: Vec<String> = texts_with_indices.iter().map(|(_, text)| text.clone()).collect();

    // Determine provider and model
    let provider = match &kwargs.provider {
        Some(provider_str) => parse_provider(provider_str).unwrap_or(Provider::OpenAI),
        None => Provider::OpenAI,
    };

    let model = kwargs
        .model
        .unwrap_or_else(|| get_default_embedding_model(provider).to_string());

    // Fetch embeddings in parallel using spawn for true parallelization
    // Clone data for 'static lifetime requirement of spawn
    let texts_owned = texts.clone();
    let model_owned = model.clone();
    let api_results = run_async(async move {
        fetch_embeddings_with_provider(&texts_owned, provider, &model_owned).await
    });

    // Map results back to original positions
    let mut results: Vec<Option<Vec<f64>>> = vec![None; ca.len()];
    for ((idx, _), result) in texts_with_indices.iter().zip(api_results.iter()) {
        results[*idx] = result.clone();
    }

    // Convert Vec<Option<Vec<f64>>> to a Series with List<Float64> dtype
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        ca.name().clone(),
        ca.len(),
        1536, // Initial capacity for inner values (typical embedding size)
        DataType::Float64,
    );

    for opt_embedding in results {
        if let Some(embedding) = opt_embedding {
            builder.append_slice(&embedding);
        } else {
            builder.append_null();
        }
    }

    Ok(builder.finish().into_series())
}

/// Output type function for embedding_async
fn embedding_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static(""),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

// ============================================================================
// Vector Similarity Operations
// ============================================================================

/// Calculate cosine similarity between two embedding vectors
#[polars_expr(output_type=Float64)]
fn cosine_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let vec1 = inputs[0].list()?;
    let vec2 = inputs[1].list()?;

    let out: Float64Chunked = vec1
        .amortized_iter()
        .zip(vec2.amortized_iter())
        .map(|(opt_v1, opt_v2)| {
            match (opt_v1, opt_v2) {
                (Some(v1), Some(v2)) => {
                    let arr1 = v1.as_ref().f64()?;
                    let arr2 = v2.as_ref().f64()?;

                    if arr1.len() != arr2.len() {
                        return Err(PolarsError::ComputeError(
                            "Vectors must have the same length for cosine similarity".into()
                        ));
                    }

                    let mut dot_product = 0.0;
                    let mut norm1 = 0.0;
                    let mut norm2 = 0.0;

                    for i in 0..arr1.len() {
                        if let (Some(a), Some(b)) = (arr1.get(i), arr2.get(i)) {
                            dot_product += a * b;
                            norm1 += a * a;
                            norm2 += b * b;
                        }
                    }

                    norm1 = norm1.sqrt();
                    norm2 = norm2.sqrt();

                    if norm1 == 0.0 || norm2 == 0.0 {
                        Ok(Some(0.0))
                    } else {
                        Ok(Some(dot_product / (norm1 * norm2)))
                    }
                },
                _ => Ok(None),
            }
        })
        .collect::<PolarsResult<Float64Chunked>>()?;

    Ok(out.into_series())
}

/// Calculate dot product between two embedding vectors
#[polars_expr(output_type=Float64)]
fn dot_product(inputs: &[Series]) -> PolarsResult<Series> {
    let vec1 = inputs[0].list()?;
    let vec2 = inputs[1].list()?;

    let out: Float64Chunked = vec1
        .amortized_iter()
        .zip(vec2.amortized_iter())
        .map(|(opt_v1, opt_v2)| {
            match (opt_v1, opt_v2) {
                (Some(v1), Some(v2)) => {
                    let arr1 = v1.as_ref().f64()?;
                    let arr2 = v2.as_ref().f64()?;

                    if arr1.len() != arr2.len() {
                        return Err(PolarsError::ComputeError(
                            "Vectors must have the same length for dot product".into()
                        ));
                    }

                    let mut result = 0.0;

                    for i in 0..arr1.len() {
                        if let (Some(a), Some(b)) = (arr1.get(i), arr2.get(i)) {
                            result += a * b;
                        }
                    }

                    Ok(Some(result))
                },
                _ => Ok(None),
            }
        })
        .collect::<PolarsResult<Float64Chunked>>()?;

    Ok(out.into_series())
}

/// Calculate Euclidean distance between two embedding vectors
#[polars_expr(output_type=Float64)]
fn euclidean_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let vec1 = inputs[0].list()?;
    let vec2 = inputs[1].list()?;

    let out: Float64Chunked = vec1
        .amortized_iter()
        .zip(vec2.amortized_iter())
        .map(|(opt_v1, opt_v2)| {
            match (opt_v1, opt_v2) {
                (Some(v1), Some(v2)) => {
                    let arr1 = v1.as_ref().f64()?;
                    let arr2 = v2.as_ref().f64()?;

                    if arr1.len() != arr2.len() {
                        return Err(PolarsError::ComputeError(
                            "Vectors must have the same length for Euclidean distance".into()
                        ));
                    }

                    let mut sum_squared_diff = 0.0;

                    for i in 0..arr1.len() {
                        if let (Some(a), Some(b)) = (arr1.get(i), arr2.get(i)) {
                            let diff = a - b;
                            sum_squared_diff += diff * diff;
                        }
                    }

                    Ok(Some(sum_squared_diff.sqrt()))
                },
                _ => Ok(None),
            }
        })
        .collect::<PolarsResult<Float64Chunked>>()?;

    Ok(out.into_series())
}

// ============================================================================
// Approximate Nearest Neighbor Operations
// ============================================================================

#[derive(Deserialize)]
pub struct KnnKwargs {
    k: usize,
}

/// Find k-nearest neighbors for each embedding using HNSW index
/// Returns a list of indices and distances for each query embedding
#[polars_expr(output_type_func=knn_output_type)]
fn knn_hnsw(inputs: &[Series], kwargs: KnnKwargs) -> PolarsResult<Series> {
    let query_embeddings = inputs[0].list()?;
    let reference_embeddings = inputs[1].list()?;
    let k = kwargs.k;

    // Extract all reference embeddings into a Vec<Vec<f64>>
    let mut ref_vecs: Vec<Vec<f64>> = Vec::new();
    for i in 0..reference_embeddings.len() {
        if let Some(embedding_list) = reference_embeddings.get(i) {
            // embedding_list is itself a list (of embeddings)
            // We need to extract each embedding from this list
            let inner_list = embedding_list.as_ref();

            // Try to downcast to ListArray to iterate through embeddings
            if let Some(list_arr) = inner_list.as_any().downcast_ref::<polars_arrow::array::ListArray<i64>>() {
                // This is a list of lists
                for inner_idx in 0..list_arr.len() {
                    if let Some(inner_series) = list_arr.get(inner_idx) {
                        if let Some(primitive_arr) = inner_series.as_any().downcast_ref::<polars_arrow::array::PrimitiveArray<f64>>() {
                            let vec: Vec<f64> = primitive_arr.values_iter().copied().collect();
                            ref_vecs.push(vec);
                        }
                    }
                }
            } else if let Some(primitive_arr) = inner_list.as_any().downcast_ref::<polars_arrow::array::PrimitiveArray<f64>>() {
                // This is a flat list of floats (single embedding)
                let vec: Vec<f64> = primitive_arr.values_iter().copied().collect();
                ref_vecs.push(vec);
            } else {
                return Err(PolarsError::ComputeError("Expected Float64 or List[Float64] array".into()));
            }
        }
    }

    // Build HNSW index
    let index = crate::ann::build_hnsw_index(ref_vecs);

    // Search for k-nearest neighbors for each query
    let mut builder = polars_core::chunked_array::builder::ListPrimitiveChunkedBuilder::<polars_core::datatypes::Int64Type>::new(
        query_embeddings.name().clone(),
        query_embeddings.len(),
        k * 2, // indices + distances
        polars::datatypes::DataType::Int64,
    );

    for i in 0..query_embeddings.len() {
        if let Some(query_embedding) = query_embeddings.get(i) {
            let inner_series = query_embedding.as_ref();
            let arr = inner_series.as_any().downcast_ref::<polars_arrow::array::PrimitiveArray<f64>>()
                .ok_or_else(|| PolarsError::ComputeError("Expected Float64 array for query".into()))?;
            let query_vec: Vec<f64> = arr.values_iter().copied().collect();

            let results = crate::ann::search_hnsw(&index, &query_vec, k);

            // Store indices only for now
            let indices: Vec<i64> = results.iter()
                .map(|(idx, _dist)| *idx as i64)
                .collect();

            builder.append_slice(&indices);
        } else {
            builder.append_null();
        }
    }

    Ok(builder.finish().into_series())
}

/// Output type function for knn_hnsw
fn knn_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static(""),
        DataType::List(Box::new(DataType::Int64)),
    ))
}

// ============================================================================
// Cost Calculation Operations
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct TokenCountKwargs {
    /// The model to use for token counting (affects tokenizer selection).
    /// Defaults to "gpt-4" which uses cl100k_base tokenizer.
    #[serde(default = "default_token_model")]
    model: String,
}

fn default_token_model() -> String {
    "gpt-4".to_string()
}

/// Count the number of tokens in each text string.
///
/// Uses tiktoken tokenizers:
/// - o200k_base for GPT-4o and o1 models
/// - cl100k_base for all other models (GPT-4, GPT-3.5, Claude, Llama, etc.)
///
/// # Arguments
/// * `model` - The model name to determine which tokenizer to use.
///             Defaults to "gpt-4" (cl100k_base tokenizer).
///
/// # Returns
/// A UInt64 series with token counts for each input row.
#[polars_expr(output_type=UInt64)]
fn count_tokens(inputs: &[Series], kwargs: TokenCountKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let model = &kwargs.model;

    let out: UInt64Chunked = ca
        .into_iter()
        .map(|opt_value| opt_value.map(|value| cost::count_tokens(value, model) as u64))
        .collect();

    Ok(out.into_series())
}

#[derive(Debug, Deserialize)]
pub struct CostKwargs {
    /// The provider name (openai, anthropic, gemini, groq, bedrock)
    #[serde(default = "default_cost_provider")]
    provider: String,
    /// The model name for pricing lookup
    #[serde(default = "default_cost_model")]
    model: String,
}

fn default_cost_provider() -> String {
    "openai".to_string()
}

fn default_cost_model() -> String {
    "gpt-4-turbo".to_string()
}

/// Calculate the input cost in USD for each text string based on token count.
///
/// Uses tiktoken for token counting and embedded pricing data for cost calculation.
/// Prices are based on published rates per 1 million tokens.
///
/// # Arguments
/// * `provider` - The provider name: "openai", "anthropic", "gemini", "groq", or "bedrock".
///                Defaults to "openai".
/// * `model` - The model name for pricing lookup (e.g., "gpt-4o", "claude-3-opus-20240229").
///             Defaults to "gpt-4-turbo".
///
/// # Returns
/// A Float64 series with input costs in USD for each row.
/// Returns null for rows where pricing is not available.
///
/// # Example
/// ```python
/// import polars as pl
///
/// df = pl.DataFrame({"text": ["Hello world", "This is a longer prompt"]})
/// df.with_columns(
///     cost=pl.col("text").llama.calculate_input_cost(provider="openai", model="gpt-4o")
/// )
/// # Sum total cost
/// total_cost = df.select(pl.col("cost").sum())
/// ```
#[polars_expr(output_type=Float64)]
fn calculate_input_cost(inputs: &[Series], kwargs: CostKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let model = &kwargs.model;
    let provider_str = &kwargs.provider;

    // Parse provider
    let provider = parse_provider(provider_str).unwrap_or(Provider::OpenAI);

    // Get pricing for this provider/model combination
    let pricing = cost::get_pricing(provider, model);

    let out: Float64Chunked = ca
        .into_iter()
        .map(|opt_value| {
            opt_value.and_then(|value| {
                pricing.map(|p| {
                    let token_count = cost::count_tokens(value, model);
                    p.calculate_input_cost(token_count)
                })
            })
        })
        .collect();

    Ok(out.into_series())
}

/// Calculate the input cost from a pre-computed token count column.
///
/// This is more efficient when you've already counted tokens and want to
/// calculate costs for multiple models without re-tokenizing.
///
/// # Arguments
/// * `provider` - The provider name: "openai", "anthropic", "gemini", "groq", or "bedrock".
/// * `model` - The model name for pricing lookup.
///
/// # Returns
/// A Float64 series with input costs in USD for each row.
#[polars_expr(output_type=Float64)]
fn calculate_cost_from_tokens(inputs: &[Series], kwargs: CostKwargs) -> PolarsResult<Series> {
    let ca: &UInt64Chunked = inputs[0].u64()?;
    let model = &kwargs.model;
    let provider_str = &kwargs.provider;

    // Parse provider
    let provider = parse_provider(provider_str).unwrap_or(Provider::OpenAI);

    // Get pricing for this provider/model combination
    let pricing = cost::get_pricing(provider, model);

    let out: Float64Chunked = ca
        .into_iter()
        .map(|opt_value| {
            opt_value.and_then(|token_count| {
                pricing.map(|p| p.calculate_input_cost(token_count as usize))
            })
        })
        .collect();

    Ok(out.into_series())
}
