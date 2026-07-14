#![allow(clippy::unused_unit)]
use crate::cache::{self, CacheStrategy, CacheConfig};
use crate::utils::*;
use crate::model_client::{Provider, Message};
use crate::cost;
use polars::prelude::*;
use polars_core::chunked_array::builder::ListPrimitiveChunkedBuilder;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::borrow::Cow;
use std::str::FromStr;
use polars::datatypes::DataType;

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

    // === Caching options ===
    /// Enable cache optimization (default: false)
    #[serde(default)]
    cache: Option<bool>,
    /// Cache strategy: "auto", "system_prompt", "schema", "full_prefix", "none"
    #[serde(default)]
    cache_strategy: Option<String>,
    /// Cache TTL for Anthropic: "5m" (default) or "1h"
    #[serde(default)]
    cache_ttl: Option<String>,
    /// Optional cache key hint for OpenAI routing
    #[serde(default)]
    cache_key: Option<String>,
    /// Minimum tokens to trigger caching (default: 1024)
    #[serde(default)]
    cache_min_tokens: Option<usize>,
    /// System prompt to prepend to all messages (enables caching for inference_async)
    #[serde(default)]
    system_prompt: Option<String>,

    /// Enable per-row usage/cost accounting (issue #76). When true, the
    /// output string for each row is a JSON envelope
    /// `{"response": ..., "usage": {...}}` instead of the bare response,
    /// decoded into a Struct on the Python side. Default: false (byte-
    /// identical to the pre-#76 output).
    #[serde(default)]
    usage: Option<bool>,
}

fn parse_provider(provider_str: &str) -> Option<Provider> {
    Provider::from_str(provider_str).ok()
}

/// Get default model for a given provider
fn get_default_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => crate::model_client::openai::DEFAULT_OPENAI_MODEL,
        Provider::Anthropic => crate::model_client::anthropic::DEFAULT_ANTHROPIC_MODEL,
        Provider::Gemini => crate::model_client::gemini::DEFAULT_GEMINI_MODEL,
        Provider::Groq => crate::model_client::groq::DEFAULT_GROQ_MODEL,
        Provider::Bedrock => crate::model_client::bedrock::DEFAULT_BEDROCK_MODEL,
    }
}

/// Resolve the provider/model pair from kwargs, falling back to OpenAI and
/// the provider's default model.
fn resolve_provider_and_model(kwargs: &InferenceKwargs) -> (Provider, String) {
    let provider = kwargs
        .provider
        .as_deref()
        .and_then(parse_provider)
        .unwrap_or(Provider::OpenAI);
    let model = kwargs
        .model
        .clone()
        .unwrap_or_else(|| get_default_model(provider).to_string());
    (provider, model)
}

// This polars_expr annotation registers the function with Polars at build time
#[polars_expr(output_type=String)]
fn inference(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;

    let (provider, model) = resolve_provider_and_model(&kwargs);

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
    if input_series.is_empty() || input_series.dtype() == &DataType::Null {
        return Ok(StringChunked::from_iter_options(
            input_series.name().clone(),
            std::iter::empty::<Option<String>>(),
        ).into_series());
    }

    let ca: &StringChunked = input_series.str()?;

    // Determine provider and model
    let provider = kwargs.provider
        .as_ref()
        .and_then(|s| parse_provider(s))
        .unwrap_or(Provider::OpenAI);

    let model = kwargs.model
        .clone()
        .unwrap_or_else(|| get_default_model(provider).to_string());

    // Check if caching is enabled with a system prompt
    let cache_enabled = kwargs.cache.unwrap_or(false);
    // Issue #76: per-row usage/cost accounting. `Option<bool>` is Copy, so
    // reading it here doesn't move anything else out of `kwargs`.
    let with_usage = kwargs.usage.unwrap_or(false);

    if let Some(system_prompt) = kwargs.system_prompt.as_ref().filter(|_| cache_enabled) {
        // Use caching path: convert text prompts to message arrays with shared system prompt

        // Build message arrays with shared system prompt
        let mut arrays_with_indices: Vec<(usize, Vec<Message>)> = Vec::new();
        for (idx, opt_value) in ca.into_iter().enumerate() {
            if let Some(user_content) = opt_value {
                let messages = vec![
                    Message {
                        role: "system".to_string(),
                        content: system_prompt.clone(),
                        cache_control: None,
                    },
                    Message {
                        role: "user".to_string(),
                        content: user_content.to_string(),
                        cache_control: None,
                    },
                ];
                arrays_with_indices.push((idx, messages));
            }
        }

        let message_arrays: Vec<Vec<Message>> = arrays_with_indices.iter().map(|(_, arr)| arr.clone()).collect();

        // Build cache config
        let cache_config = CacheConfig {
            strategy: kwargs.cache_strategy
                .as_ref()
                .map(|s| CacheStrategy::from_str(s))
                .unwrap_or_default(),
            ttl: kwargs.cache_ttl.clone().unwrap_or_else(|| "5m".to_string()),
            cache_key: kwargs.cache_key.clone(),
            min_tokens: kwargs.cache_min_tokens.unwrap_or(1024),
            report_metrics: true,
        };

        // Analyze batch for cache groups
        let cache_groups = cache::analyze_batch_for_caching(
            &message_arrays,
            cache_config.strategy,
            cache_config.min_tokens,
        );

        // Process with cache optimization
        let schema_owned = kwargs.response_schema.clone();
        let model_name_owned = kwargs.response_model_name.clone();

        let api_results = process_with_cache_groups(
            message_arrays,
            cache_groups,
            provider,
            &model,
            &cache_config,
            schema_owned.as_deref(),
            model_name_owned.as_deref(),
            with_usage,
        );

        // Map results back to original positions
        let mut results: Vec<Option<String>> = vec![None; input_series.len()];
        for ((idx, _), result) in arrays_with_indices.iter().zip(api_results.iter()) {
            results[*idx] = result.clone();
        }

        let out = StringChunked::from_iter_options(input_series.name().clone(), results.into_iter());
        return Ok(out.into_series());
    }

    // Non-caching path: original implementation
    // Collect all messages, keeping track of their original indices
    let mut indices: Vec<usize> = Vec::new();
    let mut messages: Vec<String> = Vec::new();
    for (idx, opt_value) in ca.into_iter().enumerate() {
        if let Some(value) = opt_value {
            indices.push(idx);
            messages.push(value.to_owned());
        }
    }

    let (provider, model) = resolve_provider_and_model(&kwargs);
    let schema = kwargs.response_schema;
    let model_name = kwargs.response_model_name;

    let api_results = run_async(async move {
        crate::utils::fetch_data_with_provider_opts(
            &messages,
            provider,
            &model,
            schema.as_deref(),
            model_name.as_deref(),
            with_usage,
        ).await
    });

    // Map results back to original positions
    let mut results: Vec<Option<String>> = vec![None; ca.len()];
    for (idx, result) in indices.into_iter().zip(api_results) {
        results[idx] = result;
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
    if input_series.is_empty() || input_series.dtype() == &DataType::Null {
        return Ok(StringChunked::from_iter_options(
            input_series.name().clone(),
            std::iter::empty::<Option<String>>(),
        ).into_series());
    }

    // Collect message arrays, keeping track of their original indices
    let mut arrays_with_indices: Vec<(usize, Vec<Message>)> = Vec::new();

    // Handle both String (JSON) and List(Struct) inputs
    match input_series.dtype() {
        polars::datatypes::DataType::String => {
            // Input is JSON strings - parse them
            let ca: &StringChunked = input_series.str()?;
            for (idx, opt) in ca.into_iter().enumerate() {
                if let Some(s) = opt {
                    let messages = crate::utils::parse_message_json(s).unwrap_or_default();
                    arrays_with_indices.push((idx, messages));
                }
            }
        }
        polars::datatypes::DataType::List(_) => {
            // List(Struct{role, content}) input: parse each row's list of message
            // structs directly in Rust. This keeps the default path lazy/streaming
            // instead of routing through a Python map_batches UDF.
            let list_ca = input_series.list()?;
            for (idx, opt_inner) in list_ca.amortized_iter().enumerate() {
                if let Some(inner) = opt_inner {
                    let inner = inner.deep_clone();
                    let mut messages: Vec<Message> = Vec::new();
                    if let Ok(struct_ca) = inner.struct_() {
                        let role_field = struct_ca.field_by_name("role").ok();
                        let content_field = struct_ca.field_by_name("content").ok();
                        let roles = role_field.as_ref().and_then(|s| s.str().ok());
                        let contents = content_field.as_ref().and_then(|s| s.str().ok());
                        for i in 0..inner.len() {
                            let role = roles.and_then(|c| c.get(i)).unwrap_or("user").to_string();
                            let content = contents.and_then(|c| c.get(i)).unwrap_or("").to_string();
                            messages.push(Message { role, content, cache_control: None });
                        }
                    }
                    arrays_with_indices.push((idx, messages));
                }
            }
        }
        other => {
            return Err(polars::prelude::PolarsError::InvalidOperation(
                format!("inference_messages expects String or List(Struct) input, got {other:?}").into()
            ));
        }
    }

    let (provider, model) = resolve_provider_and_model(&kwargs);
    let schema = kwargs.response_schema.clone();
    let model_name = kwargs.response_model_name.clone();

    let message_arrays: Vec<Vec<Message>> =
        arrays_with_indices.iter().map(|(_, arr)| arr.clone()).collect();

    // Check if caching is enabled
    let cache_enabled = kwargs.cache.unwrap_or(false);
    // Issue #76: per-row usage/cost accounting.
    let with_usage = kwargs.usage.unwrap_or(false);

    // Get results based on caching, provider and model
    let api_results = if cache_enabled {
        // Build cache config from kwargs
        let cache_config = CacheConfig {
            strategy: kwargs.cache_strategy
                .as_ref()
                .map(|s| CacheStrategy::from_str(s))
                .unwrap_or_default(),
            ttl: kwargs.cache_ttl.clone().unwrap_or_else(|| "5m".to_string()),
            cache_key: kwargs.cache_key.clone(),
            min_tokens: kwargs.cache_min_tokens.unwrap_or(1024),
            report_metrics: true,
        };

        // Analyze batch to find cache groups
        let cache_groups = cache::analyze_batch_for_caching(
            &message_arrays,
            cache_config.strategy,
            cache_config.min_tokens,
        );

        process_with_cache_groups(
            message_arrays,
            cache_groups,
            provider,
            &model,
            &cache_config,
            schema.as_deref(),
            model_name.as_deref(),
            with_usage,
        )
    } else {
        // Non-cached path (structured or plain): unified through the
        // `_opts` fetcher, which reproduces the pre-#76 split exactly for
        // `with_usage=false` (`errors_as_json = schema.is_some()`).
        let schema_owned = schema.clone();
        let model_name_owned = model_name.clone();
        let arrays_owned = message_arrays.clone();
        let model_owned = model.clone();

        run_async(async move {
            crate::utils::fetch_data_message_arrays_with_provider_opts(
                &arrays_owned, provider, &model_owned,
                schema_owned.as_deref(), model_name_owned.as_deref(), with_usage,
            ).await
        })
    };

    // Map results back to original positions
    let mut results: Vec<Option<String>> = vec![None; input_series.len()];
    for ((idx, _), result) in arrays_with_indices.iter().zip(api_results.iter()) {
        results[*idx] = result.clone();
    }

    let out = StringChunked::from_iter_options(input_series.name().clone(), results.into_iter());

    Ok(out.into_series())
}

/// Process message arrays with cache group optimization
#[allow(clippy::too_many_arguments)]
fn process_with_cache_groups(
    message_arrays: Vec<Vec<Message>>,
    cache_groups: Vec<cache::CacheGroup>,
    provider: Provider,
    model: &str,
    config: &CacheConfig,
    response_schema: Option<&str>,
    response_model_name: Option<&str>,
    with_usage: bool,
) -> Vec<Option<String>> {
    let total_rows = message_arrays.len();
    let mut all_results: Vec<(usize, Option<String>)> = Vec::with_capacity(total_rows);

    // Process each cache group
    // Groups are processed sequentially to ensure cache warming
    // Within each group, first request warms cache, rest are parallel
    for group in cache_groups {
        // Prepare messages with cache control markers
        let prepared_messages: Vec<Vec<Message>> = group.row_indices
            .iter()
            .map(|&row_idx| {
                cache::prepare_messages_for_caching(
                    message_arrays[row_idx].clone(),
                    provider,
                    config,
                    group.cache_breakpoint_idx,
                )
            })
            .collect();

        // Execute with cache warming pattern
        let group_results = if prepared_messages.len() > 1 {
            let msgs = prepared_messages.clone();
            let m = model.to_string();
            let schema = response_schema.map(|s| s.to_string());
            let model_name = response_model_name.map(|s| s.to_string());

            run_async(async move {
                crate::utils::fetch_with_cache_warming(
                    &msgs, provider, &m,
                    schema.as_deref(), model_name.as_deref(), with_usage,
                ).await
            })
        } else {
            // Single row in group - no cache warming benefit
            let msgs = prepared_messages.clone();
            let m = model.to_string();
            let schema = response_schema.map(|s| s.to_string());
            let model_name = response_model_name.map(|s| s.to_string());

            run_async(async move {
                crate::utils::fetch_data_message_arrays_with_provider_opts(
                    &msgs, provider, &m,
                    schema.as_deref(), model_name.as_deref(), with_usage,
                ).await
            })
        };

        // Map results back to original indices
        for (within_group_idx, result) in group_results.into_iter().enumerate() {
            let original_idx = group.row_indices[within_group_idx];
            all_results.push((original_idx, result));
        }
    }

    // Sort by original index and extract results
    all_results.sort_by_key(|(idx, _)| *idx);
    all_results.into_iter().map(|(_, r)| r).collect()
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

    // Cast all inputs to string columns once, instead of once per row
    let columns: Vec<&StringChunked> = inputs
        .iter()
        .map(|s| s.str())
        .collect::<PolarsResult<Vec<_>>>()?;

    let name = columns[0].name().clone();
    let len = columns[0].len();

    // Create a vector to store the results for each row
    let mut result_values = Vec::with_capacity(len);

    // Process each row
    for i in 0..len {
        let mut combined_messages = String::from("[");
        let mut first = true;

        // Process each input column for this row
        for ca in &columns {
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
                if msg_str.starts_with('[') && msg_str.ends_with(']') {
                    // This is already an array, so remove the brackets
                    let inner = &msg_str[1..msg_str.len() - 1];
                    if !inner.is_empty() {
                        combined_messages.push_str(inner);
                        first = false;
                    }
                } else if msg_str.starts_with('{') && msg_str.ends_with('}') {
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
                            let escaped = serde_json::to_string(msg_str)
                                .unwrap_or_else(|_| "\"\"".to_string());
                            combined_messages.push_str(&format!(
                                "{{\"role\": \"user\", \"content\": {escaped}}}"
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
    let mut indices: Vec<usize> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (idx, opt_value) in ca.into_iter().enumerate() {
        if let Some(value) = opt_value {
            indices.push(idx);
            texts.push(value.to_owned());
        }
    }

    // Determine provider and model
    let provider = match &kwargs.provider {
        Some(provider_str) => parse_provider(provider_str).unwrap_or(Provider::OpenAI),
        None => Provider::OpenAI,
    };

    let model = kwargs
        .model
        .unwrap_or_else(|| get_default_embedding_model(provider).to_string());

    // Fetch embeddings in parallel using spawn for true parallelization
    let api_results = run_async(async move {
        fetch_embeddings_with_provider(&texts, provider, &model).await
    });

    // Map results back to original positions
    let mut results: Vec<Option<Vec<f64>>> = vec![None; ca.len()];
    for (idx, result) in indices.into_iter().zip(api_results) {
        results[idx] = result;
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

/// Apply a binary metric over two List[Float64] columns element-wise.
///
/// Uses the contiguous-slice fast path when both vectors have no nulls
/// (the common case for embeddings), falling back to a null-skipping
/// element-wise iteration otherwise.
fn binary_embedding_metric<F>(
    inputs: &[Series],
    metric_name: &str,
    compute: F,
) -> PolarsResult<Series>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
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
                            format!("Vectors must have the same length for {metric_name}").into()
                        ));
                    }

                    match (arr1.cont_slice(), arr2.cont_slice()) {
                        (Ok(s1), Ok(s2)) => Ok(Some(compute(s1, s2))),
                        _ => {
                            // Null-aware fallback: skip positions where either side is null
                            let mut s1 = Vec::with_capacity(arr1.len());
                            let mut s2 = Vec::with_capacity(arr2.len());
                            for (a, b) in arr1.iter().zip(arr2.iter()) {
                                if let (Some(a), Some(b)) = (a, b) {
                                    s1.push(a);
                                    s2.push(b);
                                }
                            }
                            Ok(Some(compute(&s1, &s2)))
                        }
                    }
                },
                _ => Ok(None),
            }
        })
        .collect::<PolarsResult<Float64Chunked>>()?;

    Ok(out.into_series())
}

/// Calculate cosine similarity between two embedding vectors
#[polars_expr(output_type=Float64)]
fn cosine_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    binary_embedding_metric(inputs, "cosine similarity", |s1, s2| {
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            dot_product += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }
        norm1 = norm1.sqrt();
        norm2 = norm2.sqrt();
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    })
}

/// Calculate dot product between two embedding vectors
#[polars_expr(output_type=Float64)]
fn dot_product(inputs: &[Series]) -> PolarsResult<Series> {
    binary_embedding_metric(inputs, "dot product", |s1, s2| {
        s1.iter().zip(s2.iter()).map(|(a, b)| a * b).sum()
    })
}

/// Calculate Euclidean distance between two embedding vectors
#[polars_expr(output_type=Float64)]
fn euclidean_distance(inputs: &[Series]) -> PolarsResult<Series> {
    binary_embedding_metric(inputs, "Euclidean distance", |s1, s2| {
        s1.iter()
            .zip(s2.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    })
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
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        query_embeddings.name().clone(),
        query_embeddings.len(),
        k,
        DataType::Int64,
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
// Codebook Induction: k-means clustering (issue #78)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ClusterEmbeddingsKwargs {
    /// Fixed number of clusters. `None` (the default) auto-selects `k` in
    /// `[k_min, k_max]` via sampled silhouette (see `crate::kmeans::auto_k`).
    #[serde(default)]
    k: Option<usize>,
    #[serde(default = "default_cluster_k_min")]
    k_min: usize,
    #[serde(default = "default_cluster_k_max")]
    k_max: usize,
    #[serde(default = "default_cluster_max_iter")]
    max_iter: usize,
    #[serde(default = "default_cluster_n_init")]
    n_init: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default = "default_silhouette_sample")]
    silhouette_sample: usize,
}

fn default_cluster_k_min() -> usize {
    2
}

fn default_cluster_k_max() -> usize {
    20
}

fn default_cluster_max_iter() -> usize {
    100
}

fn default_cluster_n_init() -> usize {
    8
}

fn default_silhouette_sample() -> usize {
    200
}

/// Cluster a whole column of embeddings with hand-rolled k-means++ / Lloyd's
/// algorithm (`crate::kmeans`) -- a *whole-column* plugin expression (the
/// entire embeddings column is clustered together in one call, not row by
/// row), following the same pattern as `knn_hnsw` above: it is registered
/// `is_elementwise=True` on the Python side purely so it can be used inside
/// `with_columns`, even though the underlying computation is over the full
/// column at once.
///
/// Null / empty-vector rows are excluded from clustering and come back as a
/// null output row. Every non-null row gets a JSON-encoded
/// `{"cluster": u32, "distance": f64, "k": u32}` string, decoded into a
/// `Struct` on the Python side (`polar_llama.codebook.cluster_embeddings`) --
/// the same "emit JSON, decode with an explicit dtype" convention used by
/// `inference_async`/`inference_messages` structured output.
#[polars_expr(output_type=String)]
fn cluster_embeddings(inputs: &[Series], kwargs: ClusterEmbeddingsKwargs) -> PolarsResult<Series> {
    let embeddings = inputs[0].list()?;

    if embeddings.is_empty() || embeddings.dtype() == &DataType::Null {
        return Ok(StringChunked::from_iter_options(
            embeddings.name().clone(),
            std::iter::empty::<Option<String>>(),
        )
        .into_series());
    }

    // Extract non-null, non-empty embedding vectors, remembering their
    // original row position so results can be scattered back in order.
    let mut indices: Vec<usize> = Vec::new();
    let mut points: Vec<Vec<f64>> = Vec::new();
    for i in 0..embeddings.len() {
        if let Some(row) = embeddings.get(i) {
            let inner = row.as_ref();
            // Accept both Float64 (what embedding_async emits) and Float32
            // (common for user-supplied precomputed embeddings, e.g.
            // sentence-transformers / numpy default), casting f32 -> f64.
            // Anything else leaves the row as a null cluster.
            let v: Option<Vec<f64>> = inner
                .as_any()
                .downcast_ref::<polars_arrow::array::PrimitiveArray<f64>>()
                .map(|arr| arr.values_iter().copied().collect())
                .or_else(|| {
                    inner
                        .as_any()
                        .downcast_ref::<polars_arrow::array::PrimitiveArray<f32>>()
                        .map(|arr| arr.values_iter().map(|&x| x as f64).collect())
                });
            if let Some(v) = v {
                if !v.is_empty() {
                    indices.push(i);
                    points.push(v);
                }
            }
        }
    }

    let mut rows: Vec<Option<String>> = vec![None; embeddings.len()];

    let n = points.len();
    if n == 1 {
        let json = serde_json::json!({"cluster": 0u32, "distance": 0.0f64, "k": 1u32});
        rows[indices[0]] = Some(json.to_string());
    } else if n > 1 {
        let max_iter = kwargs.max_iter.max(1);
        let n_init = kwargs.n_init.max(1);
        let seed = kwargs.seed;
        let silhouette_sample = kwargs.silhouette_sample.max(1);

        let (chosen_k, result) = match kwargs.k {
            Some(k) => {
                let k = k.clamp(1, n);
                (k, crate::kmeans::kmeans(&points, k, max_iter, n_init, seed))
            }
            None => crate::kmeans::auto_k(
                &points,
                kwargs.k_min,
                kwargs.k_max,
                max_iter,
                n_init,
                seed,
                silhouette_sample,
            ),
        };

        for (pos, &orig_idx) in indices.iter().enumerate() {
            let json = serde_json::json!({
                "cluster": result.labels[pos] as u32,
                "distance": result.distances[pos],
                "k": chosen_k as u32,
            });
            rows[orig_idx] = Some(json.to_string());
        }
    }

    Ok(StringChunked::from_iter_options(embeddings.name().clone(), rows.into_iter()).into_series())
}

// ============================================================================
// Tool Call Execution (MCP)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ExecuteToolCallsKwargs {
    /// MCP streamable-HTTP endpoint, e.g. "http://localhost:8811/mcp"
    transport: String,
    /// Max in-flight tool calls across the whole batch
    #[serde(default = "default_tool_concurrency")]
    concurrency: usize,
    /// Per-call timeout in seconds
    #[serde(default = "default_tool_timeout")]
    timeout_s: u64,
    /// Optional JSON object mapping tool name -> input JSON schema.
    /// When present, arguments are validated before any network call.
    #[serde(default)]
    tool_schemas: Option<String>,
}

fn default_tool_concurrency() -> usize {
    32
}

fn default_tool_timeout() -> u64 {
    30
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallSpec {
    tool_name: String,
    /// Arguments either as a JSON-encoded string (strict-mode emission form)
    /// or as an inline JSON object.
    #[serde(default)]
    arguments: serde_json::Value,
}

/// Parse one row's JSON array of tool calls.
fn parse_tool_calls_row(row: &str) -> Result<Vec<ToolCallSpec>, String> {
    let value: serde_json::Value =
        serde_json::from_str(row).map_err(|e| format!("invalid tool call JSON: {e}"))?;
    match value {
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|item| {
                serde_json::from_value::<ToolCallSpec>(item)
                    .map_err(|e| format!("invalid tool call object: {e}"))
            })
            .collect(),
        serde_json::Value::Object(_) => {
            // Accept a bare single call for convenience
            serde_json::from_value::<ToolCallSpec>(value)
                .map(|c| vec![c])
                .map_err(|e| format!("invalid tool call object: {e}"))
        }
        serde_json::Value::Null => Ok(vec![]),
        _ => Err("expected a JSON array of tool calls".to_string()),
    }
}

/// Normalize arguments: emission carries them as a JSON-encoded string
/// (the strict-mode-safe form); accept inline objects too.
fn normalize_arguments(raw: &serde_json::Value) -> Result<serde_json::Value, String> {
    match raw {
        serde_json::Value::String(s) => {
            if s.trim().is_empty() {
                return Ok(serde_json::json!({}));
            }
            serde_json::from_str::<serde_json::Value>(s)
                .map_err(|e| format!("arguments is not valid JSON: {e}"))
        }
        serde_json::Value::Null => Ok(serde_json::json!({})),
        other => Ok(other.clone()),
    }
}

fn tool_result_json(
    spec: &ToolCallSpec,
    content: Option<String>,
    is_error: bool,
    error: Option<String>,
) -> serde_json::Value {
    let arguments_str = match &spec.arguments {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    serde_json::json!({
        "tool_name": spec.tool_name,
        "arguments": arguments_str,
        "content": content,
        "is_error": is_error,
        "_error": error,
    })
}

/// Execute a column of emitted tool calls against an MCP server, in parallel
/// across every call of every row. Input: Utf8 rows containing JSON arrays of
/// {tool_name, arguments}. Output: Utf8 rows containing JSON arrays of
/// {tool_name, arguments, content, is_error, _error}. Per-call failures are
/// data, not exceptions; only an unreachable/misconfigured transport raises.
#[polars_expr(output_type=String)]
fn execute_tool_calls(inputs: &[Series], kwargs: ExecuteToolCallsKwargs) -> PolarsResult<Series> {
    let input_series = &inputs[0];

    if input_series.is_empty() || input_series.dtype() == &DataType::Null {
        return Ok(StringChunked::from_iter_options(
            input_series.name().clone(),
            std::iter::empty::<Option<String>>(),
        )
        .into_series());
    }

    let ca: &StringChunked = input_series.str()?;

    // Compile per-tool argument validators up front (config errors fail fast).
    let validators: std::collections::HashMap<String, jsonschema::Validator> =
        match &kwargs.tool_schemas {
            Some(schemas_json) => {
                let schemas: serde_json::Value = serde_json::from_str(schemas_json)
                    .map_err(|e| {
                        PolarsError::ComputeError(format!("invalid tool_schemas JSON: {e}").into())
                    })?;
                let obj = schemas.as_object().ok_or_else(|| {
                    PolarsError::ComputeError("tool_schemas must be a JSON object".into())
                })?;
                let mut map = std::collections::HashMap::new();
                for (name, schema) in obj {
                    let validator = jsonschema::validator_for(schema).map_err(|e| {
                        PolarsError::ComputeError(
                            format!("invalid schema for tool '{name}': {e}").into(),
                        )
                    })?;
                    map.insert(name.clone(), validator);
                }
                map
            }
            None => std::collections::HashMap::new(),
        };

    // Parse every row, flattening calls to (row, position) so all calls in
    // the batch share one concurrency pool.
    let mut rows: Vec<Option<Vec<ToolCallSpec>>> = Vec::with_capacity(ca.len());
    let mut row_errors: Vec<Option<String>> = vec![None; ca.len()];
    for (idx, opt_value) in ca.into_iter().enumerate() {
        match opt_value {
            Some(value) => match parse_tool_calls_row(value) {
                Ok(calls) => rows.push(Some(calls)),
                Err(e) => {
                    row_errors[idx] = Some(e);
                    rows.push(Some(vec![]));
                }
            },
            None => rows.push(None),
        }
    }

    // Pre-resolve each call: validated + normalized arguments, or an
    // immediate validation error that skips the network entirely.
    enum Prepared {
        Ready(serde_json::Value),
        Invalid(String),
    }
    let mut flat: Vec<(usize, usize, ToolCallSpec, Prepared)> = Vec::new();
    for (row_idx, row) in rows.iter().enumerate() {
        if let Some(calls) = row {
            for (call_idx, spec) in calls.iter().enumerate() {
                let prepared = match normalize_arguments(&spec.arguments) {
                    Ok(args) => match validators.get(&spec.tool_name) {
                        Some(validator) if !validator.is_valid(&args) => {
                            let errors: Vec<String> = validator
                                .iter_errors(&args)
                                .map(|e| format!("{} at {}", e, e.instance_path()))
                                .collect();
                            Prepared::Invalid(format!(
                                "argument validation failed: {}",
                                errors.join("; ")
                            ))
                        }
                        _ => Prepared::Ready(args),
                    },
                    Err(e) => Prepared::Invalid(e),
                };
                flat.push((row_idx, call_idx, spec.clone(), prepared));
            }
        }
    }

    let transport = kwargs.transport.clone();
    let timeout = std::time::Duration::from_secs(kwargs.timeout_s.max(1));
    let concurrency = kwargs.concurrency.max(1);

    // One handshake per batch; an unreachable transport is a config error.
    let client = run_async(async move { crate::mcp::McpHttpClient::connect(&transport, timeout).await })
        .map_err(|e| {
            PolarsError::ComputeError(
                format!("failed to connect to MCP server '{}': {e}", kwargs.transport).into(),
            )
        })?;

    // Fan out every ready call through a shared semaphore.
    let outcomes: Vec<(usize, usize, serde_json::Value)> = {
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut handles = Vec::with_capacity(flat.len());

        for (row_idx, call_idx, spec, prepared) in flat {
            match prepared {
                Prepared::Invalid(msg) => {
                    handles.push(TaskOrReady::Ready((
                        row_idx,
                        call_idx,
                        tool_result_json(&spec, None, true, Some(msg)),
                    )));
                }
                Prepared::Ready(args) => {
                    let client = client.clone();
                    let semaphore = semaphore.clone();
                    handles.push(TaskOrReady::Task(RT.spawn(async move {
                        let _permit = semaphore.acquire().await.expect("semaphore closed");
                        let outcome = client.call_tool(&spec.tool_name, &args).await;
                        (
                            row_idx,
                            call_idx,
                            tool_result_json(&spec, outcome.content, outcome.is_error, outcome.error),
                        )
                    })));
                }
            }
        }

        run_async(async move {
            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                match handle {
                    TaskOrReady::Ready(r) => results.push(r),
                    TaskOrReady::Task(t) => results.push(t.await.expect("tool call task panicked")),
                }
            }
            results
        })
    };

    // Reassemble per-row JSON arrays, preserving emission order within rows.
    let mut per_row: Vec<Option<Vec<(usize, serde_json::Value)>>> = rows
        .iter()
        .map(|r| r.as_ref().map(|calls| Vec::with_capacity(calls.len())))
        .collect();
    for (row_idx, call_idx, result) in outcomes {
        if let Some(Some(row)) = per_row.get_mut(row_idx) {
            row.push((call_idx, result));
        }
    }

    let results: Vec<Option<String>> = per_row
        .into_iter()
        .enumerate()
        .map(|(idx, opt_row)| {
            opt_row.map(|mut row| {
                row.sort_by_key(|(call_idx, _)| *call_idx);
                let mut values: Vec<serde_json::Value> =
                    row.into_iter().map(|(_, v)| v).collect();
                if let Some(parse_err) = &row_errors[idx] {
                    values.push(serde_json::json!({
                        "tool_name": null,
                        "arguments": null,
                        "content": null,
                        "is_error": true,
                        "_error": parse_err,
                    }));
                }
                serde_json::Value::Array(values).to_string()
            })
        })
        .collect();

    let out = StringChunked::from_iter_options(ca.name().clone(), results.into_iter());
    Ok(out.into_series())
}

enum TaskOrReady {
    Task(tokio::task::JoinHandle<(usize, usize, serde_json::Value)>),
    Ready((usize, usize, serde_json::Value)),
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
/// - o200k_base for GPT-4o, GPT-4.1, GPT-5, and o-series models
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
    // Resolve the tokenizer once per batch instead of once per row
    let tokenizer_type = cost::get_tokenizer_type(&kwargs.model);

    let out: UInt64Chunked = ca
        .into_iter()
        .map(|opt_value| {
            opt_value.map(|value| cost::count_tokens_with_tokenizer(value, tokenizer_type) as u64)
        })
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
    "gpt-4o".to_string()
}

/// Calculate the input cost in USD for each text string based on token count.
///
/// Uses tiktoken for token counting and embedded pricing data for cost calculation.
/// Prices are based on published rates per 1 million tokens.
///
/// # Arguments
/// * `provider` - The provider name: "openai", "anthropic", "gemini", "groq", or "bedrock".
///                Defaults to "openai".
/// * `model` - The model name for pricing lookup (e.g., "gpt-4o", "claude-opus-4-8").
///             Defaults to "gpt-4o".
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

    // Resolve pricing and tokenizer once per batch
    let pricing = cost::get_pricing(provider, model);
    let tokenizer_type = cost::get_tokenizer_type(model);

    let out: Float64Chunked = ca
        .into_iter()
        .map(|opt_value| {
            opt_value.and_then(|value| {
                pricing.map(|p| {
                    let token_count = cost::count_tokens_with_tokenizer(value, tokenizer_type);
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

// ============================================================================
// Inter-rater reliability (issue #79)
// ============================================================================
//
// Cohen's kappa (`cohens_kappa` / `cohens_kappa_ci`) and Krippendorff's alpha
// (`krippendorffs_alpha` / `krippendorffs_alpha_ci`) are *aggregations*, not
// row-wise transforms: they reduce a whole column (or, inside
// `group_by().agg()`, a whole group) down to a single number. They are
// registered from the Python side with `returns_scalar=True`
// (`polar_llama/utils.py::register_plugin`), which is why every function
// below always returns a length-1 `Series` regardless of the input length.
//
// The pure numeric algorithms (confusion-matrix kappa, coincidence-matrix
// alpha, case-resampling bootstrap) live in `crate::metrics` and are unit
// tested there against published reference fixtures; this section only does
// the Polars/`Series` <-> plain-Rust-vector plumbing: pairwise-complete
// filtering, categorical label encoding, and building the
// `Struct{value, ci_low, ci_high}` output for the `_ci` variants.
//
// `output_type_func` cannot see kwargs, so a bootstrap request needs a
// different return dtype (`Float64` vs `Struct`) from a plain point
// estimate -- hence two Rust symbols per metric, dispatched on the Python
// side by whether `n_bootstrap` is `None` (see `polar_llama/reliability.py`).

fn default_irr_ci() -> f64 {
    0.95
}

/// A single rater's category value, used only to build the sorted label
/// list Cohen's kappa weights are indexed over. Numeric dtypes sort
/// numerically; string-like dtypes (`String`/`Categorical`/`Enum`, the
/// latter two cast to `String` first) sort lexicographically -- matching
/// `np.unique` on the concatenation of both columns, same as sklearn.
#[derive(Clone)]
enum LabelValue {
    Num(f64),
    Str(String),
}

impl LabelValue {
    fn cmp_key(&self, other: &LabelValue) -> std::cmp::Ordering {
        match (self, other) {
            (LabelValue::Num(a), LabelValue::Num(b)) => a.total_cmp(b),
            (LabelValue::Str(a), LabelValue::Str(b)) => a.cmp(b),
            // Mixed numeric/string columns aren't a documented case; sort
            // numeric labels before string labels rather than panicking.
            (LabelValue::Num(_), LabelValue::Str(_)) => std::cmp::Ordering::Less,
            (LabelValue::Str(_), LabelValue::Num(_)) => std::cmp::Ordering::Greater,
        }
    }
}

fn is_numeric_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn is_string_like_dtype(dt: &DataType) -> bool {
    dt.is_string() || dt.is_categorical() || dt.is_enum()
}

fn series_to_label_values(s: &Series) -> PolarsResult<Vec<Option<LabelValue>>> {
    if is_string_like_dtype(s.dtype()) {
        let casted = s.cast(&DataType::String)?;
        let ca = casted.str()?;
        Ok(ca
            .into_iter()
            .map(|opt| opt.map(|v| LabelValue::Str(v.to_string())))
            .collect())
    } else {
        let casted = s.cast(&DataType::Float64)?;
        let ca = casted.f64()?;
        // Treat NaN/inf as missing (consistent with krippendorffs_alpha and the
        // pairwise-complete contract) rather than as a distinct label value.
        Ok(ca
            .into_iter()
            .map(|opt| opt.filter(|v| v.is_finite()).map(LabelValue::Num))
            .collect())
    }
}

/// Pairwise-complete label encoding for Cohen's kappa: drops row `i` if
/// either column is null there, builds the sorted-union label list, and
/// returns each surviving row re-expressed as indices into that list (plus
/// the label count). `n_labels == 0` means zero pairwise-complete rows.
fn encode_pairwise_complete_labels(
    a: &Series,
    b: &Series,
) -> PolarsResult<(Vec<usize>, Vec<usize>, usize)> {
    let a_vals = series_to_label_values(a)?;
    let b_vals = series_to_label_values(b)?;
    let n = a_vals.len().min(b_vals.len());

    let mut rows: Vec<(LabelValue, LabelValue)> = Vec::new();
    for i in 0..n {
        if let (Some(av), Some(bv)) = (&a_vals[i], &b_vals[i]) {
            rows.push((av.clone(), bv.clone()));
        }
    }

    let mut domain: Vec<LabelValue> = Vec::with_capacity(rows.len() * 2);
    for (av, bv) in &rows {
        domain.push(av.clone());
        domain.push(bv.clone());
    }
    domain.sort_by(|x, y| x.cmp_key(y));
    domain.dedup_by(|x, y| x.cmp_key(y) == std::cmp::Ordering::Equal);

    let label_index = |domain: &[LabelValue], v: &LabelValue| -> usize {
        domain
            .binary_search_by(|x| x.cmp_key(v))
            .expect("label value must be present in its own domain")
    };

    let mut a_idx = Vec::with_capacity(rows.len());
    let mut b_idx = Vec::with_capacity(rows.len());
    for (av, bv) in &rows {
        a_idx.push(label_index(&domain, av));
        b_idx.push(label_index(&domain, bv));
    }

    Ok((a_idx, b_idx, domain.len()))
}

fn parse_kappa_weights(w: &Option<String>) -> PolarsResult<crate::metrics::KappaWeights> {
    match w.as_deref() {
        None => Ok(crate::metrics::KappaWeights::None),
        Some("linear") => Ok(crate::metrics::KappaWeights::Linear),
        Some("quadratic") => Ok(crate::metrics::KappaWeights::Quadratic),
        Some(other) => Err(PolarsError::ComputeError(
            format!("cohens_kappa: unknown weights {other:?} (expected 'linear' or 'quadratic')")
                .into(),
        )),
    }
}

fn parse_alpha_level(level: &str) -> PolarsResult<crate::metrics::AlphaLevel> {
    match level {
        "nominal" => Ok(crate::metrics::AlphaLevel::Nominal),
        "ordinal" => Ok(crate::metrics::AlphaLevel::Ordinal),
        "interval" => Ok(crate::metrics::AlphaLevel::Interval),
        "ratio" => Ok(crate::metrics::AlphaLevel::Ratio),
        other => Err(PolarsError::ComputeError(
            format!(
                "krippendorffs_alpha: unknown level {other:?} \
                 (expected 'nominal', 'ordinal', 'interval', or 'ratio')"
            )
            .into(),
        )),
    }
}

/// Builds Krippendorff's-alpha "units": one `Vec<f64>` per row, holding that
/// row's non-null ratings across the `M` rater columns in `inputs`.
///
/// `level == Nominal` accepts numeric or string-like (`String`/
/// `Categorical`/`Enum`) columns; string-like values are encoded to
/// arbitrary-but-consistent `f64` ids (order doesn't matter -- nominal's
/// difference function only tests equality). `Ordinal`/`Interval`/`Ratio`
/// require every column to already be numeric (raised as `ComputeError`
/// otherwise -- matches the `krippendorff` package, which needs numeric
/// values to compute a numeric difference for those levels).
fn build_reliability_units(
    inputs: &[Series],
    level: crate::metrics::AlphaLevel,
) -> PolarsResult<Vec<Vec<f64>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows = inputs[0].len();

    let use_string_encoding = matches!(level, crate::metrics::AlphaLevel::Nominal)
        && inputs.iter().any(|s| is_string_like_dtype(s.dtype()));

    if !matches!(level, crate::metrics::AlphaLevel::Nominal) {
        for s in inputs {
            if !is_numeric_dtype(s.dtype()) {
                return Err(PolarsError::ComputeError(
                    format!(
                        "krippendorffs_alpha: level={level:?} requires numeric rater columns, \
                         got column {:?} with dtype {:?}",
                        s.name(),
                        s.dtype()
                    )
                    .into(),
                ));
            }
        }
    }

    let mut columns: Vec<Vec<Option<f64>>> = Vec::with_capacity(inputs.len());
    if use_string_encoding {
        let mut ids: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut next_id = 0f64;
        for s in inputs {
            let casted = s.cast(&DataType::String)?;
            let ca = casted.str()?;
            let mut col: Vec<Option<f64>> = Vec::with_capacity(ca.len());
            for v in ca.into_iter() {
                col.push(v.map(|text| {
                    *ids.entry(text.to_string()).or_insert_with(|| {
                        let id = next_id;
                        next_id += 1.0;
                        id
                    })
                }));
            }
            columns.push(col);
        }
    } else {
        for s in inputs {
            let casted = s.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            columns.push(ca.into_iter().collect());
        }
    }

    let mut units: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let mut ratings = Vec::with_capacity(columns.len());
        for col in &columns {
            if row < col.len() {
                if let Some(v) = col[row] {
                    // Treat NaN/inf as missing (matches the reference
                    // `krippendorff` package and the null-as-missing contract);
                    // a non-finite value would otherwise poison the domain sort.
                    if v.is_finite() {
                        ratings.push(v);
                    }
                }
            }
        }
        units.push(ratings);
    }
    Ok(units)
}

/// `Struct{value: Float64, ci_low: Float64, ci_high: Float64}` -- the
/// `output_type_func` for the `_ci` plugin variants. Built natively with
/// `StructChunked` (the `dtype-struct` Cargo feature is already enabled),
/// not the JSON-string decode convention `cluster_embeddings` uses: a
/// `returns_scalar` aggregation composes awkwardly with `map_batches`
/// inside `.agg()`.
fn irr_ci_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static(""),
        DataType::Struct(vec![
            Field::new(PlSmallStr::from_static("value"), DataType::Float64),
            Field::new(PlSmallStr::from_static("ci_low"), DataType::Float64),
            Field::new(PlSmallStr::from_static("ci_high"), DataType::Float64),
        ]),
    ))
}

fn build_irr_struct(
    value: Option<f64>,
    ci_low: Option<f64>,
    ci_high: Option<f64>,
) -> PolarsResult<Series> {
    let value_s =
        Float64Chunked::from_slice_options(PlSmallStr::from_static("value"), &[value]).into_series();
    let ci_low_s =
        Float64Chunked::from_slice_options(PlSmallStr::from_static("ci_low"), &[ci_low]).into_series();
    let ci_high_s =
        Float64Chunked::from_slice_options(PlSmallStr::from_static("ci_high"), &[ci_high])
            .into_series();
    let out = StructChunked::from_series(
        PlSmallStr::from_static(""),
        1,
        [value_s, ci_low_s, ci_high_s].iter(),
    )?;
    Ok(out.into_series())
}

#[derive(Debug, Deserialize)]
pub struct KappaKwargs {
    #[serde(default)]
    weights: Option<String>,
    #[serde(default)]
    n_bootstrap: Option<usize>,
    #[serde(default = "default_irr_ci")]
    ci: f64,
    #[serde(default)]
    seed: u64,
}

/// Cohen's kappa point estimate between two rater columns (`inputs[0]`,
/// `inputs[1]`). Pairwise-complete: a row is dropped if either column is
/// null there. Returns Polars null for zero pairwise-complete rows, and
/// `f64::NAN` (not null) when expected-by-chance agreement is zero (e.g. a
/// single observed category) -- matching
/// `sklearn.metrics.cohen_kappa_score`'s `0/0 -> nan` convention exactly.
#[polars_expr(output_type=Float64)]
fn cohens_kappa(inputs: &[Series], kwargs: KappaKwargs) -> PolarsResult<Series> {
    let weights = parse_kappa_weights(&kwargs.weights)?;
    let (a_idx, b_idx, n_labels) = encode_pairwise_complete_labels(&inputs[0], &inputs[1])?;

    let value = if n_labels == 0 {
        None
    } else {
        Some(crate::metrics::cohens_kappa(&a_idx, &b_idx, n_labels, weights))
    };

    Ok(Float64Chunked::from_slice_options(PlSmallStr::from_static(""), &[value]).into_series())
}

/// Cohen's kappa with an optional case-resampling bootstrap CI. Returns
/// `Struct{value, ci_low, ci_high}`; `ci_low`/`ci_high` are null when
/// `n_bootstrap` is `None` or zero, or when more than half the resamples
/// were degenerate (see `crate::metrics::bootstrap_ci`).
#[polars_expr(output_type_func=irr_ci_output_type)]
fn cohens_kappa_ci(inputs: &[Series], kwargs: KappaKwargs) -> PolarsResult<Series> {
    let weights = parse_kappa_weights(&kwargs.weights)?;
    let (a_idx, b_idx, n_labels) = encode_pairwise_complete_labels(&inputs[0], &inputs[1])?;

    if n_labels == 0 {
        return build_irr_struct(None, None, None);
    }

    let point = crate::metrics::cohens_kappa(&a_idx, &b_idx, n_labels, weights);
    let n = a_idx.len();
    let b = kwargs.n_bootstrap.unwrap_or(0);

    let (ci_low, ci_high) = if b > 0 {
        match crate::metrics::bootstrap_ci(
            |idx: &[usize]| {
                let ra: Vec<usize> = idx.iter().map(|&i| a_idx[i]).collect();
                let rb: Vec<usize> = idx.iter().map(|&i| b_idx[i]).collect();
                crate::metrics::cohens_kappa(&ra, &rb, n_labels, weights)
            },
            n,
            b,
            kwargs.ci,
            kwargs.seed,
        ) {
            Some((lo, hi)) => (Some(lo), Some(hi)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    build_irr_struct(Some(point), ci_low, ci_high)
}

#[derive(Debug, Deserialize)]
pub struct AlphaKwargs {
    #[serde(default = "default_alpha_level")]
    level: String,
    #[serde(default)]
    n_bootstrap: Option<usize>,
    #[serde(default = "default_irr_ci")]
    ci: f64,
    #[serde(default)]
    seed: u64,
}

fn default_alpha_level() -> String {
    "nominal".to_string()
}

/// Krippendorff's alpha point estimate over `M` rater columns (`inputs`).
/// Nulls are missing ratings (alpha's whole reason for being); units
/// (rows) with fewer than 2 non-null ratings are excluded. Returns Polars
/// null when no unit has >= 2 ratings (the reference package raises there);
/// returns `1.0` for a single-category domain (trivially perfect
/// agreement; the reference package also raises there) -- both documented
/// divergences, see `crate::metrics::krippendorff_alpha`.
#[polars_expr(output_type=Float64)]
fn krippendorffs_alpha(inputs: &[Series], kwargs: AlphaKwargs) -> PolarsResult<Series> {
    let level = parse_alpha_level(&kwargs.level)?;
    let units = build_reliability_units(inputs, level)?;
    let value = crate::metrics::krippendorff_alpha(&units, level);
    Ok(Float64Chunked::from_slice_options(PlSmallStr::from_static(""), &[value]).into_series())
}

/// Krippendorff's alpha with an optional case (unit) bootstrap CI. Returns
/// `Struct{value, ci_low, ci_high}`, same null/degenerate conventions as
/// `cohens_kappa_ci`.
#[polars_expr(output_type_func=irr_ci_output_type)]
fn krippendorffs_alpha_ci(inputs: &[Series], kwargs: AlphaKwargs) -> PolarsResult<Series> {
    let level = parse_alpha_level(&kwargs.level)?;
    let units = build_reliability_units(inputs, level)?;

    let value = match crate::metrics::krippendorff_alpha(&units, level) {
        Some(v) => v,
        None => return build_irr_struct(None, None, None),
    };

    let n = units.len();
    let b = kwargs.n_bootstrap.unwrap_or(0);

    let (ci_low, ci_high) = if b > 0 {
        match crate::metrics::bootstrap_ci(
            |idx: &[usize]| {
                let resampled: Vec<Vec<f64>> = idx.iter().map(|&i| units[i].clone()).collect();
                crate::metrics::krippendorff_alpha(&resampled, level).unwrap_or(f64::NAN)
            },
            n,
            b,
            kwargs.ci,
            kwargs.seed,
        ) {
            Some((lo, hi)) => (Some(lo), Some(hi)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    build_irr_struct(Some(value), ci_low, ci_high)
}
