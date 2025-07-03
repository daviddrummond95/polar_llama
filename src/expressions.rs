#![allow(clippy::unused_unit)]
use crate::utils::*;
use crate::model_client::{Provider, Message};
use once_cell::sync::Lazy;
use polars::prelude::*;
use polars_core::prelude::CompatLevel;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::borrow::Cow;
use tokio::runtime::Runtime;
use std::str::FromStr;

// Initialize a global runtime for all async operations
static RT: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

#[derive(Debug, Deserialize)]
pub struct InferenceKwargs {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
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
    // First input: prompt/messages column
    let ca_prompt: &StringChunked = inputs[0].str()?;

    // Optional second input: provider/tool selector column
    let provider_ca: Option<&StringChunked> = if inputs.len() > 1 {
        Some(inputs[1].str()?)
    } else {
        None
    };

    // Pre-compute model fallback (if supplied as kwarg)
    let model_kwarg = kwargs.model.clone();

    // Collect the results row-by-row
    let out: StringChunked = ca_prompt
        .into_iter()
        .enumerate()
        .map(|(idx, opt_prompt)| {
            opt_prompt.and_then(|prompt| {
                // Determine provider string for this row: priority -> column value -> kwarg
                let provider_str_opt: Option<String> = provider_ca
                    .and_then(|ca| ca.get(idx).map(|s| s.to_string()))
                    .or_else(|| kwargs.provider.clone());

                // Parse provider (fallback to OpenAI)
                let provider_enum = provider_str_opt
                    .as_deref()
                    .and_then(parse_provider)
                    .unwrap_or(Provider::OpenAI);

                // Determine model: kwarg if provided else default for provider
                let model_for_provider = model_kwarg.clone().unwrap_or_else(|| {
                    get_default_model(provider_enum).to_string()
                });

                // NOTE: The synchronous path currently only supports OpenAI. We fall back to
                // `fetch_api_response_sync` regardless of provider until a provider-agnostic
                // sync implementation is available.
                let response = fetch_api_response_sync(prompt, &model_for_provider).ok();

                response
            })
        })
        .collect_ca(ca_prompt.name().clone());

    Ok(out.into_series())
}

// Register the asynchronous inference function with Polars
#[polars_expr(output_type=String)]
fn inference_async(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    // First input: prompt column
    let ca_prompt: &StringChunked = inputs[0].str()?;

    // Optional second input: provider column
    let provider_ca: Option<&StringChunked> = if inputs.len() > 1 {
        Some(inputs[1].str()?)
    } else {
        None
    };

    // Prepare a vector of futures, one per row
    let mut tasks = Vec::with_capacity(ca_prompt.len());

    for (idx, opt_prompt) in ca_prompt.into_iter().enumerate() {
        if let Some(prompt) = opt_prompt {
            // Determine provider string for this row
            let provider_str_opt: Option<String> = provider_ca
                .and_then(|ca| ca.get(idx).map(|s| s.to_string()))
                .or_else(|| kwargs.provider.clone());

            // Parse provider enum (fallback to OpenAI)
            let provider_enum = provider_str_opt
                .as_deref()
                .and_then(parse_provider)
                .unwrap_or(Provider::OpenAI);

            // Determine model
            let model_for_provider = if let Some(ref m) = kwargs.model {
                m.clone()
            } else {
                get_default_model(provider_enum).to_string()
            };

            // Clone data to move into async task
            let prompt_owned = prompt.to_owned();
            let model_clone = model_for_provider.clone();

            tasks.push(async move {
                let msgs = vec![prompt_owned];
                fetch_data_with_provider(&msgs, provider_enum, &model_clone).await.into_iter().next().unwrap_or(None)
            });
        } else {
            // Push a task that immediately resolves to None to keep indexing
            tasks.push(async { None });
        }
    }

    // Execute all tasks concurrently on the global runtime
    let results: Vec<Option<String>> = RT.block_on(async move { futures::future::join_all(tasks).await });

    let out = StringChunked::from_iter_options(ca_prompt.name().clone(), results.into_iter());

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
            Cow::Owned(format!(
                "{{\"role\": \"{}\", \"content\": \"{}\"}}",
                message_type, value
            ))
        })
    });
    Ok(out.into_series())
}

// New function to handle JSON arrays of messages
#[polars_expr(output_type=String)]
fn inference_messages(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    // First input: JSON arrays of messages
    let ca_messages: &StringChunked = inputs[0].str()?;

    // Optional second input: provider column
    let provider_ca: Option<&StringChunked> = if inputs.len() > 1 {
        Some(inputs[1].str()?)
    } else {
        None
    };

    // Build message arrays per row now because we will need them inside async tasks
    let mut parsed_message_arrays: Vec<Vec<Message>> = Vec::with_capacity(ca_messages.len());
    for opt in ca_messages.into_iter() {
        let parsed = opt
            .and_then(|s| crate::utils::parse_message_json(s).ok())
            .unwrap_or_default();
        parsed_message_arrays.push(parsed);
    }

    // Prepare async tasks per row
    let mut tasks = Vec::with_capacity(parsed_message_arrays.len());

    for (idx, messages) in parsed_message_arrays.into_iter().enumerate() {
        if messages.is_empty() {
            tasks.push(async { None });
            continue;
        }

        // Determine provider string for this row
        let provider_str_opt: Option<String> = provider_ca
            .and_then(|ca| ca.get(idx).map(|s| s.to_string()))
            .or_else(|| kwargs.provider.clone());

        let provider_enum = provider_str_opt
            .as_deref()
            .and_then(parse_provider)
            .unwrap_or(Provider::OpenAI);

        let model_for_provider = if let Some(ref m) = kwargs.model {
            m.clone()
        } else {
            get_default_model(provider_enum).to_string()
        };

        tasks.push(async move {
            let message_arrays_ref = vec![messages];
            crate::utils::fetch_data_message_arrays_with_provider(&message_arrays_ref, provider_enum, &model_for_provider)
                .await
                .into_iter()
                .next()
                .unwrap_or(None)
        });
    }

    let results: Vec<Option<String>> = RT.block_on(async move { futures::future::join_all(tasks).await });

    let out = StringChunked::from_iter_options(ca_messages.name().clone(), results.into_iter());

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
