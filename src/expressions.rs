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
    let ca: &StringChunked = inputs[0].str()?;
    
    // Default model if not provided
    let model = kwargs.model.unwrap_or_else(|| "gpt-4-turbo".to_string());
    
    let out = ca.apply(|opt_value| {
        opt_value.map(|value| {
            // If provider is specified, use fetch_api_response_with_provider
            let response = match &kwargs.provider {
                Some(provider_str) => {
                    // Try to parse provider string to Provider enum
                    if let Some(_provider) = parse_provider(provider_str) {
                        // For now, we'll still use OpenAI since we don't have a sync version with provider
                        fetch_api_response_sync(value, &model)
                    } else {
                        // Default to OpenAI if provider can't be parsed
                        fetch_api_response_sync(value, &model)
                    }
                },
                None => fetch_api_response_sync(value, &model),
            };
            Cow::Owned(response.unwrap_or_default())
        })
    });
    Ok(out.into_series())
}

// Register the asynchronous inference function with Polars
#[polars_expr(output_type=String)]
fn inference_async(inputs: &[Series], kwargs: InferenceKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let messages: Vec<String> = ca
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_owned()))
        .collect();

    // Get results based on provider and model
    let results = match (&kwargs.provider, &kwargs.model) {
        (Some(provider_str), Some(model)) => {
            // Try to parse provider string to Provider enum
            if let Some(provider) = parse_provider(provider_str) {
                // Use provider and model
                RT.block_on(fetch_data_with_provider(&messages, provider, model))
            } else {
                // Default to OpenAI if provider can't be parsed
                RT.block_on(fetch_data_with_provider(&messages, Provider::OpenAI, model))
            }
        },
        (Some(provider_str), None) => {
            // Try to parse provider string to Provider enum
            if let Some(provider) = parse_provider(provider_str) {
                // Use provider with default model
                let default_model = get_default_model(provider);
                RT.block_on(fetch_data_with_provider(&messages, provider, default_model))
            } else {
                // Default to OpenAI if provider can't be parsed
                RT.block_on(fetch_data(&messages))
            }
        },
        (None, Some(model)) => {
            // Use default provider (OpenAI) with specified model
            RT.block_on(fetch_data_with_provider(&messages, Provider::OpenAI, model))
        },
        (None, None) => {
            // Use default provider and model
            RT.block_on(fetch_data(&messages))
        },
    };

    let string_refs: Vec<Option<String>> = results.into_iter().collect();
    let out = StringChunked::from_iter_options(ca.name().clone(), string_refs.into_iter());

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
    let ca: &StringChunked = inputs[0].str()?;
    
    // Convert string inputs (JSON arrays) to vectors of messages
    let message_arrays: Vec<Vec<Message>> = ca
        .into_iter()
        .filter_map(|opt| opt.map(|s| {
            // Parse the JSON string into a vector of Messages
            crate::utils::parse_message_json(s).unwrap_or_default()
        }))
        .collect();
    
    // Get results based on provider and model
    let results = match (&kwargs.provider, &kwargs.model) {
        (Some(provider_str), Some(model)) => {
            // Try to parse provider string to Provider enum
            if let Some(provider) = parse_provider(provider_str) {
                // Use provider and model
                RT.block_on(crate::utils::fetch_data_message_arrays_with_provider(&message_arrays, provider, model))
            } else {
                // Default to OpenAI if provider can't be parsed
                RT.block_on(crate::utils::fetch_data_message_arrays(&message_arrays))
            }
        },
        (Some(provider_str), None) => {
            // Try to parse provider string to Provider enum
            if let Some(provider) = parse_provider(provider_str) {
                // Use provider with default model
                let default_model = get_default_model(provider);
                RT.block_on(crate::utils::fetch_data_message_arrays_with_provider(&message_arrays, provider, default_model))
            } else {
                // Default to OpenAI if provider can't be parsed
                RT.block_on(crate::utils::fetch_data_message_arrays(&message_arrays))
            }
        },
        (None, Some(model)) => {
            // Use default provider (OpenAI) with specified model
            RT.block_on(crate::utils::fetch_data_message_arrays_with_provider(&message_arrays, Provider::OpenAI, model))
        },
        (None, None) => {
            // Use default provider and model
            RT.block_on(crate::utils::fetch_data_message_arrays(&message_arrays))
        },
    };

    let string_refs: Vec<Option<String>> = results.into_iter().collect();
    let out = StringChunked::from_iter_options(ca.name().clone(), string_refs.into_iter());

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
