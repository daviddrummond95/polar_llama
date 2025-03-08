#![allow(clippy::unused_unit)]
use crate::utils::*;
use crate::model_client::Provider;
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
                // Default models are defined in the client implementations
                let default_model = match provider {
                    Provider::OpenAI => "gpt-4-turbo",
                    Provider::Anthropic => "claude-3-opus-20240229",
                    Provider::Gemini => "gemini-1.5-pro",
                    Provider::Groq => "llama3-70b-8192",
                };
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
