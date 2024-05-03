#![allow(clippy::unused_unit)]
use crate::utils::*;
use once_cell::sync::Lazy;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
// use serde::{Deserialize, Serialize};
use std::fmt::Write;
use tokio::runtime::Runtime;

// Initialize a global runtime for all async operations
static RT: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

#[polars_expr(output_type=String)]
fn inference(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out = ca.apply_to_buffer(|value: &str, output: &mut String| {
        let response = fetch_api_response_sync(&value, "gpt-4-turbo");
        response.unwrap().chars().for_each(|c| output.push(c));
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn inference_async(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let messages: Vec<String> = ca
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_owned()))
        .collect();

    let results = RT.block_on(fetch_data(&messages));

    let string_refs: Vec<Option<&str>> = results.iter().map(|opt| opt.as_deref()).collect();
    let out = StringChunked::from_iter_options("output", string_refs.into_iter());

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

    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        write!(
            output,
            "{{\"role\": \"{}\", \"content\": \"{}\"}}",
            message_type, value
        )
        .unwrap()
    });
    Ok(out.into_series())
}
// To be used later for the OpenAI API parsing
// #[derive(Deserialize)]
// pub struct BodyKwargs {
//     Model: String,
// }

// #[derive(Serialize, Deserialize, Debug)]
// struct ChatCompletion {
//     id: String,
//     object: String,
//     created: i64,
//     model: String,
//     choices: Vec<Choice>,
//     usage: Usage,
//     system_fingerprint: String,
// }

// #[derive(Serialize, Deserialize, Debug)]
// struct Choice {
//     index: i32,
//     message: Message,
//     logprobs: Option<serde_json::Value>, // Use serde_json::Value if the structure is not defined or varies
//     finish_reason: String,
// }

// #[derive(Serialize, Deserialize, Debug)]
// struct Message {
//     role: String,
//     content: String,
// }

// #[derive(Serialize, Deserialize, Debug)]
// struct Usage {
//     prompt_tokens: i32,
//     completion_tokens: i32,
//     total_tokens: i32,
// }
