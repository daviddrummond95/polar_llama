pub mod openai;
pub mod anthropic;
pub mod gemini;
pub mod groq;
pub mod bedrock;
pub mod streaming;

pub use streaming::{StreamEvent, stream_batch};

use reqwest::Client;
use std::error::Error;
use std::fmt;
use std::sync::LazyLock;
use std::time::Duration;
use serde_json::Value;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use futures::StreamExt;

/// Shared HTTP client reused across all requests so connections are pooled
/// instead of being re-established for every batch.
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(600))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| Client::new())
});

/// Access the shared HTTP client.
pub fn http_client() -> &'static Client {
    &HTTP_CLIENT
}

/// Maximum number of concurrent in-flight requests per batch.
/// Override with the POLAR_LLAMA_MAX_CONCURRENCY environment variable.
fn max_concurrency() -> usize {
    std::env::var("POLAR_LLAMA_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64)
}

use crate::cache::CacheControl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Cache control marker for Anthropic/Bedrock
    /// Only serialized when present
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Gemini,
    Groq,
    Bedrock,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Gemini => "gemini",
            Provider::Groq => "groq",
            Provider::Bedrock => "bedrock",
        }
    }
}

// Implement FromStr trait for Provider
impl FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAI),
            "anthropic" => Ok(Provider::Anthropic),
            "gemini" => Ok(Provider::Gemini),
            "groq" => Ok(Provider::Groq),
            "bedrock" => Ok(Provider::Bedrock),
            _ => Err(format!("Unknown provider: {s}")),
        }
    }
}

#[derive(Debug)]
pub enum ModelClientError {
    Http(u16, String),
    Serialization(serde_json::Error),
    RequestError(reqwest::Error),
    ParseError(String),
}

impl fmt::Display for ModelClientError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelClientError::Http(code, ref message) => write!(f, "HTTP Error {code}: {message}"),
            ModelClientError::Serialization(ref err) => write!(f, "Serialization Error: {err}"),
            ModelClientError::RequestError(ref err) => write!(f, "Request Error: {err}"),
            ModelClientError::ParseError(ref err) => write!(f, "Parse Error: {err}"),
        }
    }
}

impl Error for ModelClientError {}

impl From<reqwest::Error> for ModelClientError {
    fn from(err: reqwest::Error) -> Self {
        ModelClientError::RequestError(err)
    }
}

impl From<serde_json::Error> for ModelClientError {
    fn from(err: serde_json::Error) -> Self {
        ModelClientError::Serialization(err)
    }
}

#[async_trait]
pub trait ModelClient {
    /// Get the provider enum
    fn provider(&self) -> Provider;

    /// The name of the client provider
    fn provider_name(&self) -> &str {
        self.provider().as_str()
    }

    /// The API endpoint for the model
    fn api_endpoint(&self) -> String;

    /// The model name to use
    fn model_name(&self) -> &str;

    /// Format messages for the specific provider's API
    fn format_messages(&self, messages: &[Message]) -> Value;

    /// Parse the API response to extract the completion text
    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError>;

    /// Attach provider-specific authentication to a request.
    /// Default: OpenAI-style Bearer token.
    fn apply_auth(&self, request: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
        request.bearer_auth(api_key)
    }

    /// Send a request to the API
    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        self.send_request_structured(client, messages, None, None).await
    }

    /// Send a request with structured output support
    async fn send_request_structured(
        &self,
        client: &Client,
        messages: &[Message],
        schema: Option<&str>,
        model_name: Option<&str>
    ) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = self.format_request_body(messages, schema, model_name);

        let response = self
            .apply_auth(client.post(self.api_endpoint()), &api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if status.is_success() {
            self.parse_response(&text)
        } else {
            Err(ModelClientError::Http(status.as_u16(), text))
        }
    }

    /// Format the full request body including messages and model name
    fn format_request_body(&self, messages: &[Message], schema: Option<&str>, model_name: Option<&str>) -> Value {
        let formatted_messages = self.format_messages(messages);
        let mut body = serde_json::json!({
            "model": self.model_name(),
            "messages": formatted_messages
        });

        // Add structured output support based on provider
        if let Some(schema_str) = schema {
            if let Ok(schema_value) = serde_json::from_str::<Value>(schema_str) {
                match self.provider() {
                    Provider::OpenAI | Provider::Groq => {
                        // OpenAI and Groq use response_format with json_schema
                        body["response_format"] = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "name": model_name.unwrap_or("response"),
                                "strict": true,
                                "schema": schema_value
                            }
                        });
                    },
                    Provider::Anthropic => {
                        // Anthropic uses forced tool use for structured outputs
                        body["tools"] = serde_json::json!([{
                            "name": model_name.unwrap_or("response"),
                            "description": "Extract structured data according to the schema",
                            "input_schema": schema_value
                        }]);
                        body["tool_choice"] = serde_json::json!({
                            "type": "tool",
                            "name": model_name.unwrap_or("response")
                        });
                    },
                    _ => {
                        // For other providers, we'll validate post-response
                    }
                }
            }
        }

        body
    }

    /// Get the API key for this provider
    fn get_api_key(&self) -> String {
        match self.provider() {
            Provider::OpenAI => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            Provider::Anthropic => std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            Provider::Gemini => std::env::var("GEMINI_API_KEY").unwrap_or_default(),
            Provider::Groq => std::env::var("GROQ_API_KEY").unwrap_or_default(),
            Provider::Bedrock => String::new(), // Bedrock uses AWS credentials
        }
    }

    /// Streaming send: emits `StreamEvent`s for this row over `tx` as they
    /// arrive. Default implementation is a buffered fallback so providers
    /// without a native SSE override (Gemini, Bedrock) still work: it awaits
    /// the full response and emits a single `Delta` followed by `Done`.
    async fn send_request_streaming(
        &self,
        client: &Client,
        messages: &[Message],
        row: usize,
        tx: &tokio::sync::mpsc::Sender<(usize, streaming::StreamEvent)>,
    ) -> Result<(), ModelClientError> {
        let text = self.send_request(client, messages).await?;
        let _ = tx.send((row, streaming::StreamEvent::Delta(text))).await;
        let _ = tx.send((row, streaming::StreamEvent::Done)).await;
        Ok(())
    }
}

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingClient {
    /// Get the provider enum
    fn provider(&self) -> Provider;

    /// The name of the client provider
    fn provider_name(&self) -> &str {
        self.provider().as_str()
    }

    /// The API endpoint for embeddings
    fn embedding_endpoint(&self) -> String;

    /// The embedding model name to use
    fn embedding_model(&self) -> &str;

    /// Get the dimensions of the embedding vectors
    fn embedding_dimensions(&self) -> usize;

    /// Generate embeddings for a batch of texts
    async fn generate_embeddings(
        &self,
        client: &Client,
        texts: &[String],
    ) -> Result<Vec<Vec<f64>>, ModelClientError>;

    /// Get the API key for this provider
    fn get_api_key(&self) -> String {
        match self.provider() {
            Provider::OpenAI => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            Provider::Anthropic => std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            Provider::Gemini => std::env::var("GEMINI_API_KEY").unwrap_or_default(),
            Provider::Groq => std::env::var("GROQ_API_KEY").unwrap_or_default(),
            Provider::Bedrock => String::new(), // Bedrock uses AWS credentials
        }
    }
}

/// A JSON schema compiled once per batch, instead of once per row.
enum SchemaCheck {
    None,
    Valid(jsonschema::Validator),
    Invalid(String),
}

impl SchemaCheck {
    fn compile(schema: Option<&str>) -> Self {
        match schema {
            None => SchemaCheck::None,
            Some(schema_str) => {
                let schema_value: Value = match serde_json::from_str(schema_str) {
                    Ok(v) => v,
                    Err(e) => return SchemaCheck::Invalid(format!("Failed to parse schema: {e}")),
                };
                match jsonschema::validator_for(&schema_value) {
                    Ok(validator) => SchemaCheck::Valid(validator),
                    Err(e) => SchemaCheck::Invalid(format!("Failed to compile schema: {e}")),
                }
            }
        }
    }

    /// Validate a response against the compiled schema.
    fn validate(&self, response: &str) -> Result<(), String> {
        let validator = match self {
            SchemaCheck::None => return Ok(()),
            SchemaCheck::Invalid(err) => return Err(err.clone()),
            SchemaCheck::Valid(v) => v,
        };

        let response_value: Value = serde_json::from_str(response)
            .map_err(|e| format!("Failed to parse response as JSON: {e}"))?;

        let errors: Vec<String> = validator
            .iter_errors(&response_value)
            .map(|e| format!("{} at {}", e, e.instance_path()))
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("Schema validation failed: {}", errors.join("; ")))
        }
    }
}

/// Validate JSON response against a JSON schema
pub fn validate_json_schema(response: &str, schema_str: &str) -> Result<(), String> {
    SchemaCheck::compile(Some(schema_str)).validate(response)
}

/// Create an error response object
pub fn create_error_response(error_type: &str, details: &str, raw: Option<&str>) -> String {
    let error_obj = if let Some(raw_content) = raw {
        serde_json::json!({
            "_error": error_type,
            "_details": details,
            "_raw": raw_content
        })
    } else {
        serde_json::json!({
            "_error": error_type,
            "_details": details
        })
    };
    serde_json::to_string(&error_obj).unwrap_or_else(|_| format!(r#"{{"_error": "{}"}}"#, error_type))
}

/// Create a client for the given provider and model
pub fn create_client(provider: Provider, model: &str) -> Box<dyn ModelClient + Send + Sync> {
    match provider {
        Provider::OpenAI => Box::new(openai::OpenAIClient::new_with_model(model)),
        Provider::Anthropic => Box::new(anthropic::AnthropicClient::new_with_model(model)),
        Provider::Gemini => Box::new(gemini::GeminiClient::new_with_model(model)),
        Provider::Groq => Box::new(groq::GroqClient::new_with_model(model)),
        Provider::Bedrock => Box::new(bedrock::BedrockClient::new_with_model(model)),
    }
}

/// Core batch runner: sends all requests concurrently (bounded by
/// `max_concurrency`) over the shared HTTP client, preserving input order.
///
/// When `errors_as_json` is true, request errors are surfaced as structured
/// error JSON objects; otherwise they map to `None`.
async fn run_one<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[Message],
    schema: Option<&str>,
    model_name: Option<&str>,
    schema_check: &SchemaCheck,
    errors_as_json: bool,
) -> Option<String> {
    match client.send_request_structured(http_client(), messages, schema, model_name).await {
        Ok(response) => {
            match schema_check.validate(&response) {
                Ok(()) => Some(response),
                Err(validation_error) => Some(create_error_response(
                    "validation_failed",
                    &validation_error,
                    Some(&response),
                )),
            }
        },
        Err(e) => {
            eprintln!("Error fetching from {}: {}", client.provider_name(), e);
            if errors_as_json {
                Some(create_error_response("api_error", &e.to_string(), None))
            } else {
                None
            }
        }
    }
}

async fn run_batch<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>,
    errors_as_json: bool,
) -> Vec<Option<String>> {
    let schema_check = SchemaCheck::compile(schema);

    // Collect the futures eagerly so the stream holds concrete future values;
    // requests still only run when polled, bounded by `buffered`.
    let requests: Vec<_> = message_arrays
        .iter()
        .map(|messages| run_one(client, messages, schema, model_name, &schema_check, errors_as_json))
        .collect();

    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}

fn to_user_message_arrays(messages: &[String]) -> Vec<Vec<Message>> {
    messages
        .iter()
        .map(|content| {
            vec![Message {
                role: "user".to_string(),
                content: content.clone(),
                cache_control: None,
            }]
        })
        .collect()
}

/// The main function to fetch data from model providers
pub async fn fetch_data_generic<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[String]
) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, None, None, false).await
}

/// Fetch data with structured output support and validation
pub async fn fetch_data_generic_with_schema<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[String],
    schema: Option<&str>,
    model_name: Option<&str>
) -> Vec<Option<String>> {
    let message_arrays = to_user_message_arrays(messages);
    run_batch(client, &message_arrays, schema, model_name, true).await
}

/// Enhanced function to fetch data that supports either single messages or arrays of messages
pub async fn fetch_data_generic_enhanced<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>]
) -> Vec<Option<String>> {
    run_batch(client, message_arrays, None, None, false).await
}

/// Enhanced function with schema validation for message arrays
pub async fn fetch_data_generic_enhanced_with_schema<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>],
    schema: Option<&str>,
    model_name: Option<&str>
) -> Vec<Option<String>> {
    run_batch(client, message_arrays, schema, model_name, true).await
}

/// Example function showing how to use the different model clients with specific models
pub async fn example_usage(messages: &[String], provider_str: &str, model: &str) -> Vec<Option<String>> {
    let provider = Provider::from_str(provider_str).unwrap_or(Provider::OpenAI);
    let client = create_client(provider, model);
    fetch_data_generic(&*client, messages).await
}

/// Enhanced example function supporting message arrays
pub async fn example_usage_enhanced(
    message_arrays: &[Vec<Message>],
    provider_str: &str,
    model: &str
) -> Vec<Option<String>> {
    let provider = Provider::from_str(provider_str).unwrap_or(Provider::OpenAI);
    let client = create_client(provider, model);
    fetch_data_generic_enhanced(&*client, message_arrays).await
}

async fn embed_one<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    text: &String,
) -> Option<Vec<f64>> {
    match client.generate_embeddings(http_client(), std::slice::from_ref(text)).await {
        Ok(embeddings) => embeddings.into_iter().next(),
        Err(e) => {
            eprintln!("Error generating embedding from {}: {}", client.provider_name(), e);
            None
        }
    }
}

/// Parallel embedding generation function with bounded concurrency
pub async fn fetch_embeddings_generic<T: EmbeddingClient + Sync + ?Sized>(
    client: &T,
    texts: &[String]
) -> Vec<Option<Vec<f64>>> {
    let requests: Vec<_> = texts.iter().map(|text| embed_one(client, text)).collect();

    futures::stream::iter(requests)
        .buffered(max_concurrency())
        .collect()
        .await
}

/// Create an embedding client for the given provider and model
pub fn create_embedding_client(provider: Provider, model: &str) -> Box<dyn EmbeddingClient + Send + Sync> {
    match provider {
        Provider::OpenAI => Box::new(openai::OpenAIEmbeddingClient::new_with_model(model)),
        // Other providers can be added here as they're implemented
        _ => Box::new(openai::OpenAIEmbeddingClient::new_with_model(model)),
    }
}
