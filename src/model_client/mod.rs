pub mod openai;
pub mod anthropic;
pub mod gemini;
pub mod groq;
pub mod bedrock;

use reqwest::Client;
use std::error::Error;
use std::fmt;
use serde_json::Value;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use futures;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
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
            _ => Err(format!("Unknown provider: {}", s)),
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
            ModelClientError::Http(code, ref message) => write!(f, "HTTP Error {}: {}", code, message),
            ModelClientError::Serialization(ref err) => write!(f, "Serialization Error: {}", err),
            ModelClientError::RequestError(ref err) => write!(f, "Request Error: {}", err),
            ModelClientError::ParseError(ref err) => write!(f, "Parse Error: {}", err),
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
    
    /// Send a request to the API
    async fn send_request(&self, client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        let api_key = self.get_api_key();
        let body = serde_json::to_string(&self.format_request_body(messages))?;
        
        let response = client.post(self.api_endpoint())
            .bearer_auth(api_key)
            .header("Content-Type", "application/json")
            .body(body)
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
    fn format_request_body(&self, messages: &[Message]) -> Value {
        let formatted_messages = self.format_messages(messages);
        serde_json::json!({
            "model": self.model_name(),
            "messages": formatted_messages
        })
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

/// The main function to fetch data from model providers
pub async fn fetch_data_generic<T: ModelClient + Sync + ?Sized>(
    client: &T,
    messages: &[String]
) -> Vec<Option<String>> {
    let reqwest_client = Client::new();
    
    let fetch_tasks = messages.iter().map(|content| {
        let formatted_message = Message {
            role: "user".to_string(),
            content: content.clone(),
        };
        let messages = vec![formatted_message];
        let reqwest_client = &reqwest_client;
        
        async move {
            client.send_request(reqwest_client, &messages).await.ok()
        }
    }).collect::<Vec<_>>();
    
    futures::future::join_all(fetch_tasks).await
}

/// Enhanced function to fetch data that supports either single messages or arrays of messages
pub async fn fetch_data_generic_enhanced<T: ModelClient + Sync + ?Sized>(
    client: &T,
    message_arrays: &[Vec<Message>]
) -> Vec<Option<String>> {
    let reqwest_client = Client::new();
    
    let fetch_tasks = message_arrays.iter().map(|messages| {
        let messages = messages.clone();
        let reqwest_client = &reqwest_client;
        
        async move {
            client.send_request(reqwest_client, &messages).await.ok()
        }
    }).collect::<Vec<_>>();
    
    futures::future::join_all(fetch_tasks).await
}

/// Example function showing how to use the different model clients with specific models
pub async fn example_usage(messages: &[String], provider_str: &str, model: &str) -> Vec<Option<String>> {
    // Parse provider string to Provider enum
    let provider = Provider::from_str(provider_str).unwrap_or(Provider::OpenAI);
    
    // Create appropriate client with specified model
    let client = create_client(provider, model);
    
    // Use client with generic fetch function
    fetch_data_generic(&*client, messages).await
}

/// Enhanced example function supporting message arrays
pub async fn example_usage_enhanced(
    message_arrays: &[Vec<Message>], 
    provider_str: &str, 
    model: &str
) -> Vec<Option<String>> {
    // Parse provider string to Provider enum
    let provider = Provider::from_str(provider_str).unwrap_or(Provider::OpenAI);
    
    // Create appropriate client with specified model
    let client = create_client(provider, model);
    
    // Use client with enhanced generic fetch function
    fetch_data_generic_enhanced(&*client, message_arrays).await
} 