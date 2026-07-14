use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider, Usage};
use reqwest::Client;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    types::{
        CachePointBlock, CachePointType, ContentBlock, ConversationRole,
        Message as BedrockMessage, SystemContentBlock,
    },
    Client as AwsBedrockClient,
};

/// Default Bedrock model (cross-region inference profile for Claude Haiku 4.5)
pub const DEFAULT_BEDROCK_MODEL: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";

/// AWS Bedrock clients cached per region. Loading the AWS config chain is
/// expensive, so we do it once per region instead of once per request.
static BEDROCK_CLIENTS: LazyLock<tokio::sync::Mutex<HashMap<String, AwsBedrockClient>>> =
    LazyLock::new(|| tokio::sync::Mutex::new(HashMap::new()));

async fn bedrock_client_for_region(region: &str) -> AwsBedrockClient {
    let mut clients = BEDROCK_CLIENTS.lock().await;
    if let Some(client) = clients.get(region) {
        return client.clone();
    }
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = AwsBedrockClient::new(&sdk_config);
    clients.insert(region.to_string(), client.clone());
    client
}

#[derive(Clone)]
pub struct BedrockClient {
    model: String,
    region: String,
}

impl BedrockClient {
    pub fn new_with_model(model: &str) -> Self {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        Self {
            model: model.to_string(),
            region,
        }
    }

    pub fn with_region(mut self, region: &str) -> Self {
        self.region = region.to_string();
        self
    }

    fn convert_messages_to_bedrock(&self, messages: &[Message]) -> (Vec<SystemContentBlock>, Vec<BedrockMessage>) {
        let mut system_blocks: Vec<SystemContentBlock> = Vec::new();
        let mut bedrock_messages = Vec::new();
        let mut cache_requested = false;

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    system_blocks.push(SystemContentBlock::Text(msg.content.clone()));
                    if msg.cache_control.is_some() {
                        cache_requested = true;
                    }
                },
                role => {
                    let conversation_role = if role == "assistant" {
                        ConversationRole::Assistant
                    } else {
                        // Default other roles to user
                        ConversationRole::User
                    };
                    if let Ok(bedrock_msg) = BedrockMessage::builder()
                        .role(conversation_role)
                        .content(ContentBlock::Text(msg.content.clone()))
                        .build()
                    {
                        bedrock_messages.push(bedrock_msg);
                    }
                }
            }
        }

        // Bedrock prompt caching: a CachePoint after the cached content marks the
        // prefix boundary. Bedrock supports only the default (~5 minute) cache
        // type; the extended 1h TTL of the Anthropic direct API is unavailable here.
        if cache_requested && !system_blocks.is_empty() {
            if let Ok(cache_point) = CachePointBlock::builder()
                .r#type(CachePointType::Default)
                .build()
            {
                system_blocks.push(SystemContentBlock::CachePoint(cache_point));
            }
        }

        (system_blocks, bedrock_messages)
    }
}

impl Default for BedrockClient {
    fn default() -> Self {
        Self::new_with_model(DEFAULT_BEDROCK_MODEL)
    }
}

#[async_trait]
impl ModelClient for BedrockClient {
    fn provider(&self) -> Provider {
        Provider::Bedrock
    }

    fn api_endpoint(&self) -> String {
        // Bedrock doesn't use HTTP endpoints directly, but we need to implement this
        format!("https://bedrock-runtime.{}.amazonaws.com", self.region)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn format_messages(&self, messages: &[Message]) -> Value {
        // Convert to JSON for compatibility with the trait
        let formatted_messages: Vec<Value> = messages
            .iter()
            .map(|msg| {
                json!({
                    "role": msg.role,
                    "content": msg.content
                })
            })
            .collect();

        json!(formatted_messages)
    }

    fn format_request_body(&self, messages: &[Message], _schema: Option<&str>, _model_name: Option<&str>) -> Value {
        // Bedrock uses the AWS SDK, not HTTP requests; implemented for trait compliance
        json!({
            "messages": self.format_messages(messages)
        })
    }

    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        // For Bedrock, this method won't be used as we handle responses directly in send_request
        Ok(response_text.to_string())
    }

    /// The canonical Bedrock send: the AWS SDK `converse()` call, with usage
    /// captured directly from the SDK response (not via `parse_usage`, which
    /// is unused here -- `parse_response` above is a passthrough since
    /// Bedrock never goes through the generic HTTP JSON-body path). Both
    /// `send_request` and `send_request_structured` (trait defaults) route
    /// through this override, so it is the ONLY Bedrock send path -- no risk
    /// of the two desyncing. Bedrock Converse does not take a JSON schema in
    /// this integration; `schema`/`model_name` are accepted for trait
    /// compliance but unused, and responses are validated post-hoc by the
    /// caller exactly as before.
    async fn send_request_structured_with_usage(
        &self,
        _client: &Client,
        messages: &[Message],
        _schema: Option<&str>,
        _model_name: Option<&str>,
    ) -> Result<(String, Option<Usage>), ModelClientError> {
        // We don't use the reqwest client for Bedrock; we use the AWS SDK
        let bedrock_client = bedrock_client_for_region(&self.region).await;

        let (system_blocks, bedrock_messages) = self.convert_messages_to_bedrock(messages);

        let mut converse_request = bedrock_client
            .converse()
            .model_id(&self.model)
            .set_messages(Some(bedrock_messages));

        if !system_blocks.is_empty() {
            converse_request = converse_request.set_system(Some(system_blocks));
        }

        let t0 = Instant::now();
        let response = converse_request
            .send()
            .await
            .map_err(|e| ModelClientError::ParseError(format!("Bedrock API error: {e}")))?;
        let latency_ms = t0.elapsed().as_millis() as i64;

        let mut usage = Usage {
            latency_ms: Some(latency_ms),
            ..Default::default()
        };
        if let Some(sdk_usage) = response.usage.as_ref() {
            usage.input_tokens = Some(sdk_usage.input_tokens as i64);
            usage.output_tokens = Some(sdk_usage.output_tokens as i64);
            usage.cached_tokens = sdk_usage.cache_read_input_tokens.map(|v| v as i64);
        }

        // Extract the response text
        if let Some(output) = response.output {
            if let Ok(message) = output.as_message() {
                for content in &message.content {
                    if let Ok(text) = content.as_text() {
                        return Ok((text.clone(), Some(usage)));
                    }
                }
            }
        }

        Err(ModelClientError::ParseError("No text content found in Bedrock response".to_string()))
    }

    fn get_api_key(&self) -> String {
        // Bedrock uses AWS credentials, not API keys
        String::new()
    }
}
