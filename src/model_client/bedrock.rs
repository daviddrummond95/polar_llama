use serde_json::{json, Value};
use async_trait::async_trait;
use super::{ModelClient, ModelClientError, Message, Provider};
use reqwest::Client;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    types::{
        ContentBlock, ConversationRole, Message as BedrockMessage, SystemContentBlock,
    },
    Client as AwsBedrockClient,
};

pub struct BedrockClient {
    model: String,
    region: String,
    bedrock_client: Option<Box<AwsBedrockClient>>,
}

impl BedrockClient {
    pub fn new_with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
            region: "us-east-1".to_string(), // Default region
            bedrock_client: None,
        }
    }
    
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = region.to_string();
        self
    }
    
    async fn get_bedrock_client(&mut self) -> Result<&AwsBedrockClient, ModelClientError> {
        if self.bedrock_client.is_none() {
            let sdk_config = aws_config::defaults(BehaviorVersion::latest())
                .region(aws_config::Region::new(self.region.clone()))
                .load()
                .await;
            self.bedrock_client = Some(Box::new(AwsBedrockClient::new(&sdk_config)));
        }
        Ok(self.bedrock_client.as_ref().unwrap())
    }
    
    fn convert_messages_to_bedrock(&self, messages: &[Message]) -> (Option<SystemContentBlock>, Vec<BedrockMessage>) {
        let mut system_prompt = None;
        let mut bedrock_messages = Vec::new();
        
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // Store the first system message encountered
                    if system_prompt.is_none() {
                        system_prompt = Some(SystemContentBlock::Text(msg.content.clone()));
                    }
                },
                "user" => {
                    if let Ok(bedrock_msg) = BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(msg.content.clone()))
                        .build()
                    {
                        bedrock_messages.push(bedrock_msg);
                    }
                },
                "assistant" => {
                    if let Ok(bedrock_msg) = BedrockMessage::builder()
                        .role(ConversationRole::Assistant)
                        .content(ContentBlock::Text(msg.content.clone()))
                        .build()
                    {
                        bedrock_messages.push(bedrock_msg);
                    }
                },
                _ => {
                    // Default other roles to user
                    if let Ok(bedrock_msg) = BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(msg.content.clone()))
                        .build()
                    {
                        bedrock_messages.push(bedrock_msg);
                    }
                }
            }
        }
        
        (system_prompt, bedrock_messages)
    }
}

impl Default for BedrockClient {
    fn default() -> Self {
        Self::new_with_model("anthropic.claude-3-haiku-20240307-v1:0")
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
        let mut formatted_messages = Vec::new();

        for msg in messages {
            formatted_messages.push(json!({
                "role": msg.role,
                "content": msg.content
            }));
        }

        json!(formatted_messages)
    }

    fn format_request_body(&self, messages: &[Message], _schema: Option<&str>, _model_name: Option<&str>) -> Value {
        // Bedrock uses AWS SDK, not HTTP requests, but we need to implement this for trait compliance
        json!({
            "messages": self.format_messages(messages)
        })
    }

    fn parse_response(&self, response_text: &str) -> Result<String, ModelClientError> {
        // For Bedrock, this method won't be used as we handle responses directly in send_request
        // But we need to implement it for trait compliance
        Ok(response_text.to_string())
    }
    
    async fn send_request(&self, _client: &Client, messages: &[Message]) -> Result<String, ModelClientError> {
        // We don't use the reqwest client for Bedrock, we use the AWS SDK
        let mut bedrock_client_ref = self.clone();
        let bedrock_client = bedrock_client_ref.get_bedrock_client().await
            .map_err(|e| ModelClientError::ParseError(format!("Failed to create Bedrock client: {e}")))?;
        
        let (system_prompt, bedrock_messages) = self.convert_messages_to_bedrock(messages);
        
        let mut converse_request = bedrock_client
            .converse()
            .model_id(&self.model)
            .set_messages(Some(bedrock_messages));
        
        if let Some(system) = system_prompt {
            converse_request = converse_request.system(system);
        }
        
        let response = converse_request
            .send()
            .await
            .map_err(|e| ModelClientError::ParseError(format!("Bedrock API error: {e}")))?;
        
        // Extract the response text
        if let Some(output) = response.output {
            if output.is_message() {
                if let Ok(message) = output.as_message() {
                    for content in &message.content {
                        if content.is_text() {
                            if let Ok(text) = content.as_text() {
                                return Ok(text.clone());
                            }
                        }
                    }
                }
            }
        }
        
        Err(ModelClientError::ParseError("No text content found in Bedrock response".to_string()))
    }
    
    fn get_api_key(&self) -> String {
        // Bedrock uses AWS credentials, not API keys
        // AWS SDK handles authentication automatically
        String::new()
    }
}

// Implement Clone for BedrockClient to allow cloning for async operations
impl Clone for BedrockClient {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            region: self.region.clone(),
            bedrock_client: None, // Don't clone the client, it will be recreated as needed
        }
    }
} 