//! Cost calculation module for LLM inference.
//!
//! This module provides token counting using tiktoken and cost calculation
//! based on provider pricing data embedded at build time.

use crate::model_client::Provider;
use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Pricing information for a model (costs per 1 million tokens in USD)
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Cost per 1M input tokens in USD
    pub input_cost_per_million: f64,
    /// Cost per 1M output tokens in USD
    pub output_cost_per_million: f64,
}

impl ModelPricing {
    const fn new(input_cost_per_million: f64, output_cost_per_million: f64) -> Self {
        Self {
            input_cost_per_million,
            output_cost_per_million,
        }
    }

    /// Calculate cost for a given number of input tokens
    pub fn calculate_input_cost(&self, token_count: usize) -> f64 {
        (token_count as f64 / 1_000_000.0) * self.input_cost_per_million
    }

    /// Calculate cost for a given number of output tokens
    pub fn calculate_output_cost(&self, token_count: usize) -> f64 {
        (token_count as f64 / 1_000_000.0) * self.output_cost_per_million
    }
}

/// Static pricing data for all supported models.
/// Prices are in USD per 1 million tokens.
/// Last updated: January 2025
///
/// Note: These prices should be periodically updated as providers change their pricing.
static PRICING: Lazy<HashMap<(&'static str, &'static str), ModelPricing>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // ============================================================================
    // OpenAI Models
    // https://openai.com/pricing
    // ============================================================================

    // GPT-4o models
    m.insert(("openai", "gpt-4o"), ModelPricing::new(2.50, 10.00));
    m.insert(("openai", "gpt-4o-2024-11-20"), ModelPricing::new(2.50, 10.00));
    m.insert(("openai", "gpt-4o-2024-08-06"), ModelPricing::new(2.50, 10.00));
    m.insert(("openai", "gpt-4o-2024-05-13"), ModelPricing::new(5.00, 15.00));

    // GPT-4o-mini models
    m.insert(("openai", "gpt-4o-mini"), ModelPricing::new(0.15, 0.60));
    m.insert(("openai", "gpt-4o-mini-2024-07-18"), ModelPricing::new(0.15, 0.60));

    // GPT-4 Turbo models
    m.insert(("openai", "gpt-4-turbo"), ModelPricing::new(10.00, 30.00));
    m.insert(("openai", "gpt-4-turbo-2024-04-09"), ModelPricing::new(10.00, 30.00));
    m.insert(("openai", "gpt-4-turbo-preview"), ModelPricing::new(10.00, 30.00));
    m.insert(("openai", "gpt-4-1106-preview"), ModelPricing::new(10.00, 30.00));
    m.insert(("openai", "gpt-4-0125-preview"), ModelPricing::new(10.00, 30.00));

    // GPT-4 models
    m.insert(("openai", "gpt-4"), ModelPricing::new(30.00, 60.00));
    m.insert(("openai", "gpt-4-0613"), ModelPricing::new(30.00, 60.00));
    m.insert(("openai", "gpt-4-32k"), ModelPricing::new(60.00, 120.00));
    m.insert(("openai", "gpt-4-32k-0613"), ModelPricing::new(60.00, 120.00));

    // GPT-3.5 Turbo models
    m.insert(("openai", "gpt-3.5-turbo"), ModelPricing::new(0.50, 1.50));
    m.insert(("openai", "gpt-3.5-turbo-0125"), ModelPricing::new(0.50, 1.50));
    m.insert(("openai", "gpt-3.5-turbo-1106"), ModelPricing::new(1.00, 2.00));
    m.insert(("openai", "gpt-3.5-turbo-instruct"), ModelPricing::new(1.50, 2.00));

    // o1 reasoning models
    m.insert(("openai", "o1"), ModelPricing::new(15.00, 60.00));
    m.insert(("openai", "o1-2024-12-17"), ModelPricing::new(15.00, 60.00));
    m.insert(("openai", "o1-preview"), ModelPricing::new(15.00, 60.00));
    m.insert(("openai", "o1-preview-2024-09-12"), ModelPricing::new(15.00, 60.00));
    m.insert(("openai", "o1-mini"), ModelPricing::new(3.00, 12.00));
    m.insert(("openai", "o1-mini-2024-09-12"), ModelPricing::new(3.00, 12.00));

    // ============================================================================
    // Anthropic Models
    // https://www.anthropic.com/pricing
    // ============================================================================

    // Claude 3.5 models
    m.insert(("anthropic", "claude-3-5-sonnet-20241022"), ModelPricing::new(3.00, 15.00));
    m.insert(("anthropic", "claude-3-5-sonnet-20240620"), ModelPricing::new(3.00, 15.00));
    m.insert(("anthropic", "claude-3-5-haiku-20241022"), ModelPricing::new(0.80, 4.00));

    // Claude 3 models
    m.insert(("anthropic", "claude-3-opus-20240229"), ModelPricing::new(15.00, 75.00));
    m.insert(("anthropic", "claude-3-sonnet-20240229"), ModelPricing::new(3.00, 15.00));
    m.insert(("anthropic", "claude-3-haiku-20240307"), ModelPricing::new(0.25, 1.25));

    // Claude 2 models (legacy)
    m.insert(("anthropic", "claude-2.1"), ModelPricing::new(8.00, 24.00));
    m.insert(("anthropic", "claude-2.0"), ModelPricing::new(8.00, 24.00));
    m.insert(("anthropic", "claude-instant-1.2"), ModelPricing::new(0.80, 2.40));

    // ============================================================================
    // Google Gemini Models
    // https://ai.google.dev/pricing
    // ============================================================================

    // Gemini 2.0 models
    m.insert(("gemini", "gemini-2.0-flash-exp"), ModelPricing::new(0.00, 0.00)); // Free during preview

    // Gemini 1.5 models
    m.insert(("gemini", "gemini-1.5-pro"), ModelPricing::new(1.25, 5.00));
    m.insert(("gemini", "gemini-1.5-pro-latest"), ModelPricing::new(1.25, 5.00));
    m.insert(("gemini", "gemini-1.5-flash"), ModelPricing::new(0.075, 0.30));
    m.insert(("gemini", "gemini-1.5-flash-latest"), ModelPricing::new(0.075, 0.30));
    m.insert(("gemini", "gemini-1.5-flash-8b"), ModelPricing::new(0.0375, 0.15));

    // Gemini 1.0 models
    m.insert(("gemini", "gemini-1.0-pro"), ModelPricing::new(0.50, 1.50));
    m.insert(("gemini", "gemini-pro"), ModelPricing::new(0.50, 1.50));

    // ============================================================================
    // Groq Models (via Groq Cloud)
    // https://groq.com/pricing/
    // ============================================================================

    // Llama 3.3 models
    m.insert(("groq", "llama-3.3-70b-versatile"), ModelPricing::new(0.59, 0.79));

    // Llama 3.1 models
    m.insert(("groq", "llama-3.1-405b-reasoning"), ModelPricing::new(0.00, 0.00)); // Preview
    m.insert(("groq", "llama-3.1-70b-versatile"), ModelPricing::new(0.59, 0.79));
    m.insert(("groq", "llama-3.1-8b-instant"), ModelPricing::new(0.05, 0.08));

    // Llama 3 models
    m.insert(("groq", "llama3-70b-8192"), ModelPricing::new(0.59, 0.79));
    m.insert(("groq", "llama3-8b-8192"), ModelPricing::new(0.05, 0.08));

    // Mixtral models
    m.insert(("groq", "mixtral-8x7b-32768"), ModelPricing::new(0.24, 0.24));

    // Gemma models
    m.insert(("groq", "gemma2-9b-it"), ModelPricing::new(0.20, 0.20));
    m.insert(("groq", "gemma-7b-it"), ModelPricing::new(0.07, 0.07));

    // ============================================================================
    // AWS Bedrock Models
    // https://aws.amazon.com/bedrock/pricing/
    // Prices are for US East (N. Virginia) region
    // ============================================================================

    // Claude 3.5 via Bedrock
    m.insert(("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"), ModelPricing::new(3.00, 15.00));
    m.insert(("bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0"), ModelPricing::new(3.00, 15.00));
    m.insert(("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0"), ModelPricing::new(0.80, 4.00));

    // Claude 3 via Bedrock
    m.insert(("bedrock", "anthropic.claude-3-opus-20240229-v1:0"), ModelPricing::new(15.00, 75.00));
    m.insert(("bedrock", "anthropic.claude-3-sonnet-20240229-v1:0"), ModelPricing::new(3.00, 15.00));
    m.insert(("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"), ModelPricing::new(0.25, 1.25));

    // Claude 2 via Bedrock
    m.insert(("bedrock", "anthropic.claude-v2:1"), ModelPricing::new(8.00, 24.00));
    m.insert(("bedrock", "anthropic.claude-v2"), ModelPricing::new(8.00, 24.00));
    m.insert(("bedrock", "anthropic.claude-instant-v1"), ModelPricing::new(0.80, 2.40));

    // Amazon Titan models
    m.insert(("bedrock", "amazon.titan-text-premier-v1:0"), ModelPricing::new(0.50, 1.50));
    m.insert(("bedrock", "amazon.titan-text-express-v1"), ModelPricing::new(0.20, 0.60));
    m.insert(("bedrock", "amazon.titan-text-lite-v1"), ModelPricing::new(0.15, 0.20));

    // Llama 3 via Bedrock
    m.insert(("bedrock", "meta.llama3-70b-instruct-v1:0"), ModelPricing::new(2.65, 3.50));
    m.insert(("bedrock", "meta.llama3-8b-instruct-v1:0"), ModelPricing::new(0.30, 0.60));

    // Mistral via Bedrock
    m.insert(("bedrock", "mistral.mistral-large-2402-v1:0"), ModelPricing::new(4.00, 12.00));
    m.insert(("bedrock", "mistral.mixtral-8x7b-instruct-v0:1"), ModelPricing::new(0.45, 0.70));
    m.insert(("bedrock", "mistral.mistral-7b-instruct-v0:2"), ModelPricing::new(0.15, 0.20));

    m
});

/// Get pricing for a specific provider and model
pub fn get_pricing(provider: Provider, model: &str) -> Option<ModelPricing> {
    let provider_str = match provider {
        Provider::OpenAI => "openai",
        Provider::Anthropic => "anthropic",
        Provider::Gemini => "gemini",
        Provider::Groq => "groq",
        Provider::Bedrock => "bedrock",
    };

    PRICING.get(&(provider_str, model)).copied()
}

/// Get pricing by provider string and model name
pub fn get_pricing_by_str(provider: &str, model: &str) -> Option<ModelPricing> {
    let provider_lower = provider.to_lowercase();
    PRICING.get(&(provider_lower.as_str(), model)).copied()
        .or_else(|| {
            // Try to find a match with the provider string as-is
            PRICING.get(&(provider, model)).copied()
        })
}

/// Get the default model pricing for a provider
pub fn get_default_pricing(provider: Provider) -> Option<ModelPricing> {
    let model = match provider {
        Provider::OpenAI => "gpt-4-turbo",
        Provider::Anthropic => "claude-3-opus-20240229",
        Provider::Gemini => "gemini-1.5-pro",
        Provider::Groq => "llama3-70b-8192",
        Provider::Bedrock => "anthropic.claude-3-haiku-20240307-v1:0",
    };
    get_pricing(provider, model)
}

// ============================================================================
// Token Counting
// ============================================================================

/// Cached tokenizer instances for performance
static CL100K_TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| {
    cl100k_base().expect("Failed to load cl100k_base tokenizer")
});

static O200K_TOKENIZER: Lazy<CoreBPE> = Lazy::new(|| {
    o200k_base().expect("Failed to load o200k_base tokenizer")
});

/// Tokenizer type based on the model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenizerType {
    /// cl100k_base - used by GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100kBase,
    /// o200k_base - used by GPT-4o, o1 models
    O200kBase,
}

/// Determine the tokenizer type for a given model
pub fn get_tokenizer_type(model: &str) -> TokenizerType {
    let model_lower = model.to_lowercase();

    // o200k_base models (GPT-4o family, o1 family)
    if model_lower.contains("gpt-4o")
        || model_lower.starts_with("o1")
        || model_lower.contains("o1-")
    {
        return TokenizerType::O200kBase;
    }

    // Default to cl100k_base for most models
    // This covers: GPT-4, GPT-3.5-turbo, Claude (approximation),
    // Llama (approximation), Gemini (approximation)
    TokenizerType::Cl100kBase
}

/// Count tokens in a text string using the appropriate tokenizer
pub fn count_tokens(text: &str, model: &str) -> usize {
    let tokenizer_type = get_tokenizer_type(model);

    match tokenizer_type {
        TokenizerType::O200kBase => O200K_TOKENIZER.encode_ordinary(text).len(),
        TokenizerType::Cl100kBase => CL100K_TOKENIZER.encode_ordinary(text).len(),
    }
}

/// Count tokens using a specific tokenizer type
pub fn count_tokens_with_tokenizer(text: &str, tokenizer_type: TokenizerType) -> usize {
    match tokenizer_type {
        TokenizerType::O200kBase => O200K_TOKENIZER.encode_ordinary(text).len(),
        TokenizerType::Cl100kBase => CL100K_TOKENIZER.encode_ordinary(text).len(),
    }
}

/// Calculate the input cost for a given text and model
pub fn calculate_input_cost(text: &str, provider: Provider, model: &str) -> Option<f64> {
    let token_count = count_tokens(text, model);
    let pricing = get_pricing(provider, model)?;
    Some(pricing.calculate_input_cost(token_count))
}

/// Calculate the input cost for a given token count and model
pub fn calculate_input_cost_from_tokens(token_count: usize, provider: Provider, model: &str) -> Option<f64> {
    let pricing = get_pricing(provider, model)?;
    Some(pricing.calculate_input_cost(token_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counting() {
        let text = "Hello, world!";
        let tokens = count_tokens(text, "gpt-4");
        assert!(tokens > 0);
        assert!(tokens < 10); // Should be around 3-4 tokens
    }

    #[test]
    fn test_gpt4o_tokenizer() {
        let text = "Hello, world!";
        let tokens_4o = count_tokens(text, "gpt-4o");
        let tokens_4 = count_tokens(text, "gpt-4");
        // Both should produce valid token counts
        assert!(tokens_4o > 0);
        assert!(tokens_4 > 0);
    }

    #[test]
    fn test_pricing_lookup() {
        let pricing = get_pricing(Provider::OpenAI, "gpt-4o");
        assert!(pricing.is_some());
        let p = pricing.unwrap();
        assert!(p.input_cost_per_million > 0.0);
    }

    #[test]
    fn test_cost_calculation() {
        let pricing = ModelPricing::new(10.0, 30.0); // $10 per 1M input, $30 per 1M output

        // 1000 tokens should cost $0.01 for input
        let input_cost = pricing.calculate_input_cost(1000);
        assert!((input_cost - 0.01).abs() < 0.0001);

        // 1000 tokens should cost $0.03 for output
        let output_cost = pricing.calculate_output_cost(1000);
        assert!((output_cost - 0.03).abs() < 0.0001);
    }

    #[test]
    fn test_tokenizer_type_detection() {
        assert_eq!(get_tokenizer_type("gpt-4o"), TokenizerType::O200kBase);
        assert_eq!(get_tokenizer_type("gpt-4o-mini"), TokenizerType::O200kBase);
        assert_eq!(get_tokenizer_type("o1-preview"), TokenizerType::O200kBase);
        assert_eq!(get_tokenizer_type("gpt-4"), TokenizerType::Cl100kBase);
        assert_eq!(get_tokenizer_type("gpt-4-turbo"), TokenizerType::Cl100kBase);
        assert_eq!(get_tokenizer_type("claude-3-opus"), TokenizerType::Cl100kBase);
    }
}
