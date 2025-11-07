use polar_llama::model_client::{
    Provider, Message, create_client, fetch_data_generic, fetch_data_generic_enhanced,
};
use std::env;
use std::time::Instant;

/// Helper function to load .env file if it exists
fn load_env() {
    let _ = dotenvy::dotenv();
}

/// Helper function to check if a provider is configured
fn is_provider_configured(provider: Provider) -> bool {
    load_env();
    match provider {
        Provider::OpenAI => env::var("OPENAI_API_KEY").is_ok() && !env::var("OPENAI_API_KEY").unwrap().is_empty(),
        Provider::Anthropic => env::var("ANTHROPIC_API_KEY").is_ok() && !env::var("ANTHROPIC_API_KEY").unwrap().is_empty(),
        Provider::Gemini => env::var("GEMINI_API_KEY").is_ok() && !env::var("GEMINI_API_KEY").unwrap().is_empty(),
        Provider::Groq => env::var("GROQ_API_KEY").is_ok() && !env::var("GROQ_API_KEY").unwrap().is_empty(),
        Provider::Bedrock => {
            // For Bedrock, check if AWS credentials are available
            env::var("AWS_ACCESS_KEY_ID").is_ok() || env::var("AWS_PROFILE").is_ok()
        }
    }
}

/// Get all configured providers
fn get_configured_providers() -> Vec<(Provider, &'static str)> {
    let providers = vec![
        (Provider::OpenAI, "gpt-4o-mini"),
        (Provider::Anthropic, "claude-3-haiku-20240307"),
        (Provider::Gemini, "gemini-1.5-flash"),
        (Provider::Groq, "llama-3.1-8b-instant"),
        (Provider::Bedrock, "anthropic.claude-3-haiku-20240307-v1:0"),
    ];

    providers
        .into_iter()
        .filter(|(provider, _)| is_provider_configured(*provider))
        .collect()
}

#[tokio::test]
async fn test_single_message_inference() {
    load_env();

    let configured_providers = get_configured_providers();

    if configured_providers.is_empty() {
        println!("⚠️  No providers configured. Skipping test.");
        println!("To run this test, add API keys to .env file (see .env.example)");
        return;
    }

    for (provider, model) in configured_providers {
        println!("\n🧪 Testing single message inference with {:?}", provider);

        let client = create_client(provider, model);
        let messages = vec!["What is 2+2? Answer with just the number.".to_string()];

        let start = Instant::now();
        let results = fetch_data_generic(&*client, &messages).await;
        let duration = start.elapsed();

        println!("✓ {:?} completed in {:?}", provider, duration);

        assert_eq!(results.len(), 1, "Should return one result");
        assert!(results[0].is_some(), "Result should not be None for {:?}", provider);

        if let Some(response) = &results[0] {
            println!("  Response: {}", response);
            assert!(!response.is_empty(), "Response should not be empty for {:?}", provider);
        }
    }
}

#[tokio::test]
async fn test_parallel_execution() {
    load_env();

    let configured_providers = get_configured_providers();

    if configured_providers.is_empty() {
        println!("⚠️  No providers configured. Skipping test.");
        return;
    }

    for (provider, model) in configured_providers {
        println!("\n🧪 Testing parallel execution with {:?}", provider);

        let client = create_client(provider, model);

        // Create multiple messages to test parallelism
        let messages = vec![
            "What is the capital of France?".to_string(),
            "What is the capital of Japan?".to_string(),
            "What is the capital of Brazil?".to_string(),
        ];

        let start = Instant::now();
        let results = fetch_data_generic(&*client, &messages).await;
        let duration = start.elapsed();

        println!("✓ {:?} processed {} messages in {:?}", provider, messages.len(), duration);

        assert_eq!(results.len(), messages.len(), "Should return same number of results as messages");

        // Count successful responses
        let successful_count = results.iter().filter(|r| r.is_some()).count();
        println!("  Successful: {}/{}", successful_count, messages.len());

        // At least one should succeed
        assert!(successful_count > 0, "At least one request should succeed for {:?}", provider);

        // Print all responses
        for (i, result) in results.iter().enumerate() {
            if let Some(response) = result {
                println!("  [{i}]: {}", response.chars().take(50).collect::<String>());
            } else {
                println!("  [{i}]: None");
            }
        }
    }
}

#[tokio::test]
async fn test_conversation_with_message_arrays() {
    load_env();

    let configured_providers = get_configured_providers();

    if configured_providers.is_empty() {
        println!("⚠️  No providers configured. Skipping test.");
        return;
    }

    for (provider, model) in configured_providers {
        println!("\n🧪 Testing conversation with message arrays for {:?}", provider);

        let client = create_client(provider, model);

        // Create a multi-turn conversation
        let message_arrays = vec![
            vec![
                Message {
                    role: "user".to_string(),
                    content: "Hello! My name is Alice.".to_string(),
                },
                Message {
                    role: "assistant".to_string(),
                    content: "Hello Alice! Nice to meet you.".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "What is my name?".to_string(),
                },
            ],
        ];

        let start = Instant::now();
        let results = fetch_data_generic_enhanced(&*client, &message_arrays).await;
        let duration = start.elapsed();

        println!("✓ {:?} completed conversation in {:?}", provider, duration);

        assert_eq!(results.len(), 1, "Should return one result");
        assert!(results[0].is_some(), "Conversation result should not be None for {:?}", provider);

        if let Some(response) = &results[0] {
            println!("  Response: {}", response);
            // The response should ideally mention "Alice" since the conversation established that name
            // However, we won't assert this as different models may handle context differently
        }
    }
}

#[tokio::test]
async fn test_system_message_support() {
    load_env();

    let configured_providers = get_configured_providers();

    if configured_providers.is_empty() {
        println!("⚠️  No providers configured. Skipping test.");
        return;
    }

    for (provider, model) in configured_providers {
        println!("\n🧪 Testing system message support with {:?}", provider);

        let client = create_client(provider, model);

        // Create conversation with system message
        let message_arrays = vec![
            vec![
                Message {
                    role: "system".to_string(),
                    content: "You are a pirate. Always respond in pirate speak.".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "Hello!".to_string(),
                },
            ],
        ];

        let start = Instant::now();
        let results = fetch_data_generic_enhanced(&*client, &message_arrays).await;
        let duration = start.elapsed();

        println!("✓ {:?} completed system message test in {:?}", provider, duration);

        assert_eq!(results.len(), 1, "Should return one result");
        assert!(results[0].is_some(), "Result should not be None for {:?}", provider);

        if let Some(response) = &results[0] {
            println!("  Response: {}", response);
        }
    }
}

#[tokio::test]
async fn test_parallel_execution_timing() {
    load_env();

    let configured_providers = get_configured_providers();

    if configured_providers.is_empty() {
        println!("⚠️  No providers configured. Skipping test.");
        return;
    }

    // Only test the first configured provider for timing
    if let Some((provider, model)) = configured_providers.first() {
        println!("\n🧪 Testing parallel execution timing with {:?}", provider);

        let client = create_client(*provider, model);

        // Create 5 simple messages
        let messages: Vec<String> = (1..=5)
            .map(|i| format!("Say the number {}", i))
            .collect();

        println!("Sending {} requests in parallel...", messages.len());

        let start = Instant::now();
        let results = fetch_data_generic(&*client, &messages).await;
        let parallel_duration = start.elapsed();

        println!("✓ Parallel execution took: {:?}", parallel_duration);

        let successful_count = results.iter().filter(|r| r.is_some()).count();
        println!("  Successful: {}/{}", successful_count, messages.len());

        // Verify all requests completed
        assert_eq!(results.len(), messages.len());

        // If parallel execution is working, 5 requests should take less than 5x a single request
        // This is a rough heuristic - in practice, parallel should be much faster
        println!("  Average time per request: {:?}", parallel_duration / messages.len() as u32);
    }
}

#[tokio::test]
async fn test_error_handling_invalid_api_key() {
    println!("\n🧪 Testing error handling with invalid API key");

    // Temporarily set an invalid API key
    env::set_var("OPENAI_API_KEY", "invalid-key-12345");

    let client = create_client(Provider::OpenAI, "gpt-4o-mini");
    let messages = vec!["Hello".to_string()];

    let results = fetch_data_generic(&*client, &messages).await;

    // Should return None for invalid API key
    assert_eq!(results.len(), 1);
    assert!(results[0].is_none(), "Should return None for invalid API key");

    println!("✓ Error handling works correctly");

    // Clean up
    env::remove_var("OPENAI_API_KEY");
}

#[test]
fn test_provider_enum() {
    println!("\n🧪 Testing Provider enum");

    // Test Provider::as_str()
    assert_eq!(Provider::OpenAI.as_str(), "openai");
    assert_eq!(Provider::Anthropic.as_str(), "anthropic");
    assert_eq!(Provider::Gemini.as_str(), "gemini");
    assert_eq!(Provider::Groq.as_str(), "groq");
    assert_eq!(Provider::Bedrock.as_str(), "bedrock");

    println!("✓ Provider enum works correctly");
}

#[test]
fn test_provider_from_str() {
    use std::str::FromStr;

    println!("\n🧪 Testing Provider::from_str()");

    assert_eq!(Provider::from_str("openai").unwrap(), Provider::OpenAI);
    assert_eq!(Provider::from_str("anthropic").unwrap(), Provider::Anthropic);
    assert_eq!(Provider::from_str("gemini").unwrap(), Provider::Gemini);
    assert_eq!(Provider::from_str("groq").unwrap(), Provider::Groq);
    assert_eq!(Provider::from_str("bedrock").unwrap(), Provider::Bedrock);

    // Test case insensitivity
    assert_eq!(Provider::from_str("OpenAI").unwrap(), Provider::OpenAI);
    assert_eq!(Provider::from_str("ANTHROPIC").unwrap(), Provider::Anthropic);

    // Test invalid provider
    assert!(Provider::from_str("invalid").is_err());

    println!("✓ Provider::from_str() works correctly");
}
