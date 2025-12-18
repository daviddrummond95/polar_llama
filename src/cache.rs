//! Cache analysis and provider-specific cache control injection
//!
//! This module handles:
//! - Analyzing message batches to identify shared prefixes
//! - Grouping rows by cacheable content
//! - Injecting provider-specific cache_control markers

use crate::model_client::{Message, Provider};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Cache strategy determines how rows are grouped for cache optimization
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// No caching - each row processed independently
    None,
    /// Automatically detect shared prefixes (recommended)
    #[default]
    Auto,
    /// Cache system prompts only
    SystemPrompt,
    /// Cache structured output schema instructions
    Schema,
    /// Find and cache longest common prefix (most expensive analysis)
    FullPrefix,
}

impl CacheStrategy {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "none" => Self::None,
            "auto" => Self::Auto,
            "system_prompt" | "system" => Self::SystemPrompt,
            "schema" => Self::Schema,
            "full_prefix" | "full" => Self::FullPrefix,
            _ => Self::Auto,
        }
    }
}

/// Configuration for cache behavior
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub strategy: CacheStrategy,
    pub ttl: String,              // "5m" or "1h" (Anthropic)
    pub cache_key: Option<String>, // OpenAI prompt_cache_key hint
    pub min_tokens: usize,        // Minimum tokens to trigger caching
    pub report_metrics: bool,     // Include cache stats in response
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            strategy: CacheStrategy::Auto,
            ttl: "5m".to_string(),
            cache_key: None,
            min_tokens: 1024,
            report_metrics: true,
        }
    }
}

/// A group of rows that share the same cacheable prefix
#[derive(Debug, Clone)]
pub struct CacheGroup {
    /// Hash of the shared prefix for identification
    pub prefix_hash: String,
    /// Original row indices belonging to this group
    pub row_indices: Vec<usize>,
    /// The messages that are identical across the group
    pub shared_prefix: Vec<Message>,
    /// Index in message array where cache_control should be placed
    pub cache_breakpoint_idx: usize,
    /// Estimated token count of the shared prefix
    pub estimated_prefix_tokens: usize,
}

/// Cache control configuration for Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String, // "ephemeral" or "ephemeral_1h"
}

/// Analyze a batch of message arrays to identify optimal cache groups
///
/// # Arguments
/// * `message_arrays` - All message arrays from the DataFrame batch
/// * `strategy` - How aggressively to group for caching
/// * `min_tokens` - Minimum prefix tokens to consider for caching
///
/// # Returns
/// Vector of CacheGroups, each containing rows that share a prefix
pub fn analyze_batch_for_caching(
    message_arrays: &[Vec<Message>],
    strategy: CacheStrategy,
    min_tokens: usize,
) -> Vec<CacheGroup> {
    match strategy {
        CacheStrategy::None => {
            // No grouping - each row is its own "group"
            message_arrays
                .iter()
                .enumerate()
                .map(|(idx, _msgs)| CacheGroup {
                    prefix_hash: format!("row_{}", idx),
                    row_indices: vec![idx],
                    shared_prefix: vec![],
                    cache_breakpoint_idx: 0,
                    estimated_prefix_tokens: 0,
                })
                .collect()
        }

        CacheStrategy::Auto | CacheStrategy::SystemPrompt => {
            group_by_system_prompt(message_arrays, min_tokens)
        }

        CacheStrategy::Schema => {
            // When using structured outputs, all rows share the schema
            // The schema instructions are injected uniformly
            vec![CacheGroup {
                prefix_hash: "schema_group".to_string(),
                row_indices: (0..message_arrays.len()).collect(),
                shared_prefix: vec![],
                cache_breakpoint_idx: 0,
                estimated_prefix_tokens: 0, // Schema tokens counted separately
            }]
        }

        CacheStrategy::FullPrefix => {
            // For v1, fall back to system prompt grouping
            // Future: implement trie-based LCP detection
            group_by_system_prompt(message_arrays, min_tokens)
        }
    }
}

/// Group rows by identical system prompt content
fn group_by_system_prompt(message_arrays: &[Vec<Message>], min_tokens: usize) -> Vec<CacheGroup> {
    let mut groups: HashMap<String, CacheGroup> = HashMap::new();

    for (idx, messages) in message_arrays.iter().enumerate() {
        // Extract contiguous system messages from the start
        let system_messages: Vec<&Message> =
            messages.iter().take_while(|m| m.role == "system").collect();

        // Estimate token count (rough: ~4 chars per token)
        let estimated_tokens: usize = system_messages.iter().map(|m| m.content.len() / 4).sum();

        // Only group if prefix meets minimum token threshold
        if estimated_tokens < min_tokens && !system_messages.is_empty() {
            // Below threshold - treat as individual row
            groups.insert(
                format!("individual_{}", idx),
                CacheGroup {
                    prefix_hash: format!("individual_{}", idx),
                    row_indices: vec![idx],
                    shared_prefix: vec![],
                    cache_breakpoint_idx: 0,
                    estimated_prefix_tokens: 0,
                },
            );
            continue;
        }

        // Hash the system prompt content for grouping
        let prefix_hash = hash_messages(&system_messages);

        groups
            .entry(prefix_hash.clone())
            .and_modify(|g| g.row_indices.push(idx))
            .or_insert_with(|| CacheGroup {
                prefix_hash,
                row_indices: vec![idx],
                shared_prefix: system_messages.iter().map(|m| (*m).clone()).collect(),
                cache_breakpoint_idx: system_messages.len().saturating_sub(1),
                estimated_prefix_tokens: estimated_tokens,
            });
    }

    groups.into_values().collect()
}

/// Create a hash of message content for grouping
fn hash_messages(messages: &[&Message]) -> String {
    let mut hasher = Sha256::new();
    for msg in messages {
        hasher.update(msg.role.as_bytes());
        hasher.update(b":");
        hasher.update(msg.content.as_bytes());
        hasher.update(b"|");
    }
    // Use first 16 hex chars for reasonable uniqueness
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Inject cache_control markers based on provider requirements
///
/// # Arguments
/// * `messages` - The message array to modify
/// * `provider` - Target provider (determines cache control format)
/// * `config` - Cache configuration
/// * `breakpoint_idx` - Index where cache_control should be placed
///
/// # Returns
/// Modified messages with cache_control markers where appropriate
pub fn prepare_messages_for_caching(
    mut messages: Vec<Message>,
    provider: Provider,
    config: &CacheConfig,
    breakpoint_idx: usize,
) -> Vec<Message> {
    match provider {
        Provider::Anthropic | Provider::Bedrock => {
            // Anthropic/Bedrock require explicit cache_control markers
            inject_cache_control_marker(&mut messages, breakpoint_idx, &config.ttl);
        }
        Provider::OpenAI => {
            // OpenAI caching is automatic based on prefix matching
            // The cache_key hint is passed at request level, not message level
        }
        Provider::Gemini => {
            // Gemini explicit caching is handled separately via cached_content API
            // Implicit caching (2.5 models) is automatic
        }
        Provider::Groq => {
            // Groq caching is fully automatic, no markers needed
        }
    }

    messages
}

/// Inject cache_control marker at the specified breakpoint
fn inject_cache_control_marker(messages: &mut [Message], breakpoint_idx: usize, ttl: &str) {
    if let Some(msg) = messages.get_mut(breakpoint_idx) {
        msg.cache_control = Some(CacheControl {
            cache_type: match ttl {
                "1h" | "1hour" | "60m" => "ephemeral_1h".to_string(),
                _ => "ephemeral".to_string(),
            },
        });
    }
}

/// Aggregate cache metrics from a batch of responses
#[derive(Debug, Default, Clone)]
pub struct CacheMetrics {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_writes: usize,
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub cache_write_tokens: usize,
    pub cache_read_tokens: usize,
}

impl CacheMetrics {
    /// Calculate estimated cost savings from caching
    pub fn estimated_savings(&self, input_price_per_million: f64) -> f64 {
        // Anthropic: cache reads are 90% cheaper
        // OpenAI: cache reads are 50% cheaper
        let tokens_saved = self.cache_read_tokens as f64;
        let savings_rate = 0.9; // Conservative: use Anthropic's 90%
        tokens_saved * input_price_per_million * savings_rate / 1_000_000.0
    }

    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }
}

/// Usage metrics from inference response (extended for cache support)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageMetrics {
    /// Total input tokens (non-cached portion for Anthropic)
    #[serde(default)]
    pub input_tokens: Option<u64>,

    /// Output tokens generated
    #[serde(default)]
    pub output_tokens: Option<u64>,

    /// Cached tokens (OpenAI)
    #[serde(default)]
    pub cached_tokens: Option<u64>,

    /// Tokens written to cache (Anthropic) - costs 1.25x
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,

    /// Tokens read from cache (Anthropic) - costs 0.1x
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_strategy_from_str() {
        assert_eq!(CacheStrategy::from_str("none"), CacheStrategy::None);
        assert_eq!(CacheStrategy::from_str("auto"), CacheStrategy::Auto);
        assert_eq!(
            CacheStrategy::from_str("system_prompt"),
            CacheStrategy::SystemPrompt
        );
        assert_eq!(CacheStrategy::from_str("schema"), CacheStrategy::Schema);
        assert_eq!(
            CacheStrategy::from_str("full_prefix"),
            CacheStrategy::FullPrefix
        );
        assert_eq!(CacheStrategy::from_str("unknown"), CacheStrategy::Auto);
    }

    #[test]
    fn test_hash_messages() {
        let msg1 = Message {
            role: "system".to_string(),
            content: "You are a classifier".to_string(),
            cache_control: None,
        };
        let msg2 = Message {
            role: "system".to_string(),
            content: "You are a classifier".to_string(),
            cache_control: None,
        };
        let msg3 = Message {
            role: "system".to_string(),
            content: "You are a different classifier".to_string(),
            cache_control: None,
        };

        let hash1 = hash_messages(&[&msg1]);
        let hash2 = hash_messages(&[&msg2]);
        let hash3 = hash_messages(&[&msg3]);

        assert_eq!(hash1, hash2); // Same content should produce same hash
        assert_ne!(hash1, hash3); // Different content should produce different hash
    }

    #[test]
    fn test_analyze_batch_no_caching() {
        let messages = vec![
            vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                cache_control: None,
            }],
            vec![Message {
                role: "user".to_string(),
                content: "World".to_string(),
                cache_control: None,
            }],
        ];

        let groups = analyze_batch_for_caching(&messages, CacheStrategy::None, 1024);

        assert_eq!(groups.len(), 2); // Each row in its own group
    }

    #[test]
    fn test_analyze_batch_with_shared_system_prompt() {
        let system_content = "You are a document classifier. ".repeat(100); // ~2500 chars -> ~600 tokens
        let messages = vec![
            vec![
                Message {
                    role: "system".to_string(),
                    content: system_content.clone(),
                    cache_control: None,
                },
                Message {
                    role: "user".to_string(),
                    content: "Document A".to_string(),
                    cache_control: None,
                },
            ],
            vec![
                Message {
                    role: "system".to_string(),
                    content: system_content.clone(),
                    cache_control: None,
                },
                Message {
                    role: "user".to_string(),
                    content: "Document B".to_string(),
                    cache_control: None,
                },
            ],
            vec![
                Message {
                    role: "system".to_string(),
                    content: system_content.clone(),
                    cache_control: None,
                },
                Message {
                    role: "user".to_string(),
                    content: "Document C".to_string(),
                    cache_control: None,
                },
            ],
        ];

        let groups = analyze_batch_for_caching(&messages, CacheStrategy::Auto, 100);

        // All rows should be in one group since they share the system prompt
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].row_indices.len(), 3);
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = CacheMetrics {
            total_requests: 100,
            cache_hits: 90,
            cache_misses: 10,
            cache_writes: 10,
            input_tokens: 100000,
            cached_tokens: 90000,
            cache_write_tokens: 10000,
            cache_read_tokens: 90000,
        };

        assert_eq!(metrics.cache_hit_rate(), 0.9);

        // $3/M input tokens, 90% savings on cached tokens
        let savings = metrics.estimated_savings(3.0);
        assert!(savings > 0.0);
    }

    #[test]
    fn test_inject_cache_control_marker() {
        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: "System prompt".to_string(),
                cache_control: None,
            },
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                cache_control: None,
            },
        ];

        inject_cache_control_marker(&mut messages, 0, "5m");

        assert!(messages[0].cache_control.is_some());
        assert_eq!(
            messages[0].cache_control.as_ref().unwrap().cache_type,
            "ephemeral"
        );

        // Test 1h TTL
        let mut messages2 = vec![Message {
            role: "system".to_string(),
            content: "System prompt".to_string(),
            cache_control: None,
        }];

        inject_cache_control_marker(&mut messages2, 0, "1h");

        assert!(messages2[0].cache_control.is_some());
        assert_eq!(
            messages2[0].cache_control.as_ref().unwrap().cache_type,
            "ephemeral_1h"
        );
    }
}
