"""
Type definitions for Polar Llama caching functionality.
"""
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class CacheStrategy(str, Enum):
    """Cache optimization strategy.

    Determines how rows are grouped for cache optimization when processing
    DataFrames with many rows.
    """
    NONE = "none"                    # No caching - each row processed independently
    AUTO = "auto"                    # Detect shared content automatically (recommended)
    SYSTEM_PROMPT = "system_prompt"  # Cache system prompt only
    SCHEMA = "schema"                # Cache structured output schema instructions
    FULL_PREFIX = "full_prefix"      # Cache everything before user message


@dataclass
class CacheConfig:
    """Configuration for cache behavior.

    Use this class to fine-tune caching behavior when calling inference functions.

    Attributes:
        strategy: How aggressively to group rows for caching.
        min_tokens: Minimum tokens in a shared prefix to trigger caching.
        ttl: Cache time-to-live for Anthropic ("5m" or "1h").
        cache_key: Optional cache key hint for OpenAI routing.
        report_metrics: Whether to include cache stats in response.

    Example:
        >>> from polar_llama import CacheConfig, CacheStrategy
        >>> config = CacheConfig(
        ...     strategy=CacheStrategy.AUTO,
        ...     min_tokens=1024,
        ...     ttl="5m"
        ... )
    """
    strategy: CacheStrategy = CacheStrategy.AUTO
    min_tokens: int = 1024
    ttl: Literal["5m", "1h"] = "5m"
    cache_key: Optional[str] = None
    report_metrics: bool = True

    def to_kwargs(self) -> dict:
        """Convert to kwargs dict for Rust FFI."""
        return {
            "cache": True,
            "cache_strategy": self.strategy.value,
            "cache_ttl": self.ttl,
            "cache_key": self.cache_key,
            "cache_min_tokens": self.min_tokens,
        }


@dataclass
class CacheMetrics:
    """Aggregated cache performance metrics.

    Returned after a batch operation to show how effective caching was.

    Attributes:
        total_requests: Total number of API requests made.
        cache_hits: Number of requests that hit the cache.
        cache_misses: Number of requests that missed the cache.
        cache_writes: Number of cache write operations.
        input_tokens: Total input tokens processed.
        cached_tokens: Total tokens served from cache.
        cache_write_tokens: Tokens written to cache (Anthropic).
        cache_read_tokens: Tokens read from cache (Anthropic).
    """
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_writes: int
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int

    @property
    def cache_hit_rate(self) -> float:
        """Calculate the cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def estimated_savings(self, input_price_per_million: float = 3.0) -> float:
        """Estimate USD savings from cache hits.

        Args:
            input_price_per_million: Price per million input tokens (default: $3 for Claude Sonnet)

        Returns:
            Estimated USD savings from cache hits (assumes 90% discount on cached tokens).
        """
        # Anthropic: cache reads are 90% cheaper
        return self.cache_read_tokens * input_price_per_million * 0.9 / 1_000_000
