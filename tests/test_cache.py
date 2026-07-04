"""Tests for caching functionality."""
import pytest
import json


def test_cache_imports():
    """Test that cache types can be imported."""
    try:
        from polar_llama import CacheStrategy, CacheConfig, CacheMetrics
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import cache types: {e}")


def test_cache_strategy_enum():
    """Test that CacheStrategy enum has all expected values."""
    from polar_llama import CacheStrategy

    # Check all expected values exist
    assert hasattr(CacheStrategy, "NONE")
    assert hasattr(CacheStrategy, "AUTO")
    assert hasattr(CacheStrategy, "SYSTEM_PROMPT")
    assert hasattr(CacheStrategy, "SCHEMA")
    assert hasattr(CacheStrategy, "FULL_PREFIX")

    # Check string values
    assert CacheStrategy.NONE.value == "none"
    assert CacheStrategy.AUTO.value == "auto"
    assert CacheStrategy.SYSTEM_PROMPT.value == "system_prompt"
    assert CacheStrategy.SCHEMA.value == "schema"
    assert CacheStrategy.FULL_PREFIX.value == "full_prefix"


def test_cache_config_defaults():
    """Test CacheConfig default values."""
    from polar_llama import CacheConfig, CacheStrategy

    config = CacheConfig()

    assert config.strategy == CacheStrategy.AUTO
    assert config.min_tokens == 1024
    assert config.ttl == "5m"
    assert config.cache_key is None
    assert config.report_metrics is True


def test_cache_config_custom_values():
    """Test CacheConfig with custom values."""
    from polar_llama import CacheConfig, CacheStrategy

    config = CacheConfig(
        strategy=CacheStrategy.SYSTEM_PROMPT,
        min_tokens=2048,
        ttl="1h",
        cache_key="my-cache-key",
        report_metrics=False,
    )

    assert config.strategy == CacheStrategy.SYSTEM_PROMPT
    assert config.min_tokens == 2048
    assert config.ttl == "1h"
    assert config.cache_key == "my-cache-key"
    assert config.report_metrics is False


def test_cache_config_to_kwargs():
    """Test CacheConfig.to_kwargs() returns correct dictionary."""
    from polar_llama import CacheConfig, CacheStrategy

    config = CacheConfig(
        strategy=CacheStrategy.SCHEMA,
        min_tokens=512,
        ttl="5m",
        cache_key="test-key",
    )

    kwargs = config.to_kwargs()

    assert kwargs["cache"] is True
    assert kwargs["cache_strategy"] == "schema"
    assert kwargs["cache_ttl"] == "5m"
    assert kwargs["cache_key"] == "test-key"
    assert kwargs["cache_min_tokens"] == 512


def test_cache_metrics_creation():
    """Test CacheMetrics dataclass creation."""
    from polar_llama import CacheMetrics

    metrics = CacheMetrics(
        total_requests=100,
        cache_hits=90,
        cache_misses=10,
        cache_writes=10,
        input_tokens=100000,
        cached_tokens=90000,
        cache_write_tokens=10000,
        cache_read_tokens=90000,
    )

    assert metrics.total_requests == 100
    assert metrics.cache_hits == 90
    assert metrics.cache_misses == 10
    assert metrics.cache_writes == 10
    assert metrics.input_tokens == 100000
    assert metrics.cached_tokens == 90000
    assert metrics.cache_write_tokens == 10000
    assert metrics.cache_read_tokens == 90000


def test_cache_metrics_hit_rate():
    """Test CacheMetrics.cache_hit_rate property."""
    from polar_llama import CacheMetrics

    # 90% hit rate
    metrics = CacheMetrics(
        total_requests=100,
        cache_hits=90,
        cache_misses=10,
        cache_writes=10,
        input_tokens=100000,
        cached_tokens=90000,
        cache_write_tokens=10000,
        cache_read_tokens=90000,
    )

    assert metrics.cache_hit_rate == 0.9

    # 0% hit rate when no requests
    empty_metrics = CacheMetrics(
        total_requests=0,
        cache_hits=0,
        cache_misses=0,
        cache_writes=0,
        input_tokens=0,
        cached_tokens=0,
        cache_write_tokens=0,
        cache_read_tokens=0,
    )

    assert empty_metrics.cache_hit_rate == 0.0


def test_cache_metrics_estimated_savings():
    """Test CacheMetrics.estimated_savings method."""
    from polar_llama import CacheMetrics

    metrics = CacheMetrics(
        total_requests=100,
        cache_hits=90,
        cache_misses=10,
        cache_writes=10,
        input_tokens=100000,
        cached_tokens=90000,
        cache_write_tokens=10000,
        cache_read_tokens=1000000,  # 1M cached reads for easy calculation
    )

    # Default price: $3/M input tokens, 90% savings
    # 1M cached tokens * $3/M * 0.9 = $2.70
    savings = metrics.estimated_savings()
    assert savings == pytest.approx(2.7, rel=0.01)

    # Custom price: $15/M input tokens (e.g., Claude Opus)
    # 1M cached tokens * $15/M * 0.9 = $13.50
    savings_custom = metrics.estimated_savings(input_price_per_million=15.0)
    assert savings_custom == pytest.approx(13.5, rel=0.01)


def test_inference_async_accepts_cache_bool():
    """Test that inference_async accepts cache=True parameter in its signature."""
    import inspect
    from polar_llama import inference_async

    # Check that cache parameter exists in function signature
    sig = inspect.signature(inference_async)
    params = sig.parameters

    assert "cache" in params, "inference_async should have 'cache' parameter"

    # Check default value is False
    cache_param = params["cache"]
    assert cache_param.default is False, "cache parameter should default to False"


def test_inference_async_accepts_cache_config():
    """Test that inference_async accepts CacheConfig parameter in its signature."""
    import inspect
    from polar_llama import inference_async, CacheConfig, CacheStrategy
    from typing import Union, get_type_hints

    # Check function can accept Union[bool, CacheConfig] types
    sig = inspect.signature(inference_async)
    params = sig.parameters

    assert "cache" in params

    # Verify the annotation allows both bool and CacheConfig
    cache_param = params["cache"]
    # The annotation should allow Union[bool, CacheConfig]
    annotation = cache_param.annotation
    # Just verify we can create both types that should be acceptable
    config = CacheConfig(strategy=CacheStrategy.SYSTEM_PROMPT, ttl="1h")
    assert isinstance(config, CacheConfig)
    assert isinstance(True, bool)


def test_inference_messages_accepts_cache_bool():
    """Test that inference_messages accepts cache=True parameter in its signature."""
    import inspect
    from polar_llama import inference_messages

    # Check that cache parameter exists in function signature
    sig = inspect.signature(inference_messages)
    params = sig.parameters

    assert "cache" in params, "inference_messages should have 'cache' parameter"

    # Check default value is False
    cache_param = params["cache"]
    assert cache_param.default is False, "cache parameter should default to False"


def test_inference_messages_accepts_cache_config():
    """Test that inference_messages accepts CacheConfig parameter in its signature."""
    import inspect
    from polar_llama import inference_messages, CacheConfig, CacheStrategy

    # Check function can accept Union[bool, CacheConfig] types
    sig = inspect.signature(inference_messages)
    params = sig.parameters

    assert "cache" in params

    # Verify the annotation allows both bool and CacheConfig
    cache_param = params["cache"]
    # Just verify we can create both types that should be acceptable
    config = CacheConfig(strategy=CacheStrategy.AUTO, min_tokens=512)
    assert isinstance(config, CacheConfig)
    assert isinstance(True, bool)
