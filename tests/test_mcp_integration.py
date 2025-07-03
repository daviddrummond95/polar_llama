# pyright: reportMissingImports=false
import polars as pl
from polar_llama import string_to_message, inference_async, inference_messages

def test_provider_column_expression_builds():
    """Ensure that inference_async accepts column expressions for provider/model."""

    df = pl.DataFrame(
        {
            "prompt": [
                "What is the capital of Spain?",
                "Summarise general relativity.",
            ],
            "provider": ["openai", "anthropic"],
            "model": ["gpt-4o-mini", "claude-3-haiku-20240307-v1:0"],
        }
    )

    df = df.with_columns(
        string_to_message(pl.col("prompt"), message_type="user").alias("msg")
    )

    # The call should succeed and return a Polars expression when we supply column
    # expressions for provider/model – this is what MCP relies on for routing.
    expr = inference_async(
        pl.col("msg"),
        provider=pl.col("provider"),
        model=pl.col("model"),
    )

    # Verify that we received an Expr and that we can attach it to the dataframe
    assert isinstance(expr, pl.Expr)

    df2 = df.with_columns(response=expr)
    # We don't collect/execute – that would hit the network. Just ensure the lazy
    # plan can be built without errors.
    assert "response" in df2.columns

# Additional tests for other entry points

def test_provider_column_inference_builds():
    df = pl.DataFrame({
        "prompt": ["hello"],
        "provider": ["groq"],
        "model": ["llama3-8b"],
    })

    expr = inference_async(
        string_to_message(pl.col("prompt"), message_type="user"),
        provider=pl.col("provider"),
    )
    # Should still be an expression
    assert isinstance(expr, pl.Expr)


def test_inference_messages_builds():
    df = pl.DataFrame({
        "conversation": [
            "[{\"role\": \"user\", \"content\": \"Hi!\"}]",
        ],
        "provider": ["openai"],
    })

    expr = inference_messages(
        pl.col("conversation"),
        provider=pl.col("provider"),
    )
    assert isinstance(expr, pl.Expr)