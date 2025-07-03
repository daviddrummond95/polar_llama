import polars as pl
from polar_llama import string_to_message, combine_messages, inference_messages
from polars_mcp_runtime import submit_df


def test_full_mcp_flow_collects():
    """End-to-end test that builds a lazy plan and submits it via the MCP stub."""

    data = {
        "system": [
            "You are a helpful assistant.",
            "You are a theorem prover.",
        ],
        "user": [
            "Summarise quantum tunnelling.",
            "State the Pythagorean theorem.",
        ],
        "provider": ["openai", "anthropic"],
        "model": ["gpt-4o-mini", "claude-3-haiku-20240307-v1:0"],
    }

    lf = pl.DataFrame(data).lazy()

    lf = lf.with_columns(
        string_to_message(pl.col("system"), message_type="system").alias("sys_msg"),
        string_to_message(pl.col("user"), message_type="user").alias("user_msg"),
    )

    plan = lf.with_columns(
        inference_messages(
            combine_messages(pl.col("sys_msg"), pl.col("user_msg")),
            provider=pl.col("provider"),
            model=pl.col("model"),
        ).alias("assistant")
    )

    df = submit_df(plan)

    # The stub's submit_df collects the plan, so we should get a DataFrame with
    # the assistant column (contents are dummy strings because the expression
    # isn't executed in the stub environment).
    assert isinstance(df, pl.DataFrame)
    assert "assistant" in df.columns
    assert df.height == 2