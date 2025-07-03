# MCP Tools Integration Guide

This short guide shows how to combine **Polar Llama** with the *Model-Control-Plane* (MCP) tools while taking advantage of the per-row `provider` / `model` column capability.

> **TL;DR** – Nothing special is required in your code: simply pass a Polars `Expr` (column) to the `provider` argument and MCP will route each row to the correct backend automatically.

---

## 1 Prerequisites

```bash
pip install polars polar-llama polars-mcp-runtime
```

Make sure your environment variables / secrets for the individual LLM providers are set (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, _etc._).

---

## 2 Quick example

```python
import polars as pl
from polar_llama import string_to_message, inference_async
from polars_mcp_runtime import submit_df  # helper that executes Polars plans on MCP

# A heterogeneous workload: every row can target a different provider/model
workload = pl.DataFrame({
    "prompt": [
        "What is the capital of Spain?",
        "Summarise the theory of general relativity.",
    ],
    "provider": ["openai", "anthropic"],
    "model": ["gpt-4o-mini", "claude-3-haiku-20240307-v1:0"],
})

# Prepare LLM-ready messages
workload = workload.with_columns(
    string_to_message(pl.col("prompt"), message_type="user").alias("msg")
)

# Build the lazy Polars pipeline.  The key bit is that we pass **column
# expressions** for both `provider` and `model`.
plan = workload.lazy().with_columns(
    inference_async(
        pl.col("msg"),
        provider=pl.col("provider"),  # chosen per-row ✅
        model=pl.col("model"),        # chosen per-row ✅
    ).alias("response")
)

# Submit to MCP – it will split the plan by provider, fan-out to the respective
# back-ends and merge everything back together.
result_df = submit_df(plan)
print(result_df.collect())
```

---

## 3 How it works under the hood

1. `inference_async` detects when `provider` or `model` are **expressions** (instead of plain strings) and forwards them *position-ally* to the underlying Rust plugin.
2. MCP analyses the Polars logical plan, partitions the rows by the selector column (`provider`) and executes the sub-plans concurrently against the matching LLM endpoints.
3. The final DataFrame is re-assembled in the original order – no manual orchestration required.

---

## 4 Troubleshooting

• **"Provider X not authenticated"** – double-check your environment variables / secret store for that provider.

• **Runtime says _unknown provider_** – make sure the string values in the `provider` column match one of the recognised back-ends (`openai`, `anthropic`, `gemini`, `groq`, `bedrock`).

• **Need more knobs?** – you can pass additional constant keyword arguments (e.g. `temperature=0.7`) or add extra per-row columns and forward them positionally – the pattern is identical to the `provider` / `model` columns.

---

## 5 End-to-end conversation example (multi-message)

Below we build a tiny multi-turn conversation dataset that mixes providers **per row** and lets MCP execute it in parallel.

```python
import polars as pl
from polar_llama import (
    string_to_message,
    combine_messages,
    inference_messages,
)
from polars_mcp_runtime import submit_df

conversations = [
    {
        "system": "You are a helpful assistant.",
        "user": "Explain quantum entanglement in two sentences.",
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    {
        "system": "You are a grumpy but accurate mathematician.",
        "user": "What is the Riemann hypothesis?",
        "provider": "gemini",
        "model": "gemini-1.5-pro",
    },
]

# Build the initial DataFrame
lazy_conv = pl.DataFrame(conversations).lazy()

# Step 1 – turn plain strings into messages
lazy_conv = lazy_conv.with_columns(
    string_to_message(pl.col("system"), message_type="system").alias("sys_msg"),
    string_to_message(pl.col("user"), message_type="user").alias("user_msg"),
)

# Step 2 – bundle into a conversation and send to the model chosen *per row*
plan = lazy_conv.with_columns(
    inference_messages(
        combine_messages(pl.col("sys_msg"), pl.col("user_msg")),
        provider=pl.col("provider"),
        model=pl.col("model"),
    ).alias("assistant_response")
)

result = submit_df(plan)
print(result.collect())
```

The example demonstrates:
1. Multiple turns (system + user) → `combine_messages`.
2. Per-row provider/model selection.
3. A single `submit_df` call that MCP splits and fans-out behind the scenes.

---

Happy parallel prompting! 🚀