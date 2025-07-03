# MCP Tools Integration Guide

This short guide shows how to combine **Polar Llama** with the *Model-Control-Plane* (MCP) tools while taking advantage of the per-row `provider` / `model` column capability.

> **TL;DR** – Nothing special is required in your code: simply pass a Polars `Expr` (column) to the `provider` argument and MCP will route each row to the correct backend automatically.

---

## 1 Prerequisites

```bash
pip install polars polar-llama mcp-runtime   # fictitious MCP runtime package
```

Make sure your environment variables / secrets for the individual LLM providers are set (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, _etc._).

---

## 2 Quick example

```python
import polars as pl
from polar_llama import string_to_message, inference_async
from mcp_runtime import submit_df  # fictitious helper that executes Polars plans on MCP

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

Happy parallel prompting! 🚀