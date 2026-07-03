# Tool Use and MCP Integration

Polar Llama supports tool use in a dataframe-native way. Instead of running
an opaque agent loop inside each row, the loop is **unrolled into the
dataframe**: one "turn" is one `with_columns` pass, and every intermediate is
an ordinary column you can inspect, explode, filter, cache, and resume.

```
row text ──► emit calls ──► execute (parallel) ──► synthesize ──► answer
 (col)         (col)             (col)                (col)
```

The rationale and trade-offs are documented in
[docs/design/MCP_TOOL_INTEGRATION.md](design/MCP_TOOL_INTEGRATION.md).

## The three primitives

| Function | What it does |
|---|---|
| `tools_to_response_model(tools)` | Builds a Pydantic emission schema from tool definitions — the LLM *emits* tool calls as ordinary structured output; nothing is executed. |
| `execute_tool_calls(expr, transport=... \| executor=...)` | Executes a column of emitted calls — every call of every row in parallel. Failures are data (`is_error`, `_error`), not exceptions. |
| `tool_results_to_message(expr)` | Renders a results column as a message for the synthesis turn (`combine_messages` + `inference_messages`). |

Plus one introspection helper:

| Function | What it does |
|---|---|
| `mcp_tools(transport)` | Fetches tool definitions from an MCP server (`tools/list`). Supports streamable HTTP (`"http://host:port/mcp"`) and stdio (`"stdio:my-server --flag"`). |

MCP's role is deliberately narrow: it is a **schema source and a call target**
(`tools/list` + `tools/call`) — a registry, not a runtime. Sessions, sampling,
and per-row server state are out of scope by design.

## Quick start

```python
import json
import polars as pl
from polar_llama import (
    Provider, mcp_tools, tools_to_response_model,
    execute_tool_calls, tool_results_to_message,
    combine_messages, inference_messages,
)

# 1. Get tool definitions — from an MCP server, or plain dicts / Pydantic models
tools = mcp_tools("http://localhost:8811/mcp")

# 2. Emission: the LLM parameterizes zero or more calls per row.
#    This is ordinary structured output — nothing is executed.
ToolCalls = tools_to_response_model(tools)

df = pl.DataFrame({"meal": [
    "two eggs, sourdough toast with butter, black coffee",
    "chipotle chicken bowl, no rice, extra guac",
]})

df = df.with_columns(
    calls=pl.col("meal").llama.inference_async(
        provider=Provider.OPENAI, model="gpt-4o-mini",
        response_model=ToolCalls,
    )
)

# 3. Execution: all calls across all rows run concurrently on the Rust
#    async runtime. Arguments are validated against each tool's own input
#    schema before any network call; invalid calls fail fast as data.
df = df.with_columns(
    results=execute_tool_calls(
        pl.col("calls"),
        transport="http://localhost:8811/mcp",
        tools=tools,           # optional: enables pre-execution validation
        concurrency=64,
        timeout_s=30,
    )
)

# 4. Synthesis: fold results back through a second inference pass.
df = df.with_columns(
    answer=inference_messages(
        combine_messages(
            pl.col("meal").llama.to_message(role="user"),
            tool_results_to_message(pl.col("results")),
        ),
        provider=Provider.OPENAI, model="gpt-4o-mini",
    )
)
```

Want another round of tool calls? Repeat steps 2–3. That repetition *is* the
agent loop — visible, bounded by construction, and resumable from any column.

## Tool definitions

`tools_to_response_model`, `execute_tool_calls(tools=...)`, and `mcp_tools`
all speak the same normalized form: `{name, description, input_schema}`.
Accepted inputs:

- MCP `tools/list` entries (`inputSchema`) — including the output of `mcp_tools`
- Anthropic-style dicts (`input_schema`)
- OpenAI function-style dicts (`parameters`, with or without the
  `{"type": "function", "function": {...}}` wrapper)
- Pydantic `BaseModel` subclasses (class name = tool name, docstring =
  description, model schema = input schema)

## The emission format

The generated response model is the strict-mode-safe tagged union:

```json
{"calls": [{"tool_name": "search_food_db",
            "arguments": "{\"query\": \"eggs\", \"database\": \"usda\"}"}]}
```

`tool_name` is constrained to an enum of your tool names. `arguments` is a
JSON-encoded string because strict structured-output modes handle it reliably
where open `anyOf` unions are not; each tool's input schema is embedded in the
field description so the model generates against the tool's real contract,
and `execute_tool_calls(tools=...)` re-validates every call against the
selected tool's schema **before** it is sent. Validation failures cost no
network call and land in `_error`.

As a column, the emission is a regular struct:

```python
df.select(pl.col("calls").struct.field("calls").list.len())   # calls per row
df.filter(pl.col("calls").struct.field("_error").is_not_null())  # emission failures
```

## The results format

`execute_tool_calls` returns
`List[Struct{tool_name, arguments, content, is_error, _error}]` — one entry
per emitted call, order preserved within each row:

- `content` — text returned by the tool (text blocks concatenated;
  `structuredContent` serialized as JSON)
- `is_error` — the tool itself reported failure (MCP `isError`), or the call
  never completed
- `_error` — transport/validation error detail, `null` for clean calls

Failures never throw mid-batch. Audit and retry with ordinary dataframe ops:

```python
failed = df.filter(
    pl.col("results").list.eval(pl.element().struct.field("is_error")).list.any()
)
```

The one exception: an unreachable or misconfigured `transport` raises
immediately — a typo'd URL failing silently into 10,000 error rows would be
worse.

## Non-MCP targets

Anything callable from Python can be a tool target via the `executor` escape
hatch — a database, an internal API, a local function. It runs on a thread
pool with the same errors-as-data semantics (at the cost of the GIL):

```python
def executor(tool_name: str, arguments: dict):
    if tool_name == "search_food_db":
        return my_db.search(**arguments)      # str or JSON-serializable
    raise ValueError(f"unknown tool {tool_name}")

df = df.with_columns(
    results=execute_tool_calls(pl.col("calls"), executor=executor, concurrency=16)
)
```

Return a `(content, is_error)` tuple to signal a tool-level failure without
raising.

## Namespace API

Both steps are available on the `.llama` namespace:

```python
df = df.with_columns(
    results=pl.col("calls").llama.execute_tool_calls(transport="http://localhost:8811/mcp"),
).with_columns(
    msg=pl.col("results").llama.tool_results_to_message(),
)
```

## What is deliberately not here

- **No per-row agent loop.** Multi-turn pipelines are written as explicit
  emit → execute passes. See the design doc for the acceptance criteria a
  future `agent_async` would have to meet.
- **No MCP sessions/state per row.** Rows are independent; one MCP session
  serves the whole batch.
- **No planners, memory, or graphs.** Polar Llama's lane is batch.
