# Design Proposal: MCP / Tool Use in Polar Llama

**Status:** Accepted — Phases 0–1 implemented (see `docs/TOOL_USE.md`); Phase 2 remains deferred
**Author:** drafted for discussion
**Date:** 2026-07-03

---

## 0. Summary

This document evaluates whether Polar Llama should support tool use (and MCP
specifically), and proposes a design that resolves the core ergonomic tension.

The recommendation, in one sentence:

> **Do not put an agent loop inside a row. Unroll the loop into the
> dataframe: tool-call *emission* is a structured output (which we already
> have), tool-call *execution* is a new batch-parallel expression, and every
> intermediate "turn" is an ordinary column.**

This gives us tool use that composes with dataframe semantics instead of
fighting them, reuses ~90% of existing machinery (structured outputs, the
Tokio runtime, the async fan-out), and leaves the genuinely speculative part —
a multi-turn ReAct loop — explicitly deferred until there is demonstrated
demand.

---

## 1. The honest framing: is this hype?

Tool use / MCP is unambiguously riding a hype cycle. That is not itself a
reason to build it or to refuse to build it. The question is whether it
survives a filter that has nothing to do with the hype:

**The filter: a feature belongs in Polar Llama if and only if it (a) is a
row-wise or batch-wise data transformation, (b) benefits from massively
parallel async execution, and (c) produces output that is usable as a column
without leaving the dataframe.**

Everything currently in the library passes this filter: inference, structured
outputs, embeddings, similarity, ANN search, taxonomy tagging. They are all
"columns in → column out, N rows in flight at once."

Applied to tool use, the filter splits the feature into three parts with very
different verdicts:

| Capability | Passes the filter? | Verdict |
|---|---|---|
| LLM **emits** tool calls (parameterizes a call from row text) | Yes — it is literally structured output | Build (mostly exists) |
| Harness **executes** tool calls in parallel across rows | Yes — batch async I/O fan-out, our core competency | Build |
| LLM runs an **agentic ReAct loop** per row (hidden multi-turn state) | No — hidden per-row state, unbounded latency/cost, opaque intermediaries | Defer / likely never |

The hype cycle is selling the third row. The durable value for a dataframe
library is in the first two. Most "agent framework" libraries have no story
for *"execute this tool call for 100,000 rows concurrently"* — we do, for
free, because it is the same shape as `inference_async`. That is the
genuine, non-clout-chasing gain.

---

## 2. The objections, taken seriously

The arguments against integration (raised in the issue prompting this doc)
are the right ones. Each is addressed by the design in §3, but stated fairly
first:

### 2.1 "It defeats the core purpose"

Polar Llama's premise is that *the LLM itself* is the transformation — text
in, transformed text/struct out. A tool loop inverts that: the LLM becomes a
controller of external side effects, and the dataframe becomes an
afterthought.

**Accepted, with a boundary.** This is a decisive argument against the ReAct
loop, and it is why the loop is deferred. It is *not* an argument against
emission + execution: "parameterize a lookup from messy text" is exactly the
kind of textual transformation the library exists for, and "run the lookup
10,000× concurrently" is exactly the kind of parallelism it exists for.

### 2.2 "Who makes the call — the LLM mid-loop, or the harness?"

If the LLM calls tools inside a session/row (ReAct), the intermediaries are
trapped inside an opaque per-row transcript: not inspectable, not cacheable,
not resumable, not joinable. If the harness makes the call, then the LLM's
output is just a structured object — so why is that different from what
`response_model` gives us today, plus some `apply`?

**Accepted — and this is the crux.** The answer is: it *isn't* meaningfully
different, and we should stop pretending otherwise. The design below embraces
the second horn of the dilemma. Emission **is** structured output. What we
add is not a new inference mode but:

1. **Schema plumbing** — deriving the emission schema *from* a tool's own
   `inputSchema` (MCP or otherwise) so users don't hand-transcribe it, and so
   the output is guaranteed to be in the shape the tool accepts (this answers
   "how would the LLM reformat per the MCP" — it never has to; the schema it
   generates against *is* the MCP schema).
2. **A parallel executor** — because the naive alternative ("some sort of
   apply") is a serial Python `map_elements` loop doing blocking I/O, which
   throws away the entire point of the library. `execute_tool_calls` is to
   `apply` what `inference_async` is to `openai.chat.completions.create` in a
   for-loop.

The intermediaries question answers itself under this design: **every
intermediate is a column.** Emitted calls: a column. Tool results: a column.
The follow-up synthesis: a column. Nothing is hidden; everything can be
filtered, exploded, joined, cached to parquet, and resumed.

### 2.3 "Can strict formats even support multiple tools?"

Strict structured-output modes (OpenAI strict mode, etc.) dislike open unions,
and a multi-tool schema is a tagged union: *"either a `search_food(query,
db)` call or a `convert_units(qty, from, to)` call."*

**Real, but solvable — and we already own the solution surface.** We already
maintain schema-lowering logic (`_pydantic_to_json_schema`,
`_validate_strict_mode_schema`, `anyOf` handling for `Optional`). The design
handles multi-tool as follows:

- The emission dtype is `List[Struct{ tool_name: Utf8, arguments: Utf8 }]` —
  a list of calls, arguments carried as a JSON string.
- For providers with robust union support, we generate a proper
  `anyOf`-of-tool-schemas and serialize `arguments` after the fact.
- For strict-mode providers, we generate the degenerate-but-reliable form
  (`tool_name` as an enum, `arguments` as a JSON-encoded string) and
  **validate `arguments` against the selected tool's `inputSchema` in Rust**
  (the `jsonschema` crate is already a dependency). Invalid calls surface in
  the same `_error` / `_details` / `_raw` convention structured outputs use
  today — the failure mode is a filterable column, not an exception.

This also answers "how would a user utilize the output effectively": the
same way they use any struct column today — `.list.explode()`,
`.struct.field(...)`, filter on `_error`, join results back.

---

## 3. Proposed design: the unrolled loop

### 3.1 Mental model

A ReAct loop is:

```
while not done:
    llm_output = llm(context)            # may contain tool calls
    tool_results = run(llm_output.calls)
    context += tool_results
```

In a dataframe, each iteration of that loop is a `with_columns` pass. One
"turn" = one column. The user decides how many turns their pipeline has,
statically, by writing them — exactly like every other multi-step Polar
Llama pipeline (see the embedding → tag → filter → knn pattern in the
README). The loop is unrolled, visible, and bounded by construction.

```
row text ──► emit calls ──► execute (parallel) ──► synthesize ──► answer
 (col)         (col)             (col)                (col)
```

### 3.2 API surface

#### Phase 0 — Tool-call emission (pure Python, no new core code)

```python
from polar_llama import tools_to_response_model, Provider

# Tools defined as plain dicts (OpenAI-function-style / MCP inputSchema),
# introspected from a live MCP server, or given as Pydantic models.
tools = [
    {
        "name": "search_food_db",
        "description": "Search a nutrition database for a food item",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "database": {"type": "string", "enum": ["usda", "openfoodfacts"]},
            },
            "required": ["query", "database"],
        },
    },
    # ... more tools
]

ToolCalls = tools_to_response_model(tools)   # -> a generated Pydantic model

df = df.with_columns(
    calls=pl.col("meal_description").llama.inference_async(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        response_model=ToolCalls,            # existing machinery, unchanged
    )
)
# df["calls"] : Struct{ calls: List[Struct{tool_name, arguments}], _error, ... }
```

`tools_to_response_model` is ~100 lines of schema transformation sitting on
top of code paths that already exist. It also ships with an MCP flavor:

```python
from polar_llama import mcp_tools

tools = mcp_tools("stdio:nutrition-server")   # tools/list via MCP handshake
# or: mcp_tools("http://localhost:8811/mcp")
```

MCP's role in this design is deliberately narrow: **it is a schema source and
a call target — a registry, not a runtime.** We take `tools/list` and
`tools/call`; we do not take sessions, sampling, or roots.

#### Phase 1 — Batch-parallel execution (new Rust expression)

```python
from polar_llama import execute_tool_calls

df = df.with_columns(
    results=execute_tool_calls(
        pl.col("calls"),
        transport="http://localhost:8811/mcp",   # MCP server
        concurrency=64,                           # per-batch cap
        timeout_s=30,
    )
)
# df["results"] : List[Struct{ tool_name, arguments, content: Utf8,
#                              is_error: Boolean, _error: Utf8 }]
```

Implementation notes:

- Same skeleton as `inference_async` in `src/expressions.rs`: iterate rows,
  `tokio::spawn` per call, `join_all`, collect into a Series. The MCP
  `tools/call` request over streamable HTTP is a small JSON-RPC client on
  `reqwest`, which we already depend on; stdio transport can come later.
- Per-call failures are **data**, not exceptions (`is_error`, `_error`),
  consistent with structured-output error handling.
- A `python_tool_executor` escape hatch (`execute_tool_calls(...,
  executor=my_callable)`) lets users target non-MCP things (a database, an
  internal API) with the same parallel machinery, at the cost of the GIL —
  this keeps MCP optional rather than load-bearing.

#### Phase 1.5 — Synthesis (nothing new needed)

Folding results back into a final answer is composition of existing
primitives:

```python
df = df.with_columns(
    answer=inference_messages(
        combine_messages(
            pl.col("meal_description").llama.to_message(role="user"),
            tool_results_to_message(pl.col("results")),   # thin formatter
        ),
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        response_model=NutritionSummary,
    )
)
```

`tool_results_to_message` is a small helper that renders the results column
as a message (analogous to `string_to_message`). Users who want a second
round of tool calls simply repeat the emit → execute pair — that is the
unrolled loop.

#### Phase 2 — `agent_async` (explicitly deferred)

A bounded in-row loop (`max_turns=N`, full trace returned as a
`List[Struct]` transcript column) is sketched but **not proposed for
implementation**. It only becomes worth revisiting if, after Phases 0–1
ship, users demonstrably chain 3+ emit/execute rounds and the column
bookkeeping is the bottleneck. Acceptance criteria before building it:

1. At least a handful of real users/issues showing multi-round unrolled
   pipelines in the wild.
2. A design for surfacing per-turn intermediaries that is as inspectable as
   columns (a transcript struct column, not a log file).
3. Hard bounds: `max_turns`, per-row timeout, per-row cost ceiling (the
   `cost.rs` machinery gives us the hooks).

If those are never met, Phase 2 is never built, and that outcome is fine —
it would confirm that the loop was the hype and the batch executor was the
substance.

### 3.3 What is explicitly out of scope

- **MCP sessions/state per row.** Rows are independent; anything requiring a
  stateful server session per row breaks batch semantics and is rejected.
- **MCP sampling / elicitation / roots.** We are an MCP *client* for
  `tools/list` + `tools/call` only.
- **Streaming tool interactions.** Orthogonal; tracked by the existing
  streaming roadmap item.
- **Being an agent framework.** No planners, no memory, no graphs. Users who
  want that should use an agent framework and call it per row; our lane is
  batch.

---

## 4. Worked example: the calorie tracker

The motivating "genuinely helpful" case: a column of free-text meal logs;
each row implies *N* food items that must be searched across *M* nutrition
databases, then reconciled.

```python
import polars as pl
from polar_llama import (
    Provider, mcp_tools, tools_to_response_model,
    execute_tool_calls, inference_messages, combine_messages,
    tool_results_to_message,
)

tools = mcp_tools("http://localhost:8811/mcp")        # search_usda, search_off, ...
FoodSearches = tools_to_response_model(tools)

meals = pl.DataFrame({"meal": [
    "two eggs, sourdough toast with butter, black coffee",
    "chipotle chicken bowl, no rice, extra guac",
]})

result = (
    meals
    # Turn 1a: LLM parameterizes N searches per row (pure structured output)
    .with_columns(
        calls=pl.col("meal").llama.inference_async(
            provider=Provider.OPENAI, model="gpt-4o-mini",
            response_model=FoodSearches,
        )
    )
    # Turn 1b: all searches across all rows execute concurrently
    .with_columns(
        results=execute_tool_calls(
            pl.col("calls"), transport="http://localhost:8811/mcp",
        )
    )
    # Turn 2: reconcile results into a typed summary
    .with_columns(
        nutrition=inference_messages(
            combine_messages(
                pl.col("meal").llama.to_message(role="user"),
                tool_results_to_message(pl.col("results")),
            ),
            provider=Provider.OPENAI, model="gpt-4o-mini",
            response_model=NutritionSummary,
        )
    )
)
```

Properties worth noticing:

- With 10,000 meal logs averaging 4 food items, turn 1b is ~40,000 concurrent
  lookups through the existing Tokio fan-out. This is the step no agent
  framework does well and the step that justifies the feature's existence.
- Every intermediate (`calls`, `results`) is a real column: the user can
  `explode` it, audit which DB was chosen per item, filter rows where a
  search errored and re-run *only those* — the batch-processing answers to
  "how does the user get the intermediary."
- Deleting `execute_tool_calls` from this pipeline leaves valid Polar Llama
  code. Tool use is an *addition to* the algebra, not a new paradigm.

---

## 5. Decision summary

| Question raised | Answer under this design |
|---|---|
| ReAct loop or harness-executed? | Harness-executed, single-shot per turn; the loop is unrolled into columns. |
| Why not just structured outputs + apply? | It *is* structured outputs; the additions are schema derivation from tool/MCP definitions and a parallel executor, because serial `apply` forfeits the library's reason to exist. |
| How does the user get intermediaries? | They are columns. |
| How does the LLM reformat per the MCP? | It doesn't — emission is generated against the tool's own `inputSchema`, and Rust-side validation catches drift as `_error` data. |
| Multiple tools under strict formats? | `List[Struct{tool_name: enum, arguments: json-str}]` with post-hoc per-tool validation; native `anyOf` where the provider supports it. |
| Does it defeat the core purpose? | The loop would; the loop is deferred behind evidence-based criteria. Emission + batch execution *extend* the core purpose. |
| Hype or substance? | The per-row agent is the hype; batch-parallel tool fan-out over dataframes is the substance nobody else offers. |

## 6. Proposed sequencing

1. **Phase 0** (small PR, Python only): `tools_to_response_model`,
   `mcp_tools` (introspection only), docs + calorie-tracker example.
   De-risks nothing in the core; immediately useful.
2. **Phase 1** (Rust PR): `execute_tool_calls` over MCP streamable-HTTP +
   Python-callable executor; error-as-data semantics; concurrency/timeout
   controls; `.llama` namespace parity.
3. **Re-evaluate** with real usage before any Phase 2 discussion.
