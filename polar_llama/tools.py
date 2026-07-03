"""
Tool use for Polar Llama: emission schemas, MCP introspection, and
batch-parallel execution.

Design: docs/design/MCP_TOOL_INTEGRATION.md. The ReAct loop is unrolled into
the dataframe — tool-call *emission* is ordinary structured output generated
against the tool's own input schema, tool-call *execution* is a
batch-parallel expression, and every intermediate turn is a column.

MCP's role is deliberately narrow: it is a schema source (``tools/list``)
and a call target (``tools/call``) — a registry, not a runtime.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Type, Union

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr
    from pydantic import BaseModel

MCP_PROTOCOL_VERSION = "2025-03-26"

# Dtype of one executed tool call; execute_tool_calls returns a List of these
# per row. Failures are data (`is_error`, `_error`), not exceptions, matching
# the structured-output error convention.
TOOL_RESULT_STRUCT = pl.Struct([
    pl.Field("tool_name", pl.Utf8),
    pl.Field("arguments", pl.Utf8),
    pl.Field("content", pl.Utf8),
    pl.Field("is_error", pl.Boolean),
    pl.Field("_error", pl.Utf8),
])
TOOL_RESULT_DTYPE = pl.List(TOOL_RESULT_STRUCT)


# ============================================================================
# Tool spec normalization
# ============================================================================

def _normalize_tool(tool: Any) -> Dict[str, Any]:
    """Normalize a tool definition to {name, description, input_schema}.

    Accepts:
    - dicts in MCP form ({name, description, inputSchema}),
      Anthropic form ({name, description, input_schema}),
      or OpenAI function form ({name, description, parameters} or
      {"type": "function", "function": {...}})
    - Pydantic BaseModel subclasses (class name / docstring / model schema)
    """
    try:
        from pydantic import BaseModel
        if isinstance(tool, type) and issubclass(tool, BaseModel):
            return {
                "name": tool.__name__,
                "description": (tool.__doc__ or "").strip(),
                "input_schema": tool.model_json_schema(),
            }
    except ImportError:
        pass

    if not isinstance(tool, dict):
        raise TypeError(
            f"Tool must be a dict or a Pydantic BaseModel subclass, got {type(tool)!r}"
        )

    # OpenAI tool-list wrapper: {"type": "function", "function": {...}}
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        tool = tool["function"]

    name = tool.get("name")
    if not name:
        raise ValueError(f"Tool definition is missing a name: {tool!r}")

    schema = (
        tool.get("input_schema")
        or tool.get("inputSchema")
        or tool.get("parameters")
        or {"type": "object", "properties": {}}
    )

    return {
        "name": name,
        "description": tool.get("description", "") or "",
        "input_schema": schema,
    }


def _normalize_tools(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    specs = [_normalize_tool(t) for t in tools]
    if not specs:
        raise ValueError("At least one tool is required")
    seen = set()
    for spec in specs:
        if spec["name"] in seen:
            raise ValueError(f"Duplicate tool name: {spec['name']!r}")
        seen.add(spec["name"])
    return specs


# ============================================================================
# Emission: tools -> response model (Phase 0)
# ============================================================================

def tools_to_response_model(
    tools: Sequence[Any],
    *,
    model_name: str = "ToolCalls",
) -> Type["BaseModel"]:
    """
    Build a Pydantic response model for tool-call *emission*.

    Pass the result as ``response_model=`` to ``inference_async`` /
    ``inference_messages``: the LLM then emits zero or more tool calls per
    row as ordinary structured output. Nothing is executed — execution is a
    separate, explicit step (``execute_tool_calls``).

    The generated shape is the strict-mode-safe tagged union::

        {"calls": [{"tool_name": "<enum of tool names>",
                    "arguments": "<JSON-encoded object>"}, ...]}

    ``arguments`` is carried as a JSON-encoded string because strict
    structured-output modes handle it reliably where open ``anyOf`` unions
    are not; each tool's input schema is embedded in the field description
    so the model generates against the tool's own contract, and
    ``execute_tool_calls`` re-validates arguments against the selected
    tool's schema before any call is made.

    Parameters
    ----------
    tools : sequence of dict or Pydantic BaseModel
        Tool definitions (MCP ``tools/list`` entries, Anthropic/OpenAI-style
        dicts, or the output of ``mcp_tools``).
    model_name : str
        Name for the generated root model.

    Returns
    -------
    Type[pydantic.BaseModel]
        Root model with a single ``calls`` field. The normalized tool specs
        are attached as ``__polar_llama_tools__`` so ``execute_tool_calls``
        can validate against them.
    """
    try:
        from pydantic import Field, create_model
    except ImportError:
        raise ImportError(
            "Pydantic is required for tool use. Install with: pip install pydantic>=2.0.0"
        )

    specs = _normalize_tools(tools)
    names = tuple(spec["name"] for spec in specs)

    tool_menu = "\n".join(
        f"- {spec['name']}: {spec['description']}".rstrip(": ") for spec in specs
    )
    schema_menu = json.dumps(
        {spec["name"]: spec["input_schema"] for spec in specs}, indent=None
    )

    tool_call_model = create_model(
        "ToolCall",
        tool_name=(
            Literal[names],  # type: ignore[valid-type]
            Field(..., description=f"The tool to call. Available tools:\n{tool_menu}"),
        ),
        arguments=(
            str,
            Field(
                ...,
                description=(
                    "JSON-encoded object of arguments for the selected tool, "
                    "matching that tool's input schema exactly. "
                    f"Input schemas by tool name: {schema_menu}"
                ),
            ),
        ),
    )

    response_model = create_model(
        model_name,
        calls=(
            List[tool_call_model],  # type: ignore[valid-type]
            Field(
                ...,
                description=(
                    "The tool calls needed to fulfil the request, in order. "
                    "Return an empty list if no tool call is needed."
                ),
            ),
        ),
    )
    response_model.__polar_llama_tools__ = specs
    return response_model


# ============================================================================
# MCP introspection: tools/list (Phase 0)
# ============================================================================

def mcp_tools(transport: str, *, timeout_s: float = 30.0) -> List[Dict[str, Any]]:
    """
    Fetch tool definitions from an MCP server via ``tools/list``.

    Parameters
    ----------
    transport : str
        Either a streamable-HTTP endpoint (``"http://host:port/mcp"``) or a
        stdio command prefixed with ``stdio:`` (``"stdio:my-server --flag"``).
    timeout_s : float
        Timeout for each protocol request.

    Returns
    -------
    list of dict
        Normalized tool specs ``{name, description, input_schema}``, ready
        for ``tools_to_response_model`` and ``execute_tool_calls(tools=...)``.
    """
    session = _open_mcp_session(transport, timeout_s)
    try:
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = session.request("tools/list", params)
            tools.extend(_normalize_tool(t) for t in result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools
    finally:
        session.close()


def _open_mcp_session(transport: str, timeout_s: float) -> "_McpSession":
    if transport.startswith("stdio:"):
        session: _McpSession = _McpStdioSession(transport[len("stdio:"):], timeout_s)
    elif transport.startswith(("http://", "https://")):
        session = _McpHttpSession(transport, timeout_s)
    else:
        raise ValueError(
            f"Unsupported MCP transport {transport!r}: expected an http(s) URL "
            "or a 'stdio:<command>' string"
        )
    session.initialize()
    return session


class _McpSession:
    """Minimal MCP client session: initialize + request/notify."""

    def initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "polar-llama", "version": "0"},
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("MCP initialize returned no result")
        self.notify("notifications/initialized", {})

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _McpHttpSession(_McpSession):
    def __init__(self, endpoint: str, timeout_s: float):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.session_id: Optional[str] = None
        self._next_id = 1

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        message_id = self._next_id
        self._next_id += 1
        body = {"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}
        payload = self._post(body)
        if payload is None:
            raise RuntimeError(f"MCP server returned no response for {method}")
        if "error" in payload:
            raise RuntimeError(f"MCP error for {method}: {payload['error']}")
        return payload.get("result", {})

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params})

    def _post(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
                content_type = response.headers.get("Content-Type", "")
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"MCP HTTP request failed: {e.code} {e.reason}: "
                f"{e.read().decode('utf-8', errors='replace')[:500]}"
            ) from e

        if not text.strip():
            return None
        if content_type.startswith("text/event-stream"):
            return _parse_sse_response(text)
        return json.loads(text)


def _parse_sse_response(body: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON-RPC response (the event carrying an id) from an SSE body."""
    response = None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                value = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "id" in value and ("result" in value or "error" in value):
                response = value
    return response


class _McpStdioSession(_McpSession):
    def __init__(self, command: str, timeout_s: float):
        self.timeout_s = timeout_s
        self._next_id = 1
        self.process = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        message_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params})
        # Read newline-delimited JSON until our response arrives, skipping
        # any server-initiated notifications.
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP stdio server exited before responding to {method}")
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == message_id:
                if "error" in payload:
                    raise RuntimeError(f"MCP error for {method}: {payload['error']}")
                return payload.get("result", {})

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, body: Dict[str, Any]) -> None:
        self.process.stdin.write(json.dumps(body) + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


# ============================================================================
# Execution (Phase 1)
# ============================================================================

def execute_tool_calls(
    expr: "IntoExpr",
    *,
    transport: Optional[str] = None,
    executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    tools: Optional[Sequence[Any]] = None,
    concurrency: int = 32,
    timeout_s: int = 30,
) -> pl.Expr:
    """
    Execute a column of emitted tool calls, in parallel across all calls of
    all rows.

    The input column is the output of an emission step (a struct with a
    ``calls`` field, as produced by ``response_model=tools_to_response_model(...)``),
    or directly a ``List[Struct{tool_name, arguments}]`` column, or a Utf8
    column of JSON arrays.

    Exactly one of ``transport`` / ``executor`` must be given:

    - ``transport``: an MCP streamable-HTTP endpoint. Calls run on the Rust
      async runtime — the same fan-out as ``inference_async`` — so 10k rows
      x 4 calls each is 40k concurrent-capped requests, not a serial loop.
    - ``executor``: a Python callable ``(tool_name, arguments_dict) -> content``
      for non-MCP targets (a database, an internal API). Runs on a thread
      pool; return a string (or any JSON-serializable value), or a
      ``(content, is_error)`` tuple. Raising marks that call failed.

    Per-call failures are **data** (``is_error``, ``_error`` fields in the
    result), never exceptions — filter and re-run failed rows instead of
    losing the batch. Only an unreachable/misconfigured transport raises.

    Parameters
    ----------
    expr
        The emitted-calls column.
    transport : str, optional
        MCP streamable-HTTP endpoint (e.g. ``"http://localhost:8811/mcp"``).
    executor : callable, optional
        Python fallback executor for non-MCP targets.
    tools : sequence, optional
        Tool definitions (e.g. from ``mcp_tools``). When provided, each
        call's arguments are validated against the selected tool's input
        schema *before* execution; invalid calls fail fast as data.
    concurrency : int
        Maximum in-flight calls across the whole batch (default 32).
    timeout_s : int
        Per-call timeout in seconds (default 30).

    Returns
    -------
    polars.Expr
        ``List[Struct{tool_name, arguments, content, is_error, _error}]``,
        one entry per emitted call, order preserved within each row.
    """
    if (transport is None) == (executor is None):
        raise ValueError("Provide exactly one of `transport` or `executor`")

    expr = parse_into_expr(expr)
    calls_json = expr.map_batches(_serialize_calls_batch, return_dtype=pl.Utf8)

    tool_schemas: Optional[Dict[str, Any]] = None
    if tools is not None:
        tool_schemas = {
            spec["name"]: spec["input_schema"] for spec in _normalize_tools(tools)
        }

    if executor is not None:
        return calls_json.map_batches(
            lambda s: _execute_batch_python(s, executor, tool_schemas, concurrency, timeout_s),
            return_dtype=TOOL_RESULT_DTYPE,
        )

    from polar_llama.expressions import get_lib_path

    kwargs: Dict[str, Any] = {
        "transport": transport,
        "concurrency": int(concurrency),
        "timeout_s": int(timeout_s),
    }
    if tool_schemas is not None:
        kwargs["tool_schemas"] = json.dumps(tool_schemas)

    result = register_plugin(
        args=[calls_json],
        symbol="execute_tool_calls",
        is_elementwise=True,
        lib=get_lib_path(),
        kwargs=kwargs,
    )
    return result.map_batches(
        lambda s: s.str.json_decode(dtype=TOOL_RESULT_DTYPE),
        return_dtype=TOOL_RESULT_DTYPE,
    )


def _serialize_calls_batch(series: pl.Series) -> pl.Series:
    """Serialize an emitted-calls column to one JSON array string per row."""
    out: List[Optional[str]] = []
    for row in series.to_list():
        if row is None:
            out.append(None)
            continue
        calls = row
        if isinstance(row, dict):
            # Structured-output root struct: {"calls": [...], "_error": ...}.
            # An emission error means no calls; the error stays visible in
            # the emission column itself.
            calls = row.get("calls")
        if calls is None:
            calls = []
        if isinstance(calls, str):
            out.append(calls)
            continue
        normalized = []
        for call in calls:
            if call is None:
                continue
            normalized.append({
                "tool_name": call.get("tool_name"),
                "arguments": call.get("arguments"),
            })
        out.append(json.dumps(normalized))
    return pl.Series(series.name, out, dtype=pl.Utf8)


def _execute_batch_python(
    series: pl.Series,
    executor: Callable[[str, Dict[str, Any]], Any],
    tool_schemas: Optional[Dict[str, Any]],
    concurrency: int,
    timeout_s: int,
) -> pl.Series:
    rows: List[Optional[List[Dict[str, Any]]]] = []
    for value in series.to_list():
        rows.append(None if value is None else json.loads(value))

    flat = [
        (row_idx, call_idx, call)
        for row_idx, row in enumerate(rows)
        if row is not None
        for call_idx, call in enumerate(row)
    ]

    results: Dict[tuple, Dict[str, Any]] = {}
    if flat:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {
                pool.submit(_run_one_python_call, executor, call, tool_schemas): (row_idx, call_idx)
                for row_idx, call_idx, call in flat
            }
            for future, key in futures.items():
                row_idx, call_idx = key
                call = rows[row_idx][call_idx]
                try:
                    results[key] = future.result(timeout=timeout_s)
                except Exception as e:  # includes TimeoutError
                    results[key] = _tool_result(call, None, True, f"{type(e).__name__}: {e}")

    out: List[Optional[List[Dict[str, Any]]]] = []
    for row_idx, row in enumerate(rows):
        if row is None:
            out.append(None)
        else:
            out.append([results[(row_idx, call_idx)] for call_idx in range(len(row))])
    return pl.Series(series.name, out, dtype=TOOL_RESULT_DTYPE)


def _run_one_python_call(
    executor: Callable[[str, Dict[str, Any]], Any],
    call: Dict[str, Any],
    tool_schemas: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    tool_name = call.get("tool_name")
    raw_arguments = call.get("arguments")

    try:
        arguments = _parse_arguments(raw_arguments)
    except ValueError as e:
        return _tool_result(call, None, True, str(e))

    if tool_schemas is not None:
        if tool_name not in tool_schemas:
            return _tool_result(call, None, True, f"unknown tool: {tool_name!r}")
        error = _validate_arguments(arguments, tool_schemas[tool_name])
        if error:
            return _tool_result(call, None, True, f"argument validation failed: {error}")

    try:
        value = executor(tool_name, arguments)
    except Exception as e:
        return _tool_result(call, None, True, f"{type(e).__name__}: {e}")

    is_error = False
    if isinstance(value, tuple) and len(value) == 2:
        value, is_error = value
    content = value if isinstance(value, str) else json.dumps(value)
    return _tool_result(call, content, bool(is_error), None)


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"arguments is not valid JSON: {e}")
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
        return parsed
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"arguments must be a JSON object or JSON-encoded string, got {type(raw)!r}")


def _validate_arguments(arguments: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """Validate with the `jsonschema` package when available; otherwise only
    check required keys (the Rust path always validates fully)."""
    try:
        import jsonschema as _jsonschema  # type: ignore

        validator_cls = _jsonschema.validators.validator_for(schema)
        errors = [
            f"{e.message} at /{'/'.join(str(p) for p in e.absolute_path)}"
            for e in validator_cls(schema).iter_errors(arguments)
        ]
        return "; ".join(errors) if errors else None
    except ImportError:
        missing = [
            key for key in schema.get("required", []) if key not in arguments
        ]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"
        return None


def _tool_result(
    call: Dict[str, Any],
    content: Optional[str],
    is_error: bool,
    error: Optional[str],
) -> Dict[str, Any]:
    raw_arguments = call.get("arguments")
    if raw_arguments is not None and not isinstance(raw_arguments, str):
        raw_arguments = json.dumps(raw_arguments)
    return {
        "tool_name": call.get("tool_name"),
        "arguments": raw_arguments,
        "content": content,
        "is_error": is_error,
        "_error": error,
    }


# ============================================================================
# Synthesis helper: results -> message
# ============================================================================

def tool_results_to_message(expr: "IntoExpr", *, role: str = "user") -> pl.Expr:
    """
    Render an ``execute_tool_calls`` results column as a message, ready for
    ``combine_messages`` + ``inference_messages`` (the synthesis turn).

    Parameters
    ----------
    expr
        Results column (``List[Struct{tool_name, arguments, content, is_error, _error}]``).
    role : str
        Message role, default ``"user"``.

    Returns
    -------
    polars.Expr
        JSON message expression, same shape as ``string_to_message`` output.
    """
    expr = parse_into_expr(expr)
    text = expr.map_batches(_format_results_batch, return_dtype=pl.Utf8)
    # Imported lazily to avoid a circular import at module load time.
    from polar_llama import string_to_message

    return string_to_message(text, message_type=role)


def _format_results_batch(series: pl.Series) -> pl.Series:
    out: List[Optional[str]] = []
    for row in series.to_list():
        if row is None:
            out.append(None)
            continue
        lines = ["Tool results:"]
        if not row:
            lines.append("(no tool calls were made)")
        for i, result in enumerate(row, start=1):
            result = result or {}
            name = result.get("tool_name") or "<unknown tool>"
            arguments = result.get("arguments") or "{}"
            lines.append(f"{i}. {name}({arguments})")
            if result.get("is_error"):
                reason = result.get("_error") or result.get("content") or "unknown error"
                lines.append(f"   ERROR: {reason}")
            else:
                lines.append(f"   -> {result.get('content')}")
        out.append("\n".join(lines))
    return pl.Series(series.name, out, dtype=pl.Utf8)
