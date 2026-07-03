"""
Tests for tool use: emission schemas, MCP introspection, and batch execution.

These tests run against an in-process mock MCP server (no API keys needed),
so they exercise the full pipeline deterministically:
emission schema -> serialized calls -> parallel execution -> results column.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import polars as pl
import pytest

from polar_llama import (
    TOOL_RESULT_DTYPE,
    execute_tool_calls,
    mcp_tools,
    tool_results_to_message,
    tools_to_response_model,
)
from polar_llama import _json_schema_to_polars_dtype, _pydantic_to_json_schema

SEARCH_FOOD_TOOL = {
    "name": "search_food_db",
    "description": "Search a nutrition database for a food item",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "database": {"type": "string", "enum": ["usda", "openfoodfacts"]},
        },
        "required": ["query", "database"],
    },
}

CONVERT_UNITS_TOOL = {
    "name": "convert_units",
    "description": "Convert a quantity between units",
    "inputSchema": {
        "type": "object",
        "properties": {
            "quantity": {"type": "number"},
            "from_unit": {"type": "string"},
            "to_unit": {"type": "string"},
        },
        "required": ["quantity", "from_unit", "to_unit"],
    },
}


# ============================================================================
# Mock MCP server (streamable HTTP)
# ============================================================================

class MockMcpHandler(BaseHTTPRequestHandler):
    """Implements initialize / tools/list / tools/call over streamable HTTP."""

    calls_received = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        method = body.get("method")

        if method == "initialize":
            self._respond(body, {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-nutrition", "version": "0.1"},
            }, session_id="test-session-1")
        elif method == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
        elif method == "tools/list":
            self._respond(body, {"tools": [SEARCH_FOOD_TOOL, CONVERT_UNITS_TOOL]})
        elif method == "tools/call":
            params = body.get("params", {})
            type(self).calls_received.append(params)
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "search_food_db":
                content = json.dumps({
                    "food": args.get("query"),
                    "database": args.get("database"),
                    "calories_per_100g": 155,
                })
                self._respond(body, {"content": [{"type": "text", "text": content}]})
            elif name == "convert_units":
                self._respond(body, {"content": [{"type": "text", "text": "converted"}]})
            elif name == "always_fails":
                self._respond(body, {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True,
                })
            else:
                self._respond(body, error={"code": -32602, "message": f"unknown tool {name}"})
        else:
            self._respond(body, error={"code": -32601, "message": "method not found"})

    def _respond(self, request, result=None, error=None, session_id=None):
        payload = {"jsonrpc": "2.0", "id": request.get("id")}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # keep test output clean


@pytest.fixture(scope="module")
def mcp_server():
    server = HTTPServer(("127.0.0.1", 0), MockMcpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    server.shutdown()


@pytest.fixture(autouse=True)
def clear_recorded_calls():
    MockMcpHandler.calls_received = []


# ============================================================================
# Phase 0: emission schema
# ============================================================================

def test_tools_to_response_model_shape():
    model = tools_to_response_model([SEARCH_FOOD_TOOL, CONVERT_UNITS_TOOL])
    schema = model.model_json_schema()

    assert "calls" in schema["properties"]
    tool_call_schema = schema["$defs"]["ToolCall"]
    assert set(tool_call_schema["required"]) == {"tool_name", "arguments"}
    assert tool_call_schema["properties"]["tool_name"]["enum"] == [
        "search_food_db",
        "convert_units",
    ]
    # Each tool's own input schema is embedded so the LLM generates against
    # the tool's actual contract.
    assert "search_food_db" in tool_call_schema["properties"]["arguments"]["description"]
    assert "openfoodfacts" in tool_call_schema["properties"]["arguments"]["description"]

    # Round-trips through the library's structured-output machinery.
    instance = model(calls=[{
        "tool_name": "search_food_db",
        "arguments": json.dumps({"query": "eggs", "database": "usda"}),
    }])
    assert instance.calls[0].tool_name == "search_food_db"

    with pytest.raises(Exception):
        model(calls=[{"tool_name": "not_a_tool", "arguments": "{}"}])


def test_tools_to_response_model_polars_dtype():
    """The emission model must lower to a usable Polars struct dtype."""
    model = tools_to_response_model([SEARCH_FOOD_TOOL])
    dtype = _json_schema_to_polars_dtype(_pydantic_to_json_schema(model))

    fields = {f.name: f.dtype for f in dtype.fields}
    assert isinstance(fields["calls"], pl.List)
    inner = fields["calls"].inner
    inner_fields = {f.name: f.dtype for f in inner.fields}
    assert inner_fields == {"tool_name": pl.Utf8, "arguments": pl.Utf8}
    # Error columns from the structured-output convention are present.
    assert "_error" in fields and "_raw" in fields


def test_tools_normalization_accepts_multiple_forms():
    from pydantic import BaseModel

    class lookup_recipe(BaseModel):
        """Look up a recipe by name."""
        name: str

    openai_form = {
        "type": "function",
        "function": {"name": "f1", "description": "d", "parameters": {"type": "object"}},
    }
    anthropic_form = {"name": "f2", "input_schema": {"type": "object"}}

    model = tools_to_response_model([lookup_recipe, openai_form, anthropic_form])
    specs = model.__polar_llama_tools__
    assert [s["name"] for s in specs] == ["lookup_recipe", "f1", "f2"]
    assert specs[0]["description"] == "Look up a recipe by name."
    assert "name" in specs[0]["input_schema"]["properties"]


def test_duplicate_tool_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        tools_to_response_model([SEARCH_FOOD_TOOL, SEARCH_FOOD_TOOL])


# ============================================================================
# Phase 0: MCP introspection
# ============================================================================

def test_mcp_tools_introspection(mcp_server):
    tools = mcp_tools(mcp_server)
    assert [t["name"] for t in tools] == ["search_food_db", "convert_units"]
    assert tools[0]["input_schema"]["required"] == ["query", "database"]

    # Introspected tools feed straight into the emission model.
    model = tools_to_response_model(tools)
    assert model.__polar_llama_tools__[0]["name"] == "search_food_db"


def test_mcp_tools_rejects_unknown_transport():
    with pytest.raises(ValueError, match="transport"):
        mcp_tools("ftp://nope")


# ============================================================================
# Phase 1: execution — Rust MCP path
# ============================================================================

def _emitted_calls_df():
    """A dataframe shaped like the output of the emission step."""
    calls_dtype = pl.Struct([
        pl.Field("calls", pl.List(pl.Struct([
            pl.Field("tool_name", pl.Utf8),
            pl.Field("arguments", pl.Utf8),
        ]))),
        pl.Field("_error", pl.Utf8),
    ])
    return pl.DataFrame({
        "meal": ["two eggs and toast", "chicken bowl", "just water"],
        "calls": pl.Series([
            {"calls": [
                {"tool_name": "search_food_db",
                 "arguments": json.dumps({"query": "eggs", "database": "usda"})},
                {"tool_name": "search_food_db",
                 "arguments": json.dumps({"query": "toast", "database": "openfoodfacts"})},
            ], "_error": None},
            {"calls": [
                {"tool_name": "search_food_db",
                 "arguments": json.dumps({"query": "chicken bowl", "database": "usda"})},
            ], "_error": None},
            {"calls": [], "_error": None},
        ], dtype=calls_dtype),
    })


def test_execute_tool_calls_mcp(mcp_server):
    df = _emitted_calls_df().with_columns(
        results=execute_tool_calls(pl.col("calls"), transport=mcp_server)
    )

    assert df["results"].dtype == TOOL_RESULT_DTYPE
    row0 = df["results"][0].to_list()
    assert len(row0) == 2
    assert row0[0]["tool_name"] == "search_food_db"
    assert row0[0]["is_error"] is False
    assert json.loads(row0[0]["content"])["food"] == "eggs"
    # Order within a row is preserved even though execution is parallel.
    assert json.loads(row0[1]["content"])["food"] == "toast"
    # Empty emission -> empty results, not null and not an error.
    assert df["results"][2].to_list() == []
    # All 3 calls actually reached the server.
    assert len(MockMcpHandler.calls_received) == 3


def test_execute_tool_calls_validates_arguments(mcp_server):
    """Invalid arguments fail fast as data — no network call is made."""
    df = pl.DataFrame({
        "calls": [json.dumps([
            {"tool_name": "search_food_db",
             "arguments": json.dumps({"query": "eggs"})},  # missing `database`
        ])],
    }).with_columns(
        results=execute_tool_calls(
            pl.col("calls"),
            transport=mcp_server,
            tools=[SEARCH_FOOD_TOOL, CONVERT_UNITS_TOOL],
        )
    )

    result = df["results"][0].to_list()[0]
    assert result["is_error"] is True
    assert "validation failed" in result["_error"]
    assert len(MockMcpHandler.calls_received) == 0


def test_execute_tool_calls_tool_error_is_data(mcp_server):
    df = pl.DataFrame({
        "calls": [json.dumps([{"tool_name": "always_fails", "arguments": "{}"}])],
    }).with_columns(
        results=execute_tool_calls(pl.col("calls"), transport=mcp_server)
    )
    result = df["results"][0].to_list()[0]
    assert result["is_error"] is True
    assert result["content"] == "boom"


def test_execute_tool_calls_unreachable_transport_raises():
    df = pl.DataFrame({
        "calls": [json.dumps([{"tool_name": "x", "arguments": "{}"}])],
    })
    with pytest.raises(pl.exceptions.ComputeError, match="MCP"):
        df.with_columns(
            results=execute_tool_calls(
                pl.col("calls"), transport="http://127.0.0.1:9/mcp", timeout_s=2
            )
        )


# ============================================================================
# Phase 1: execution — Python executor escape hatch
# ============================================================================

def test_execute_tool_calls_python_executor():
    seen = []

    def executor(tool_name, arguments):
        seen.append((tool_name, arguments))
        if arguments.get("query") == "explode":
            raise RuntimeError("db unavailable")
        return {"hits": 1, "query": arguments.get("query")}

    df = _emitted_calls_df().with_columns(
        results=execute_tool_calls(pl.col("calls"), executor=executor)
    )

    row0 = df["results"][0].to_list()
    assert json.loads(row0[0]["content"]) == {"hits": 1, "query": "eggs"}
    assert row0[0]["is_error"] is False
    assert len(seen) == 3

    # A raising executor becomes error data on the right call.
    df2 = pl.DataFrame({
        "calls": [json.dumps([
            {"tool_name": "search_food_db", "arguments": json.dumps({"query": "explode"})},
            {"tool_name": "search_food_db", "arguments": json.dumps({"query": "fine"})},
        ])],
    }).with_columns(results=execute_tool_calls(pl.col("calls"), executor=executor))
    results = df2["results"][0].to_list()
    assert results[0]["is_error"] is True
    assert "db unavailable" in results[0]["_error"]
    assert results[1]["is_error"] is False


def test_execute_tool_calls_requires_exactly_one_target():
    with pytest.raises(ValueError, match="exactly one"):
        execute_tool_calls(pl.col("calls"))
    with pytest.raises(ValueError, match="exactly one"):
        execute_tool_calls(pl.col("calls"), transport="http://x", executor=lambda n, a: "")


def test_null_rows_stay_null():
    df = pl.DataFrame({"calls": pl.Series([None], dtype=pl.Utf8)}).with_columns(
        results=execute_tool_calls(pl.col("calls"), executor=lambda n, a: "ok")
    )
    assert df["results"][0] is None


# ============================================================================
# Synthesis: results -> message
# ============================================================================

def test_tool_results_to_message():
    results = pl.Series([[
        {"tool_name": "search_food_db",
         "arguments": '{"query": "eggs", "database": "usda"}',
         "content": '{"calories_per_100g": 155}',
         "is_error": False, "_error": None},
        {"tool_name": "search_food_db",
         "arguments": '{"query": "toast"}',
         "content": None, "is_error": True,
         "_error": "argument validation failed: 'database' is required"},
    ]], dtype=TOOL_RESULT_DTYPE)

    df = pl.DataFrame({"results": results}).with_columns(
        message=tool_results_to_message(pl.col("results"))
    )
    message = json.loads(df["message"][0])
    assert message["role"] == "user"
    assert "search_food_db" in message["content"]
    assert "calories_per_100g" in message["content"]
    assert "ERROR" in message["content"]
