"""CI-safe tests for the in-process local backend.

These tests never import ``mlx`` or touch a GPU: they exercise the engine seam
(``LocalEngine`` / ``FakeEngine`` / registry) and the ``inference_local``
``engine="in_process"`` map_batches path entirely on CPU. The real
``MlxBatchEngine`` path is intentionally NOT exercised here (BLOCKED-ON-GPU).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from polar_llama.local.engine import (
    FAKE_FAIL_MARKER,
    FakeEngine,
    LocalEngine,
    clear_registry,
    error_response,
    generate_chunked,
    get_engine,
    register_engine,
)
from polar_llama.local.expr import inference_local

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global registry between tests."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# FakeEngine basics
# ---------------------------------------------------------------------------
def test_fake_engine_is_local_engine():
    engine = FakeEngine("fake-model")
    assert isinstance(engine, LocalEngine)


def test_fake_engine_row_order_preserved():
    engine = FakeEngine()
    prompts = [f"prompt-{i}" for i in range(20)]
    out = engine.generate(prompts, max_tokens=512)
    assert out == [f"echo:prompt-{i}" for i in range(20)]


def test_fake_engine_max_tokens_truncation():
    engine = FakeEngine()
    # "echo:one two three four" -> 4 whitespace tokens.
    (result,) = engine.generate(["one two three four"], max_tokens=2)
    assert result == "echo:one two"

    (full,) = engine.generate(["one two three four"], max_tokens=512)
    assert full == "echo:one two three four"


def test_fake_engine_stop_sequence():
    engine = FakeEngine()
    (result,) = engine.generate(["hello|world"], stop=["|"])
    assert result == "echo:hello"


# ---------------------------------------------------------------------------
# Per-row error isolation
# ---------------------------------------------------------------------------
def test_error_response_shape_matches_rust():
    payload = error_response("api_error", "boom")
    obj = json.loads(payload)
    assert obj == {"_error": "api_error", "_details": "boom"}

    with_raw = json.loads(error_response("validation_failed", "bad", raw="{}"))
    assert with_raw == {
        "_error": "validation_failed",
        "_details": "bad",
        "_raw": "{}",
    }


def test_per_row_error_isolation_marker():
    engine = FakeEngine()
    prompts = ["good-0", f"bad-{FAKE_FAIL_MARKER}", "good-2"]
    out = engine.generate(prompts)

    assert out[0] == "echo:good-0"
    assert out[2] == "echo:good-2"

    err = json.loads(out[1])
    assert err["_error"] == "local_generation_error"
    assert "simulated failure" in err["_details"]


def test_per_row_error_isolation_predicate():
    engine = FakeEngine(fail_on=lambda p: p == "explode")
    out = engine.generate(["a", "explode", "b"])
    assert out[0] == "echo:a"
    assert out[2] == "echo:b"
    assert json.loads(out[1])["_error"] == "local_generation_error"


# ---------------------------------------------------------------------------
# Registry / singleton behaviour
# ---------------------------------------------------------------------------
def test_get_engine_singleton_identity():
    a = get_engine("m", engine="fake")
    b = get_engine("m", engine="fake")
    assert a is b


def test_get_engine_distinct_keys():
    a = get_engine("model-a", engine="fake")
    b = get_engine("model-b", engine="fake")
    assert a is not b


def test_clear_registry_forces_rebuild():
    a = get_engine("m", engine="fake")
    clear_registry()
    b = get_engine("m", engine="fake")
    assert a is not b


def test_register_engine_injection():
    sentinel = FakeEngine("injected")
    register_engine("m", sentinel, engine="in_process")
    # get_engine must return the pre-registered instance without building an mlx
    # engine (which would require the [local] extra).
    assert get_engine("m", engine="in_process") is sentinel


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------
def test_generate_chunked_preserves_order_across_chunks():
    engine = FakeEngine()
    prompts = [f"p{i}" for i in range(10)]
    out = generate_chunked(engine, prompts, chunk_size=3)
    assert out == [f"echo:p{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# End-to-end: inference_local(engine="in_process") over a DataFrame
# ---------------------------------------------------------------------------
def test_inference_local_in_process_via_registry():
    # Pre-register a FakeEngine under the in_process key so the map_batches UDF
    # resolves to it (no mlx import).
    register_engine("dummy-model", FakeEngine("dummy-model"), engine="in_process")

    df = pl.DataFrame({"prompt": ["What is 2+2?", "Name a color.", "Say hi."]})
    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model="dummy-model",
            engine="in_process",
            max_tokens=512,
        )
    )

    assert result.schema["answer"] == pl.String
    assert result["answer"].to_list() == [
        "echo:What is 2+2?",
        "echo:Name a color.",
        "echo:Say hi.",
    ]


def test_inference_local_in_process_env_override(monkeypatch):
    # POLAR_LLAMA_LOCAL_ENGINE=fake forces FakeEngine even for engine="in_process"
    # so the whole path is exercisable without pre-registering.
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    df = pl.DataFrame({"prompt": ["alpha", "beta", "gamma"]})
    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model="whatever",
            engine="in_process",
        )
    )
    assert result["answer"].to_list() == ["echo:alpha", "echo:beta", "echo:gamma"]


def test_inference_local_system_prefix_and_nulls():
    register_engine("sys-model", FakeEngine("sys-model"), engine="in_process")

    df = pl.DataFrame({"prompt": ["question", None]})
    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model="sys-model",
            system="You are helpful.",
            engine="in_process",
        )
    )
    values = result["answer"].to_list()
    # System prefix is folded in immutable-prefix-first.
    assert values[0] == "echo:You are helpful.\n\nquestion"
    # Null input -> null output (row order preserved).
    assert values[1] is None


def test_inference_local_error_row_isolated_end_to_end():
    register_engine("err-model", FakeEngine("err-model"), engine="in_process")

    df = pl.DataFrame({"prompt": ["ok-1", f"boom-{FAKE_FAIL_MARKER}", "ok-3"]})
    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model="err-model",
            engine="in_process",
        )
    )
    values = result["answer"].to_list()
    assert values[0] == "echo:ok-1"
    assert values[2] == "echo:ok-3"
    assert json.loads(values[1])["_error"] == "local_generation_error"


def test_inference_local_row_order_large_column():
    register_engine("big-model", FakeEngine("big-model"), engine="in_process")

    n = 1000
    df = pl.DataFrame({"prompt": [f"row-{i}" for i in range(n)]})
    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model="big-model",
            engine="in_process",
        )
    )
    assert result["answer"].to_list() == [f"echo:row-{i}" for i in range(n)]


def test_inference_local_unknown_engine_raises():
    with pytest.raises(ValueError):
        inference_local(pl.col("prompt"), model="m", engine="nonsense")
