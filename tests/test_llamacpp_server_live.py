"""Live-endpoint tests for the ``engine="server"`` local backend against a
REAL llama.cpp ``llama-server`` process (issue #84, Tier 1).

Gated on ``POLAR_LLAMA_LLAMACPP_URL``, set by the ``llamacpp_server_test`` CI
job (``.github/workflows/CI.yml``) once it has downloaded a pinned llama.cpp
release build, downloaded a pinned tiny GGUF, started ``llama-server``, and
polled ``/health`` until it answered 200. Skipped everywhere else -- this
module never downloads, starts, or manages a server itself.

This is the live-endpoint sibling of ``tests/test_local_server_backend.py``,
which stands up an in-process fake OpenAI-compatible HTTP server and always
runs (no gate, no external binary). That file exercises the Rust async
fan-out's request/response plumbing against a *controlled* mock; this file
exercises the exact same code path (``inference_local(engine="server")`` ->
``inference_local_server`` -> ``inference_async``/``inference_messages`` with
``OPENAI_BASE_URL`` retargeted, see ``polar_llama/local/server_backend.py``)
against llama.cpp's *real* ``/v1/chat/completions`` implementation, including
its real ``usage`` block.

These tests assert SHAPE, not content: the CI job's pinned model is a tiny
(~135M-parameter) instruction model with no strong instruction-following
guarantees, so asserting exact wording would be flaky. What's asserted is row
count, non-null/non-empty String completions, the system-prompt code path,
and that ``usage=True`` decodes a real usage block -- never the
``{"_error": ...}`` in-band error shape (``create_error_response``,
``src/model_client/mod.rs``) that a failed row would produce instead.

Run locally (no CI needed) by starting a real ``llama-server`` yourself::

    llama-server -m /path/to/some-tiny-model.gguf --port 8080
    POLAR_LLAMA_LLAMACPP_URL=http://127.0.0.1:8080 \\
        uv run pytest tests/test_llamacpp_server_live.py -v
"""

from __future__ import annotations

import os
import warnings

import polars as pl
import pytest

from polar_llama.local import inference_local

LLAMACPP_URL = os.environ.get("POLAR_LLAMA_LLAMACPP_URL")

pytestmark = pytest.mark.skipif(
    not LLAMACPP_URL,
    reason=(
        "POLAR_LLAMA_LLAMACPP_URL is not set -- this test only runs against "
        "a real llama.cpp llama-server. In CI this is started by the "
        "llamacpp_server_test job (.github/workflows/CI.yml). Locally: "
        "start `llama-server -m <model.gguf> --port 8080`, then run "
        "`POLAR_LLAMA_LLAMACPP_URL=http://127.0.0.1:8080 pytest "
        "tests/test_llamacpp_server_live.py`."
    ),
)

# llama-server routes purely by whichever single GGUF it was started with --
# the request body's `model` field is only ever echoed back in the response,
# never used to pick a model server-side -- so any non-empty string is a
# valid model name here.
_MODEL = "local-llamacpp-gguf"


def _looks_like_error_envelope(value: str) -> bool:
    """True if `value` is the `{"_error": ...}` in-band error shape
    (`create_error_response`, `src/model_client/mod.rs`) rather than a real
    completion string.
    """
    return value.lstrip().startswith("{") and '"_error"' in value


@pytest.fixture(autouse=True)
def _isolated_openai_base_url(monkeypatch):
    # OPENAI_BASE_URL is a process-global env var, read per-request by the
    # Rust client (src/model_client/openai.rs). Start every test clean so
    # nothing leaks in from -- or out to -- other test modules.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    yield


def test_completions_have_correct_shape_and_no_errors():
    prompts = [
        "Reply with the single word: hello.",
        "What is 2 + 2? Reply with just the number.",
        "Name a primary color.",
    ]
    df = pl.DataFrame({"prompt": prompts})

    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model=_MODEL,
            engine="server",
            base_url=LLAMACPP_URL,
        )
    )

    answers = result["answer"].to_list()
    assert len(answers) == len(prompts)
    for val in answers:
        assert val is not None
        assert isinstance(val, str)
        assert val != ""
        assert not _looks_like_error_envelope(val)


def test_system_prompt_case():
    prompts = ["The sky is blue.", "Water boils at 100C at sea level."]
    df = pl.DataFrame({"prompt": prompts})

    result = df.with_columns(
        answer=inference_local(
            pl.col("prompt"),
            model=_MODEL,
            engine="server",
            base_url=LLAMACPP_URL,
            system="You are a terse assistant.",
        )
    )

    answers = result["answer"].to_list()
    assert len(answers) == len(prompts)
    for val in answers:
        assert val is not None
        assert isinstance(val, str)
        assert val != ""
        assert not _looks_like_error_envelope(val)


def test_usage_true_returns_real_usage_block():
    df = pl.DataFrame({"prompt": ["Say hello."]})

    with warnings.catch_warnings():
        # A local GGUF model name isn't in the packaged price table, so a
        # one-time "unknown model" cost-resolution warning is expected here
        # and is not the point of this test -- it cares about the usage
        # block's token/latency fields, which come from llama-server's own
        # real response, not from cost resolution.
        warnings.simplefilter("ignore")
        result = df.with_columns(
            answer=inference_local(
                pl.col("prompt"),
                model=_MODEL,
                engine="server",
                base_url=LLAMACPP_URL,
                usage=True,
            )
        )

    row = result["answer"][0]
    assert row is not None
    response_text = row["response"]
    usage = row["usage"]

    assert isinstance(response_text, str)
    assert response_text != ""
    assert not _looks_like_error_envelope(response_text)

    # llama-server's real `/v1/chat/completions` usage block --
    # `prompt_tokens` / `completion_tokens` /
    # `prompt_tokens_details.cached_tokens` -- decoded by
    # `src/model_client/openai.rs::parse_usage` into
    # `input_tokens`/`output_tokens`/`cached_tokens`. Verified manually
    # against a real `llama-server` (b10004): both counts are > 0 for any
    # non-trivial prompt.
    assert usage["input_tokens"] is not None and usage["input_tokens"] > 0
    assert usage["output_tokens"] is not None and usage["output_tokens"] > 0
    assert usage["latency_ms"] is not None and usage["latency_ms"] >= 0
    # cost_usd is null (local model name isn't in the packaged price table)
    # unless the caller registers a price -- both shapes are acceptable here.
    assert usage["cost_usd"] is None or isinstance(usage["cost_usd"], float)
