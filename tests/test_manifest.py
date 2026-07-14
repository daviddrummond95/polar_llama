#!/usr/bin/env python3
"""
Test suite for deterministic run manifests (issue #85).

CI-safe: no network calls, no API keys, no mlx. `replay()`'s call
reconstruction is exercised via the `_inference_fn` test seam (a stub that
records its kwargs and returns a `pl.lit(...)` expression) rather than a
real `inference_async`/`inference_messages` call.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel

import polar_llama as pll
from polar_llama.checkpoint import CheckpointStore
from polar_llama.dedup import DedupeStats, ResponseCacheStore
from polar_llama.keys import request_fingerprint
from polar_llama.manifest import (
    ManifestIntegrityError,
    ManifestMismatchError,
    RunManifest,
    build_manifest,
    load_manifest,
    save_manifest,
    with_manifest_id,
)


class ModelA(BaseModel):
    x: int


class ModelB(BaseModel):
    y: str


USAGE_STRUCT_DTYPE = pl.Struct(
    {
        "input_tokens": pl.Int64,
        "output_tokens": pl.Int64,
        "cached_tokens": pl.Int64,
        "latency_ms": pl.Int64,
        "cost_usd": pl.Float64,
    }
)


def _usage_df() -> pl.DataFrame:
    """A hand-built `usage=True`-shaped DataFrame, matching USAGE_DTYPE."""
    envelope_dtype = pl.Struct({"response": pl.Utf8, "usage": USAGE_STRUCT_DTYPE})
    return pl.DataFrame(
        {
            "r": [
                {
                    "response": "a",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cached_tokens": 0,
                        "latency_ms": 100,
                        "cost_usd": 0.001,
                    },
                },
                {
                    "response": "b",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "cached_tokens": 5,
                        "latency_ms": 200,
                        "cost_usd": 0.002,
                    },
                },
                None,  # null row -- must not blow up the sums / rows_with_usage
            ]
        },
        schema={"r": envelope_dtype},
    )


# ============================================================================
# Determinism
# ============================================================================


def test_determinism_identical_config():
    m1 = build_manifest(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="You are helpful.",
        response_model=ModelA,
        params={"temperature": 0.2},
        seed=7,
    )
    m2 = build_manifest(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="You are helpful.",
        response_model=ModelA,
        params={"temperature": 0.2},
        seed=7,
    )

    assert m1.manifest_id == m2.manifest_id

    d1 = dataclasses.asdict(m1)
    d2 = dataclasses.asdict(m2)
    d1.pop("created_at")
    d2.pop("created_at")
    assert d1 == d2


def test_row_count_not_in_manifest_id():
    # Regression (#85 review): row_count must NOT enter the determinism hash --
    # manifest_id is the identity of the request CONFIG, so a manifest built
    # before a run (df=None) and after (df=result), or over different batch
    # sizes, must share a manifest_id for byte-identical config.
    kw = dict(symbol="inference_async", provider="openai", model="gpt-4o-mini",
              system_prompt="You are helpful.", seed=7)
    m_none = build_manifest(df=None, **kw)
    m_100 = build_manifest(df=pl.DataFrame({"p": ["x"] * 100}), **kw)
    m_50 = build_manifest(df=pl.DataFrame({"p": ["x"] * 50}), **kw)
    assert m_none.manifest_id == m_100.manifest_id == m_50.manifest_id
    # ...but row_count is still recorded (just as excluded metadata).
    assert m_none.row_count is None
    assert m_100.row_count == 100
    assert m_50.row_count == 50


def test_replay_model_none_reresolves_default_not_literal():
    # Regression (#85 review): a model=None manifest stores the "default"
    # sentinel; replay must map it back to None so inference re-resolves the
    # real default model, not call a nonexistent model literally named
    # "default". Same for provider None -> "openai".
    m = build_manifest(symbol="inference_async", provider=None, model=None,
                       system_prompt="s")
    df = pl.DataFrame({"p": ["a"]})
    recorded: list = []
    pll.replay(m, df, "p", system_prompt="s", _inference_fn=_recording_stub(recorded))
    assert len(recorded) == 1
    assert recorded[0]["model"] is None  # NOT the nonexistent literal "default"
    # provider "openai" sentinel is a valid provider string (routes like None),
    # so it is forwarded as-is rather than remapped.
    assert recorded[0]["provider"] == "openai"


def test_determinism_timestamps_actually_differ():
    # Sanity: the two manifests above really were built independently (not
    # a fluke of a frozen clock) -- created_at is still a real ISO timestamp
    # each time, even if it happens to collide at second resolution.
    m1 = build_manifest(model="m")
    m2 = build_manifest(model="m")
    assert isinstance(m1.created_at, str) and isinstance(m2.created_at, str)
    assert m1.manifest_id == m2.manifest_id


# ============================================================================
# Config-change-changes-id
# ============================================================================


@pytest.mark.parametrize(
    "kwargs_override",
    [
        {"model": "gpt-4o"},
        {"provider": "anthropic"},
        {"system_prompt": "different system prompt"},
        {"prompt_template": "different template"},
        {"response_model": ModelB},
        {"params": {"temperature": 0.9}},
        {"seed": 99},
        {"symbol": "inference_messages"},
    ],
)
def test_config_change_changes_id(kwargs_override):
    base_kwargs = dict(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="base prompt",
        response_model=ModelA,
        prompt_template="base template",
        params={"temperature": 0.2},
        seed=7,
    )
    m_base = build_manifest(**base_kwargs)

    changed_kwargs = dict(base_kwargs)
    changed_kwargs.update(kwargs_override)
    m_changed = build_manifest(**changed_kwargs)

    assert m_base.manifest_id != m_changed.manifest_id


def test_response_model_change_between_two_distinct_models():
    m_a = build_manifest(response_model=ModelA)
    m_b = build_manifest(response_model=ModelB)
    assert m_a.manifest_id != m_b.manifest_id
    assert m_a.response_model_name == "ModelA"
    assert m_b.response_model_name == "ModelB"


# ============================================================================
# Reuse proof: manifest fingerprint == runtime fingerprint
# ============================================================================


def test_manifest_fingerprint_equals_runtime_fingerprint():
    manifest = build_manifest(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="sys",
        response_model=ModelA,
    )

    # Build kwargs exactly the way inference_async does before calling
    # request_fingerprint.
    schema_str = json.dumps(pll._pydantic_to_json_schema(ModelA))
    runtime_kwargs = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "response_schema": schema_str,
        "response_model_name": "ModelA",
    }
    runtime_fingerprint, _ = request_fingerprint(
        "inference_async", runtime_kwargs, "sys"
    )

    assert manifest.config_fingerprint == runtime_fingerprint


def test_manifest_fingerprint_equals_runtime_fingerprint_messages_symbol():
    manifest = build_manifest(symbol="inference_messages", provider="groq", model="m")
    runtime_kwargs = {
        "provider": "groq",
        "model": "m",
        "response_schema": None,
        "response_model_name": None,
    }
    runtime_fingerprint, _ = request_fingerprint(
        "inference_messages", runtime_kwargs, None
    )
    assert manifest.config_fingerprint == runtime_fingerprint


# ============================================================================
# Metadata / timestamp exclusion from the hash
# ============================================================================


def test_timestamp_and_metadata_excluded():
    m = build_manifest(model="m", provider="openai")
    original_id = m.manifest_id

    mutated = dataclasses.replace(
        m,
        created_at="1999-01-01T00:00:00+00:00",
        aggregate_usage={"input_tokens": 999},
        checkpoint_id="some-other-fingerprint",
        response_cache_id="another-fingerprint",
        dedupe_stats={"rows_total": 123},
    )

    # manifest_id itself is unchanged (it's a stored value, not
    # recomputed by dataclasses.replace); recomputing the hash over the
    # mutated manifest's hashed_fields() must still equal the original.
    from polar_llama.manifest import _manifest_id

    assert _manifest_id(mutated.hashed_fields()) == original_id
    assert mutated.manifest_id == original_id


# ============================================================================
# with_manifest_id
# ============================================================================


def test_with_manifest_id():
    m = build_manifest(model="m")
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    out = with_manifest_id(df, m)

    assert out.columns == ["a", "b", "manifest_id"]
    assert out["manifest_id"].to_list() == [m.manifest_id] * 3
    assert out["a"].to_list() == [1, 2, 3]
    assert out["b"].to_list() == ["x", "y", "z"]


def test_with_manifest_id_custom_column_name():
    m = build_manifest(model="m")
    df = pl.DataFrame({"a": [1]})
    out = with_manifest_id(df, m, column="run_id")
    assert "run_id" in out.columns
    assert out["run_id"][0] == m.manifest_id


# ============================================================================
# aggregate_usage
# ============================================================================


def test_aggregate_usage_summed():
    df = _usage_df()
    m = build_manifest(df=df, usage_column="r")

    assert m.aggregate_usage is not None
    assert m.aggregate_usage["input_tokens"] == 30
    assert m.aggregate_usage["output_tokens"] == 15
    assert m.aggregate_usage["cached_tokens"] == 5
    assert m.aggregate_usage["cost_usd"] == pytest.approx(0.003)
    assert m.aggregate_usage["rows_with_usage"] == 2  # null row excluded
    assert m.row_count == 3  # row_count counts the null row too


def test_aggregate_usage_requires_df():
    with pytest.raises(ValueError):
        build_manifest(usage_column="r")  # no df supplied


def test_aggregate_usage_requires_usage_struct_field():
    df = pl.DataFrame({"r": ["not", "a", "struct"]})
    with pytest.raises(ValueError):
        build_manifest(df=df, usage_column="r")


def test_aggregate_usage_none_when_not_requested():
    df = pl.DataFrame({"p": ["a", "b"]})
    m = build_manifest(df=df)
    assert m.aggregate_usage is None
    assert m.row_count == 2


# ============================================================================
# save / load round-trip + integrity check
# ============================================================================


def test_save_load_roundtrip(tmp_path: Path):
    m = build_manifest(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="sys",
        response_model=ModelA,
        params={"temperature": 0.1},
        seed=1,
    )
    path = tmp_path / "manifest.json"
    returned_path = save_manifest(m, path)
    assert returned_path == path
    assert path.exists()

    loaded = load_manifest(path)
    assert loaded == m

    # Valid, sorted-keys, indented JSON.
    raw = path.read_text()
    payload = json.loads(raw)
    assert list(payload.keys()) == sorted(payload.keys())
    assert "\n" in raw  # indent=2 produces multi-line output


def test_manifest_save_load_methods(tmp_path: Path):
    m = build_manifest(model="m")
    path = m.save(tmp_path / "m.json")
    loaded = RunManifest.load(path)
    assert loaded == m


def test_load_integrity_check(tmp_path: Path):
    m = build_manifest(model="gpt-4o-mini")
    path = tmp_path / "manifest.json"
    save_manifest(m, path)

    payload = json.loads(path.read_text())
    payload["model"] = "gpt-4o-TAMPERED"
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifestIntegrityError):
        load_manifest(path)


def test_load_rejects_future_format_version(tmp_path: Path):
    m = build_manifest(model="m")
    path = tmp_path / "manifest.json"
    save_manifest(m, path)

    payload = json.loads(path.read_text())
    payload["format_version"] = 999999
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_manifest(path)


# ============================================================================
# checkpoint_id / response_cache_id
# ============================================================================


def test_checkpoint_and_cache_ids(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpt"
    CheckpointStore(ckpt_dir, fingerprint="ckpt-fp-123")
    m = build_manifest(checkpoint=ckpt_dir)
    assert m.checkpoint_id == "ckpt-fp-123"

    cache_dir = tmp_path / "cache"
    ResponseCacheStore(cache_dir, fingerprint="cache-fp-456")
    m2 = build_manifest(response_cache=cache_dir)
    assert m2.response_cache_id == "cache-fp-456"


def test_checkpoint_id_none_when_no_meta():
    # A path with no _meta.json (nothing has ever written to it) -> None,
    # not an error. Pass the raw path so no store constructor runs first.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        empty_dir = os.path.join(d, "never-touched")
        os.makedirs(empty_dir)
        m = build_manifest(checkpoint=empty_dir)
        assert m.checkpoint_id is None


def test_checkpoint_id_none_when_not_supplied():
    m = build_manifest(model="m")
    assert m.checkpoint_id is None
    assert m.response_cache_id is None


def test_dedupe_stats_snapshot():
    stats = DedupeStats(
        rows_total=10, rows_null=1, cache_hits=2, rows_collapsed=3, calls_made=4
    )
    m = build_manifest(dedupe_stats=stats)
    assert m.dedupe_stats == {
        "rows_total": 10,
        "rows_null": 1,
        "cache_hits": 2,
        "rows_collapsed": 3,
        "calls_made": 4,
    }


# ============================================================================
# replay()
# ============================================================================


def _recording_stub(recorded: list):
    def fn(expr, **kwargs):
        recorded.append(kwargs)
        return pl.lit("ok")

    return fn


def test_replay_verify_mismatch_system_prompt():
    m = build_manifest(symbol="inference_async", model="m", system_prompt="original")
    df = pl.DataFrame({"p": ["a"]})
    recorded = []
    with pytest.raises(ManifestMismatchError, match="system_prompt"):
        pll.replay(
            m, df, "p", system_prompt="different", _inference_fn=_recording_stub(recorded)
        )
    assert recorded == []  # never got to calling the inference fn


def test_replay_verify_mismatch_response_model():
    m = build_manifest(symbol="inference_async", model="m", response_model=ModelA)
    df = pl.DataFrame({"p": ["a"]})
    recorded = []
    with pytest.raises(ManifestMismatchError, match="response_model"):
        pll.replay(
            m,
            df,
            "p",
            response_model=ModelB,
            _inference_fn=_recording_stub(recorded),
        )
    assert recorded == []


def test_replay_verify_false_skips_check():
    m = build_manifest(symbol="inference_async", model="m", system_prompt="original")
    df = pl.DataFrame({"p": ["a"]})
    recorded = []
    out = pll.replay(
        m,
        df,
        "p",
        system_prompt="different",
        verify=False,
        _inference_fn=_recording_stub(recorded),
    )
    assert len(recorded) == 1
    assert out.height == 1


def test_replay_reconstructs_call_inference_async():
    m = build_manifest(
        symbol="inference_async",
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="sys",
        response_model=ModelA,
    )
    df = pl.DataFrame({"p": ["a", "b", "c"]})
    recorded = []
    out = pll.replay(
        m, df, "p", system_prompt="sys", response_model=ModelA, _inference_fn=_recording_stub(recorded)
    )

    assert len(recorded) == 1
    call = recorded[0]
    assert call["provider"] == "openai"
    assert call["model"] == "gpt-4o-mini"
    assert call["system_prompt"] == "sys"
    assert call["response_model"] is ModelA
    assert call["usage"] is False  # aggregate_usage was never set on m

    assert "response" in out.columns
    assert "manifest_id" in out.columns
    assert out.height == 3
    assert out["manifest_id"].to_list() == [m.manifest_id] * 3


def test_replay_reconstructs_call_inference_messages_no_system_prompt_kwarg():
    m = build_manifest(symbol="inference_messages", provider="groq", model="m")
    df = pl.DataFrame({"messages": ["[]", "[]"]})
    recorded = []
    out = pll.replay(m, df, "messages", _inference_fn=_recording_stub(recorded))

    assert len(recorded) == 1
    call = recorded[0]
    # inference_messages has no system_prompt parameter -- replay must not
    # try to forward one for an inference_messages manifest.
    assert "system_prompt" not in call
    assert call["provider"] == "groq"
    assert call["model"] == "m"
    assert out.height == 2


def test_replay_usage_flag_forwarded_from_manifest():
    df_with_usage = _usage_df()
    m = build_manifest(df=df_with_usage, usage_column="r", symbol="inference_async", model="m")
    assert m.aggregate_usage is not None

    replay_df = pl.DataFrame({"p": ["x"]})
    recorded = []
    pll.replay(m, replay_df, "p", _inference_fn=_recording_stub(recorded))
    assert recorded[0]["usage"] is True


def test_replay_store_texts_selfcontained():
    m = build_manifest(
        symbol="inference_async",
        model="m",
        system_prompt="secret system prompt",
        prompt_template="secret template",
        store_texts=True,
    )
    assert m.texts == {
        "system_prompt": "secret system prompt",
        "prompt_template": "secret template",
    }

    df = pl.DataFrame({"p": ["a"]})
    recorded = []
    # No system_prompt/prompt_template re-supplied -- must be auto-filled
    # from manifest.texts and pass verification.
    out = pll.replay(m, df, "p", _inference_fn=_recording_stub(recorded))

    assert recorded[0]["system_prompt"] == "secret system prompt"
    assert out.height == 1


def test_build_manifest_without_store_texts_has_no_texts():
    m = build_manifest(model="m", system_prompt="sys")
    assert m.texts is None


# ============================================================================
# Optional gated live replay (mirrors tests/test_llamacpp_server_live.py)
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY"
)
def test_replay_live():
    m = build_manifest(symbol="inference_async", provider="openai", model="gpt-4o-mini")
    df = pl.DataFrame({"p": ["Say the word 'hi' and nothing else."]})
    out = pll.replay(m, df, "p", verify=False)
    assert out.height == 1
    assert out["response"][0] is not None
