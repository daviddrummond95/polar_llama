"""Tests for codebook induction (issue #78): `induce_codebook`,
`apply_codebook`, `codebook_to_taxonomy`, and the strict-schema safety of
the generated response models.

`inference_messages` is monkeypatched with a deterministic, keyword-based
fake so these tests exercise the *real* prompt-building, dtype-decoding,
dedupe, and taxonomy-bridging logic without any network call or API key --
they must never skip. A separate, real-API smoke test at the bottom is
gated on `OPENAI_API_KEY`.
"""

import json
import os
import re

import polars as pl
import pytest

import polar_llama
from polar_llama import (
    Codebook,
    CodebookEntry,
    apply_codebook,
    cluster_embeddings,
    codebook_to_taxonomy,
    dedupe_codebook_entries,
    induce_codebook,
)
from polar_llama.codebook import (
    _cluster_code_model,
    _code_application_model,
)


# ============================================================================
# Mock inference_messages: keyword-based, deterministic, no network
# ============================================================================


def _extract_messages(raw: str):
    return json.loads(raw)


def _mock_response_for(messages, response_model) -> dict:
    field_names = set(response_model.model_fields.keys())
    system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")

    if field_names == {"code", "definition", "rationale"}:
        # Cluster-naming mock: keyword-derive a code from the exemplar text.
        # Both "billing" clusters intentionally map to the same code, to
        # exercise cross-cluster dedupe.
        text = user_content.lower()
        if "billing" in text:
            code = "billing_issue"
        elif "login" in text:
            code = "login_issue"
        else:
            code = "other_issue"
        return {
            "code": code,
            "definition": f"Documents about {code.replace('_', ' ')}.",
            "rationale": "mocked rationale",
        }

    if field_names == {"applications"}:
        # Multi-label apply mock: recover candidate codes from the system
        # prompt's "- **code**: definition" lines, decide `applies` by
        # keyword overlap with the document.
        candidates = re.findall(r"- \*\*(.+?)\*\*: (.+)", system_content)
        applications = []
        for code, _definition in candidates:
            keyword = code.split("_")[0]
            applies = keyword in user_content.lower()
            applications.append(
                {
                    "code": code,
                    "applies": applies,
                    "confidence": 0.9 if applies else 0.1,
                    "evidence": user_content[:50],
                }
            )
        return {"applications": applications}

    raise AssertionError(f"mock: unexpected response_model field set {field_names!r}")


def _make_mock_inference_messages():
    """Build a fake `inference_messages(expr, *, response_model=...)` that
    decodes the *real* JSON schema (via the real `_pydantic_to_json_schema`
    / `_json_schema_to_polars_dtype` / `_parse_json_to_struct` helpers) so
    the struct dtype callers see matches production exactly -- only the
    network call is faked."""

    def fake_inference_messages(
        expr, *, provider=None, model=None, response_model=None, **_kw
    ):
        assert response_model is not None

        def generate(series: pl.Series) -> pl.Series:
            out = []
            for raw in series.to_list():
                if raw is None:
                    out.append(None)
                    continue
                messages = _extract_messages(raw)
                response = _mock_response_for(messages, response_model)
                response["_error"] = None
                response["_details"] = None
                response["_raw"] = json.dumps(response)
                out.append(json.dumps(response))
            return pl.Series(series.name, out, dtype=pl.Utf8)

        json_col = expr.map_batches(generate, return_dtype=pl.Utf8)

        schema = polar_llama._pydantic_to_json_schema(response_model)
        dtype = polar_llama._json_schema_to_polars_dtype(schema)
        return json_col.map_batches(
            lambda s: polar_llama._parse_json_to_struct(s, dtype), return_dtype=dtype
        )

    return fake_inference_messages


@pytest.fixture
def mock_inference(monkeypatch):
    """Patch `polar_llama.inference_messages` -- `induce_codebook` /
    `apply_codebook` import it lazily *by name* from the `polar_llama`
    package at call time, so patching the package attribute is sufficient
    (see `polar_llama/codebook.py`'s module docstring)."""
    fake = _make_mock_inference_messages()
    monkeypatch.setattr(polar_llama, "inference_messages", fake)
    return fake


# ============================================================================
# Synthetic data: three embedding blobs, two of which share a keyword so
# their induced codes collide (exercises dedupe).
# ============================================================================


def _synthetic_df(n_per_cluster: int = 5) -> pl.DataFrame:
    import random

    rng = random.Random(0)

    def blob(cx, cy, n, spread=0.3):
        return [
            [cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread)]
            for _ in range(n)
        ]

    texts = (
        [
            f"Billing problem #{i}: charged twice for the subscription."
            for i in range(n_per_cluster)
        ]
        + [
            f"Billing dispute #{i}: invoice amount looks wrong."
            for i in range(n_per_cluster)
        ]
        + [
            f"Cannot login #{i}: password reset link is broken."
            for i in range(n_per_cluster)
        ]
    )
    embeddings = (
        blob(20.0, 0.0, n_per_cluster)
        + blob(-20.0, 0.0, n_per_cluster)
        + blob(0.0, 20.0, n_per_cluster)
    )
    return pl.DataFrame({"text": texts, "emb": embeddings})


# ============================================================================
# induce_codebook
# ============================================================================


def test_induce_codebook_schema_and_exemplar_counts(mock_inference):
    df = _synthetic_df(n_per_cluster=5)
    result = induce_codebook(
        df, "text", embedding_column="emb", k=3, n_exemplars=5, seed=0
    )

    # DataFrame-shaped: same row count, cluster columns appended.
    assert result.df.height == df.height
    assert "cluster_id" in result.df.columns
    assert "cluster_distance" in result.df.columns
    assert result.df["text"].to_list() == df["text"].to_list()

    # Three raw clusters induced, but the two "billing" clusters collide on
    # the same code and get merged -> two final codes.
    assert len(result.codebook) == 2
    codes = set(result.codebook.codes)
    assert codes == {"billing_issue", "login_issue"}

    billing_entry = next(e for e in result.codebook if e.code == "billing_issue")
    login_entry = next(e for e in result.codebook if e.code == "login_issue")

    assert billing_entry.size == 10  # 5 + 5 merged
    assert set(billing_entry.cluster_ids) == {
        result.df.filter(pl.col("text").str.starts_with("Billing problem"))[
            "cluster_id"
        ][0],
        result.df.filter(pl.col("text").str.starts_with("Billing dispute"))[
            "cluster_id"
        ][0],
    }
    assert login_entry.size == 5
    assert len(login_entry.cluster_ids) == 1

    # Exemplars merged from two sub-clusters of 5 unique texts each, capped
    # at max(2*n_exemplars, n_exemplars) = 10.
    assert 5 < len(billing_entry.exemplars) <= 10
    assert len(login_entry.exemplars) <= 5


def test_induce_codebook_k_override_and_determinism(mock_inference):
    df = _synthetic_df(n_per_cluster=4)
    r1 = induce_codebook(df, "text", embedding_column="emb", k=3, seed=5)
    r2 = induce_codebook(df, "text", embedding_column="emb", k=3, seed=5)

    assert r1.df["cluster_id"].to_list() == r2.df["cluster_id"].to_list()
    assert sorted(r1.codebook.codes) == sorted(r2.codebook.codes)


def test_induce_codebook_computes_embeddings_when_not_given(
    monkeypatch, mock_inference
):
    """Without `embedding_column=`, induce_codebook calls `embedding_async`
    -- also mocked here so the test stays network-free."""

    def fake_embedding_async(expr, *, provider=None, model=None):
        # Deterministic fixed-length embedding derived from text content,
        # placed in one of two well-separated directions by keyword.
        def generate(series: pl.Series) -> pl.Series:
            out = []
            for text in series.to_list():
                if text is None:
                    out.append(None)
                elif "billing" in text.lower():
                    out.append([20.0, 0.0])
                else:
                    out.append([0.0, 20.0])
            return pl.Series(series.name, out, dtype=pl.List(pl.Float64))

        return expr.map_batches(generate, return_dtype=pl.List(pl.Float64))

    monkeypatch.setattr(polar_llama, "embedding_async", fake_embedding_async)

    df = pl.DataFrame(
        {
            "text": [
                "Billing problem: double charge",
                "Billing dispute: wrong invoice",
                "Cannot login: broken link",
                "Cannot login: expired token",
            ]
        }
    )
    result = induce_codebook(df, "text", k=2, seed=0)
    assert result.df.height == 4
    assert set(result.codebook.codes) <= {"billing_issue", "login_issue", "other_issue"}


def test_induce_codebook_raises_on_empty_dataframe():
    with pytest.raises(ValueError):
        induce_codebook(
            pl.DataFrame({"text": [], "emb": []}), "text", embedding_column="emb"
        )


def test_induce_codebook_raises_on_missing_column(mock_inference):
    df = _synthetic_df()
    with pytest.raises(ValueError):
        induce_codebook(df, "nonexistent", embedding_column="emb")


# ============================================================================
# apply_codebook: zero-reshaping round trip
# ============================================================================


def test_apply_codebook_zero_reshaping_round_trip(mock_inference):
    df = _synthetic_df(n_per_cluster=3)
    result = induce_codebook(df, "text", embedding_column="emb", k=3, seed=0)

    # No join, no explode -- apply_codebook composes directly onto the
    # DataFrame induce_codebook returned via an ordinary with_columns.
    coded = result.df.with_columns(
        labels=apply_codebook(pl.col("text"), result.codebook)
    )

    assert coded.height == result.df.height
    assert coded["text"].to_list() == result.df["text"].to_list()

    labels = coded["labels"].to_list()
    n_codes = len(result.codebook)
    for row_labels in labels:
        assert len(row_labels) == n_codes
        seen_codes = {entry["code"] for entry in row_labels}
        assert seen_codes == set(result.codebook.codes)
        for entry in row_labels:
            assert isinstance(entry["applies"], bool)
            assert 0.0 <= entry["confidence"] <= 1.0

    # Row talking about billing should have the billing code applied.
    billing_row = next(r for r in labels[:3])
    applied_codes = {e["code"] for e in billing_row if e["applies"]}
    assert "billing_issue" in applied_codes


def test_apply_codebook_accepts_plain_dict_codebook(mock_inference):
    df = pl.DataFrame({"text": ["Billing problem here", "Cannot login at all"]})
    hand_codebook = [
        {"code": "billing_issue", "definition": "Billing-related complaints."},
        {"code": "login_issue", "definition": "Login/auth failures."},
    ]
    out = df.with_columns(labels=apply_codebook(pl.col("text"), hand_codebook))
    assert out["labels"].list.len().to_list() == [2, 2]


def test_apply_codebook_raises_on_empty_codebook():
    with pytest.raises(ValueError):
        apply_codebook(pl.col("text"), [])


# ============================================================================
# codebook_to_taxonomy bridge
# ============================================================================


def test_codebook_to_taxonomy_feeds_create_taxonomy_pydantic_model():
    entries = [
        CodebookEntry(
            code="billing_issue",
            definition="Billing-related complaints.",
            rationale="",
            cluster_ids=[0],
            size=5,
            exemplars=["a"],
        ),
        CodebookEntry(
            code="login_issue",
            definition="Login/auth failures.",
            rationale="",
            cluster_ids=[1],
            size=5,
            exemplars=["b"],
        ),
    ]
    codebook = Codebook(entries)

    taxonomy = codebook_to_taxonomy(codebook, field_name="topic")
    assert set(taxonomy.keys()) == {"topic"}
    assert taxonomy["topic"]["values"] == {
        "billing_issue": "Billing-related complaints.",
        "login_issue": "Login/auth failures.",
    }

    # Must round-trip cleanly through the real taxonomy model builder.
    model = polar_llama._create_taxonomy_pydantic_model(taxonomy)
    assert "topic" in model.model_fields


def test_codebook_to_taxonomy_rejects_duplicate_codes():
    entries = [
        CodebookEntry("dup", "d1", "", [0], 1, []),
        CodebookEntry("dup", "d2", "", [1], 1, []),
    ]
    with pytest.raises(ValueError):
        codebook_to_taxonomy(Codebook(entries))


def test_codebook_to_taxonomy_rejects_empty_codebook():
    with pytest.raises(ValueError):
        codebook_to_taxonomy(Codebook([]))


# ============================================================================
# dedupe_codebook_entries (pure function, no mock needed)
# ============================================================================


def test_dedupe_codebook_entries_merges_case_insensitive_collisions():
    entries = [
        CodebookEntry("Billing_Issue", "def a", "rat a", [0], 3, ["x1", "x2"]),
        CodebookEntry("billing_issue", "def a", "rat b", [1], 4, ["x2", "x3"]),
        CodebookEntry("  Login_Issue  ", "def c", "rat c", [2], 2, ["y1"]),
    ]
    merged = dedupe_codebook_entries(entries)
    assert len(merged) == 2
    billing = next(e for e in merged if _norm(e.code) == "billing_issue")
    assert billing.size == 7
    assert set(billing.cluster_ids) == {0, 1}
    assert billing.exemplars == ["x1", "x2", "x3"]  # deduped, order preserved


def _norm(code: str) -> str:
    return " ".join(code.strip().lower().split())


def test_dedupe_codebook_entries_no_collision_is_a_noop():
    entries = [
        CodebookEntry("a", "da", "ra", [0], 1, []),
        CodebookEntry("b", "db", "rb", [1], 1, []),
    ]
    merged = dedupe_codebook_entries(entries)
    assert len(merged) == 2


# ============================================================================
# Codebook container behavior
# ============================================================================


def test_codebook_container_protocol():
    entries = [
        CodebookEntry("a", "da", "ra", [0], 1, ["x"]),
        CodebookEntry("b", "db", "rb", [1], 1, ["y"]),
    ]
    codebook = Codebook(entries)
    assert len(codebook) == 2
    assert codebook[0].code == "a"
    assert [e.code for e in codebook] == ["a", "b"]
    assert codebook.codes == ["a", "b"]
    dicts = codebook.to_dicts()
    assert dicts[0]["code"] == "a"
    df = codebook.to_polars()
    assert df["code"].to_list() == ["a", "b"]


# ============================================================================
# Strict-schema compliance (issue #51 lesson)
# ============================================================================


def _walk_assert_strict_compliant(node):
    """Recursively assert every object node in a JSON schema is OpenAI
    strict-mode compliant. Mirrors `tests/test_taxonomy_strict_schema.py`'s
    walker (kept local/self-contained rather than imported cross-file)."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert "properties" in node, f"object node missing properties: {node}"
            assert node.get("additionalProperties") is False, (
                f"object node without additionalProperties=false: {node}"
            )
            assert "required" in node and sorted(node["required"]) == sorted(
                node["properties"].keys()
            ), (
                f"required != property keys: {node.get('required')} vs {list(node['properties'])}"
            )
        assert not isinstance(node.get("additionalProperties"), dict), (
            f"dynamic map object leaked into schema: {node}"
        )
        if "$ref" in node:
            assert set(node.keys()) == {"$ref"}, (
                f"$ref node has sibling keywords: {node}"
            )
        for value in node.values():
            _walk_assert_strict_compliant(value)
    elif isinstance(node, list):
        for item in node:
            _walk_assert_strict_compliant(item)


def test_apply_codebook_response_model_is_strict_compliant():
    """The multi-label apply model is a List of fixed-key structs -- never a
    Dict/dynamic-key object -- so it must pass the same strict-mode walk
    that would have caught issue #51."""
    model = _code_application_model()
    schema = polar_llama._pydantic_to_json_schema(model)
    _walk_assert_strict_compliant(schema)

    # And explicitly: `applications` must be a List, never a Dict keyed by code.
    props = schema["properties"]
    assert props["applications"]["type"] == "array"


def test_cluster_code_model_is_strict_compliant():
    model = _cluster_code_model()
    schema = polar_llama._pydantic_to_json_schema(model)
    _walk_assert_strict_compliant(schema)
    assert set(schema["properties"].keys()) == {"code", "definition", "rationale"}


def test_apply_codebook_response_model_scales_with_codebook_size():
    """The schema *shape* must not depend on how many codes are in the
    codebook (only prompt content does) -- List[...] has no fixed item
    count, so a 3-code and a 30-code codebook produce byte-identical
    schemas."""
    model = _code_application_model()
    schema = polar_llama._pydantic_to_json_schema(model)
    # No enum/const tying the schema to a specific set of codes.
    assert "enum" not in json.dumps(schema)


# ============================================================================
# Real-API smoke test (gated; never runs in CI without a key)
# ============================================================================


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_induce_and_apply_codebook_real_api():
    from polar_llama import Provider, embedding_async

    texts = [
        "The server crashed at 3am and customers could not check out.",
        "Checkout page returned a 500 error during peak traffic.",
        "I love the new dashboard, it's so much faster now!",
        "Great redesign, the dashboard finally loads instantly.",
    ]
    df = pl.DataFrame({"text": texts}).with_columns(
        emb=embedding_async(pl.col("text"), provider=Provider.OPENAI)
    )

    result = induce_codebook(
        df,
        "text",
        embedding_column="emb",
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        k=2,
    )
    assert len(result.codebook) >= 1
    assert result.df.height == len(texts)

    coded = result.df.with_columns(
        labels=apply_codebook(
            pl.col("text"),
            result.codebook,
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
        )
    )
    assert coded.height == len(texts)
    assert all(len(row) == len(result.codebook) for row in coded["labels"].to_list())
