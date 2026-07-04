"""CI-safe tests for the local-backend prefix-cache layer and parity harness.

No mlx, no GPU, no network: everything runs against FakeTokenizer/FakeEngine.
"""

import json

import pytest

from polar_llama.local.parity import (
    FAKE_DIVERGENCE_THRESHOLD,
    FakeEngine,
    FakeTokenizer,
    GenParams,
    compare_batched_vs_sequential,
    compare_cache_parity,
    measure_ttft,
    run_fake_parity_suite,
    token_divergence_rate,
)
from polar_llama.local.prefix_cache import (
    MatchClass,
    PrefixKey,
    PrefixStore,
    TrimNotSupportedError,
    assemble_prompt,
    classify,
    error_payload,
    verify_token_boundary,
)

pytestmark = pytest.mark.local


# ---------------------------------------------------------------------------
# classify(): EXACT / EXTEND / DIVERGE
# ---------------------------------------------------------------------------


def test_classify_exact():
    assert classify("sys\n", "sys\n") is MatchClass.EXACT


def test_classify_extend():
    # Cached is a strict prefix of new: the safe extend path.
    assert classify("sys\n", "sys\nfewshot\n") is MatchClass.EXTEND


def test_classify_diverge_different_text():
    assert classify("sys A\n", "sys B\n") is MatchClass.DIVERGE


def test_classify_longer_cache_is_diverge_not_trim():
    # New prompt is a prefix of the CACHE: the trim-tempting case.
    # Must be DIVERGE (mlx-lm #980: the trim path is broken).
    assert classify("sys\nfewshot\n", "sys\n") is MatchClass.DIVERGE


def test_classify_empty_cached_prefix_extends():
    assert classify("", "anything") is MatchClass.EXTEND


# ---------------------------------------------------------------------------
# verify_token_boundary(): the BPE seam check
# ---------------------------------------------------------------------------


def test_token_boundary_catches_bpe_merge_violation():
    # Vocab has "walking": encode("walk")=[walk] but
    # encode("walk"+"ing")=[walking] -- a merge across the seam. Raw-text
    # prefix match holds, token-level prefix does NOT.
    tok = FakeTokenizer(vocab=["walking", "walk", "ing"])
    assert "walk" + "ing" == "walking"  # text-level prefix match...
    assert verify_token_boundary(tok, "walk", "ing") is False  # ...token no


def test_token_boundary_passes_on_safe_separator():
    tok = FakeTokenizer(vocab=["walking", "walk", "ing", "\n"])
    # No vocab entry spans "\n", so ending the prefix at a newline is safe.
    assert verify_token_boundary(tok, "walk\n", "ing") is True


def test_token_boundary_empty_prefix_trivially_true():
    tok = FakeTokenizer()
    assert verify_token_boundary(tok, "", "whatever") is True


def test_token_boundary_matches_definition():
    # The check is exactly tokenize(prefix+suffix)[:len(tokenize(prefix))]
    # == tokenize(prefix).
    tok = FakeTokenizer(vocab=["ab", "a", "b", "c"])
    prefix, suffix = "a", "bc"  # "a"+"bc" -> ["ab","c"], prefix -> ["a"]
    full = tok.encode(prefix + suffix)
    pre = tok.encode(prefix)
    assert full[: len(pre)] != pre
    assert verify_token_boundary(tok, prefix, suffix) is False


# ---------------------------------------------------------------------------
# assemble_prompt(): immutable prefix first, structurally
# ---------------------------------------------------------------------------


def test_assemble_prompt_splits_at_cache_seam():
    parts = assemble_prompt("be terse", [("hi", "yo")], "what is 2+2?")
    assert parts.full == parts.prefix + parts.suffix
    assert "be terse" in parts.prefix
    assert "hi" in parts.prefix and "yo" in parts.prefix
    # Per-row content structurally cannot enter the prefix.
    assert "2+2" not in parts.prefix
    assert "2+2" in parts.suffix


def test_assemble_prompt_shared_prefix_yields_shared_key():
    a = assemble_prompt("sys", None, "row one")
    b = assemble_prompt("sys", None, "row two")
    assert a.prefix == b.prefix
    assert PrefixKey.from_text("m", a.prefix) == PrefixKey.from_text("m", b.prefix)
    c = assemble_prompt("other sys", None, "row one")
    assert PrefixKey.from_text("m", c.prefix) != PrefixKey.from_text("m", a.prefix)


def test_assemble_prompt_rejects_non_string_user():
    with pytest.raises(TypeError):
        assemble_prompt("sys", None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PrefixStore: match, no-trim policy, replication
# ---------------------------------------------------------------------------


def _store_with(prefix_text, tokenizer=None, **kwargs):
    engine = FakeEngine(tokenizer or FakeTokenizer())
    store = PrefixStore(tokenizer=engine.tokenizer, **kwargs)
    cache = engine.prefill(prefix_text)
    entry = store.put("m", prefix_text, cache, token_count=cache.token_count)
    return store, entry, engine


def test_store_match_exact_and_extend():
    store, entry, _ = _store_with("sys\n")
    cls, hit = store.match("m", "sys\n")
    assert cls is MatchClass.EXACT and hit is entry
    cls, hit = store.match("m", "sys\nmore\n")
    assert cls is MatchClass.EXTEND and hit is entry


def test_store_match_is_model_scoped():
    store, _, _ = _store_with("sys\n")
    cls, hit = store.match("other-model", "sys\n")
    assert cls is MatchClass.DIVERGE and hit is None


def test_no_trim_policy_skip_recomputes():
    # Cache holds a LONGER prefix than requested: default policy treats it
    # as a miss (recompute) rather than trimming.
    store, _, _ = _store_with("sys\nfewshot\n", on_longer_cache="skip")
    cls, hit = store.match("m", "sys\n")
    assert cls is MatchClass.DIVERGE and hit is None


def test_no_trim_policy_raise():
    store, _, _ = _store_with("sys\nfewshot\n", on_longer_cache="raise")
    with pytest.raises(TrimNotSupportedError):
        store.match("m", "sys\n")


def test_replicate_into_batch_row_aligned_and_independent():
    store, entry, _ = _store_with("sys\n")
    n = 5
    copies = entry.replicate_into_batch(n)
    assert len(copies) == n
    assert all(c.prefix_text == "sys\n" for c in copies)  # row-aligned
    # Independent: in-place decode mutation of one row must not leak.
    copies[0].consumed += 1
    copies[0].log.append("row0")
    assert all(c.consumed == 0 and c.log == [] for c in copies[1:])
    assert entry.cache_obj.consumed == 0 and entry.cache_obj.log == []


def test_checked_batch_caches_requires_boundary_on_every_row():
    tok = FakeTokenizer(vocab=["walking", "walk", "ing", "\n"])
    store, _, _ = _store_with("walk", tokenizer=tok)
    # One row's suffix merges across the seam -> reuse denied for the batch.
    assert store.checked_batch_caches("m", "walk", ["ing", "\n ok"]) is None
    # All rows boundary-safe -> row-aligned caches come back.
    caches = store.checked_batch_caches("m", "walk", ["\n a", "\n b"])
    assert caches is not None and len(caches) == 2


def test_store_lru_eviction():
    store = PrefixStore(tokenizer=FakeTokenizer(), max_entries=2)
    for i in range(3):
        store.put("m", f"p{i}\n", object(), token_count=1)
    assert len(store) == 2
    assert store.get_exact("m", "p0\n") is None  # oldest evicted
    assert store.get_exact("m", "p2\n") is not None


# ---------------------------------------------------------------------------
# Parity harness on the fakes
# ---------------------------------------------------------------------------


def test_token_divergence_rate():
    assert token_divergence_rate([1, 2, 3], [1, 2, 3]) == 0.0
    assert token_divergence_rate([], []) == 0.0
    assert token_divergence_rate([1, 2, 3, 4], [1, 2, 9, 9]) == 0.5
    assert token_divergence_rate([1, 2], [1, 2, 3, 4]) == 0.5
    assert token_divergence_rate([9], [1]) == 1.0


def test_batched_vs_sequential_identical_on_fake_engine():
    tok = FakeTokenizer()
    engine = FakeEngine(tok)
    prompts = [f"sys\nq{i}\n" for i in range(8)]
    report = compare_batched_vs_sequential(
        engine, tok, prompts, threshold=FAKE_DIVERGENCE_THRESHOLD
    )
    assert report.passed
    assert report.mean_rate == 0.0 and report.max_rate == 0.0


def test_parity_rejects_non_greedy_params():
    tok = FakeTokenizer()
    engine = FakeEngine(tok)
    with pytest.raises(ValueError):
        compare_batched_vs_sequential(
            engine, tok, ["p"], GenParams(temperature=0.7)
        )


def test_with_cache_equals_without_cache_on_fake_engine():
    tok = FakeTokenizer()
    engine = FakeEngine(tok)
    prefix = "<|system|>\nYou are terse.\n"
    suffixes = [f"<|user|>\nq{i}\n<|assistant|>\n" for i in range(6)]
    report = compare_cache_parity(engine, tok, prefix, suffixes)
    assert report.threshold == 0.0  # cache parity must be exact, always
    assert report.passed
    assert report.divergent == []
    # The prefix was computed once, not per row.
    assert engine.prefill_count == 1


def test_cache_parity_falls_back_cold_on_boundary_violation():
    # Prefix ends mid-mergeable-word; the harness must generate that row
    # cold in both arms and still be identical.
    tok = FakeTokenizer(vocab=["walking", "walk", "ing", "\n"])
    engine = FakeEngine(tok)
    report = compare_cache_parity(engine, tok, "walk", ["ing", "\n ok"])
    assert report.passed and report.mean_rate == 0.0


def test_fake_parity_suite_passes():
    reports = run_fake_parity_suite()
    assert len(reports) == 2
    assert all(r.passed for r in reports)


def test_measure_ttft_shape():
    tok = FakeTokenizer()
    engine = FakeEngine(tok)
    samples = measure_ttft(engine, tok, "sys\n", ["a", "b", "c"])
    assert [s.row for s in samples] == [0, 1, 2]
    assert all(s.cold_s >= 0.0 and s.warm_s >= 0.0 for s in samples)


# ---------------------------------------------------------------------------
# Per-row error payloads (mirror src/model_client/mod.rs create_error_response)
# ---------------------------------------------------------------------------


def test_error_payload_shape_mirrors_rust():
    obj = json.loads(error_payload("api_error", "boom"))
    assert obj == {"_error": "api_error", "_details": "boom"}
    obj = json.loads(error_payload("validation_failed", "bad", raw="{}"))
    assert obj == {"_error": "validation_failed", "_details": "bad", "_raw": "{}"}


def test_per_row_failure_does_not_kill_batch():
    tok = FakeTokenizer()
    engine = FakeEngine(tok, fail_substring="POISON")
    out = engine.generate(["ok1", "POISON row", "ok2"], GenParams())
    assert len(out) == 3
    assert json.loads(out[1])["_error"] == "engine_error"
    assert not out[0].startswith("{") and not out[2].startswith("{")
