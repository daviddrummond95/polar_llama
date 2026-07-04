"""CI-safe unit tests for polar_llama.local.batch_quantized_kv.

Everything runs with fake tensors on the CPU stream (no GPU / no model
download), and the whole module is skipped when mlx / mlx-lm are not
installed (e.g. Linux CI). Run locally with:

    pytest tests/test_batch_quantized_kv.py -m local
"""

import pytest

pytestmark = pytest.mark.local

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx_lm.models.base as mlx_base  # noqa: E402
from mlx.utils import tree_map  # noqa: E402
from mlx_lm.models.cache import BatchKVCache, KVCache, QuantizedKVCache  # noqa: E402

try:
    from polar_llama.local.batch_quantized_kv import (  # noqa: E402
        BatchQuantizedKVCache,
        MergeableQuantizedKVCache,
        apply_batched_quantized_sdpa_mask_patch,
        make_quantized_prompt_cache,
        to_mergeable_quantized,
    )
except ImportError:  # package deps (polars/plugin) absent: load from path
    import importlib.util
    from pathlib import Path

    _spec = importlib.util.spec_from_file_location(
        "batch_quantized_kv",
        Path(__file__).resolve().parents[1]
        / "polar_llama"
        / "local"
        / "batch_quantized_kv.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BatchQuantizedKVCache = _mod.BatchQuantizedKVCache
    MergeableQuantizedKVCache = _mod.MergeableQuantizedKVCache
    apply_batched_quantized_sdpa_mask_patch = (
        _mod.apply_batched_quantized_sdpa_mask_patch
    )
    make_quantized_prompt_cache = _mod.make_quantized_prompt_cache
    to_mergeable_quantized = _mod.to_mergeable_quantized

mx.set_default_device(mx.cpu)

H, DK, DV = 2, 64, 64
GS, BITS = 32, 4


def _rand(n_pos, seed):
    mx.random.seed(seed)
    k = mx.random.normal((1, H, n_pos, DK)).astype(mx.float16)
    v = mx.random.normal((1, H, n_pos, DV)).astype(mx.float16)
    return k, v


def _single_cache(n_pos, seed, chunk=None):
    """Single-sequence quantized cache filled via update_and_fetch."""
    c = MergeableQuantizedKVCache(group_size=GS, bits=BITS)
    k, v = _rand(n_pos, seed)
    if chunk is None:
        c.update_and_fetch(k, v)
    else:
        for s in range(0, n_pos, chunk):
            c.update_and_fetch(k[..., s : s + chunk, :], v[..., s : s + chunk, :])
    return c, k, v


def _dequant(qtuple):
    return mx.dequantize(*qtuple, group_size=GS, bits=BITS)


def _row_state(batch_cache, idx):
    """Dequantized (keys, values) of one row of a batch cache, padding stripped."""
    c = batch_cache.extract(idx)
    return _dequant(c.keys), _dequant(c.values)


def test_update_and_fetch_quantizes_on_write():
    c = BatchQuantizedKVCache([0, 0], group_size=GS, bits=BITS)
    mx.random.seed(0)
    k = mx.random.normal((2, H, 7, DK)).astype(mx.float16)
    v = mx.random.normal((2, H, 7, DV)).astype(mx.float16)
    (qk, qv) = c.update_and_fetch(k, v)
    assert c._idx == 7
    assert c.offset.tolist() == [7, 7]
    # Returned quantized keys equal direct mx.quantize of the fp16 input
    ref = mx.quantize(k, group_size=GS, bits=BITS)
    for a, b in zip(qk, ref):
        assert mx.array_equal(a, b)
    ref_v = mx.quantize(v, group_size=GS, bits=BITS)
    for a, b in zip(qv, ref_v):
        assert mx.array_equal(a, b)


def test_buffer_growth_past_step():
    c = BatchQuantizedKVCache([0], group_size=GS, bits=BITS)
    total = BatchQuantizedKVCache.step + 40  # force one expansion
    mx.random.seed(1)
    k = mx.random.normal((1, H, total, DK)).astype(mx.float16)
    v = mx.random.normal((1, H, total, DV)).astype(mx.float16)
    for s in range(0, total, 50):  # unaligned chunks
        c.update_and_fetch(k[..., s : s + 50, :], v[..., s : s + 50, :])
    assert c._idx == total
    got_k = _dequant(tree_map(lambda x: x[..., :total, :], c.keys))
    want_k = _dequant(mx.quantize(k, group_size=GS, bits=BITS))
    assert mx.allclose(got_k, want_k)


def test_merge_left_pads_and_roundtrips():
    c1, k1, v1 = _single_cache(11, seed=2)
    c2, k2, v2 = _single_cache(5, seed=3)
    batch = MergeableQuantizedKVCache.merge([c1, c2])
    assert isinstance(batch, BatchQuantizedKVCache)
    assert batch._idx == 11
    assert batch.left_padding.tolist() == [0, 6]
    assert batch.offset.tolist() == [11, 5]
    # Row contents identical to the per-sequence caches (exact: same ints)
    for idx, c in ((0, c1), (1, c2)):
        rk, rv = _row_state(batch, idx)
        assert mx.array_equal(rk, _dequant(c.state[0]))
        assert mx.array_equal(rv, _dequant(c.state[1]))


def test_merge_quantizes_fp16_kvcache_inputs():
    fp = KVCache()
    k, v = _rand(9, seed=4)
    fp.update_and_fetch(k, v)
    q, _, _ = _single_cache(9, seed=4)
    batch = BatchQuantizedKVCache.merge([fp, q], group_size=GS, bits=BITS)
    rk0, rv0 = _row_state(batch, 0)
    rk1, rv1 = _row_state(batch, 1)
    assert mx.array_equal(rk0, rk1)
    assert mx.array_equal(rv0, rv1)


def test_right_padding_finalize_matches_unpadded():
    # Sequence 2 is 4 shorter; feed it right-padded junk, then finalize.
    n1, n2 = 12, 8
    _, k1, v1 = _single_cache(n1, seed=5)
    _, k2, v2 = _single_cache(n2, seed=6)
    junk_k = mx.ones((1, H, n1 - n2, DK), dtype=mx.float16)
    junk_v = mx.ones((1, H, n1 - n2, DV), dtype=mx.float16)
    k2p = mx.concatenate([k2, junk_k], axis=2)
    v2p = mx.concatenate([v2, junk_v], axis=2)

    batch = BatchQuantizedKVCache([0, 0], group_size=GS, bits=BITS)
    batch.prepare(lengths=[n1, n2], right_padding=[0, n1 - n2])
    batch.update_and_fetch(
        mx.concatenate([k1, k2p], axis=0), mx.concatenate([v1, v2p], axis=0)
    )
    batch.finalize()

    assert batch.left_padding.tolist() == [0, n1 - n2]
    assert batch.offset.tolist() == [n1, n2]
    rk, rv = _row_state(batch, 1)
    want_k = _dequant(mx.quantize(k2, group_size=GS, bits=BITS))
    assert mx.array_equal(rk, want_k)


def test_filter_shifts_left_padding_and_preserves_rows():
    c1, _, _ = _single_cache(10, seed=7)
    c2, _, _ = _single_cache(6, seed=8)
    c3, _, _ = _single_cache(4, seed=9)
    batch = BatchQuantizedKVCache.merge([c1, c2, c3])
    before = [_row_state(batch, i) for i in range(3)]
    batch.filter([1, 2])  # drop the longest row -> min left pad shift kicks in
    assert batch.left_padding.min().item() == 0
    assert batch.offset.tolist() == [6, 4]
    for new_idx, old_idx in ((0, 1), (1, 2)):
        rk, rv = _row_state(batch, new_idx)
        assert mx.array_equal(rk, before[old_idx][0])
        assert mx.array_equal(rv, before[old_idx][1])


def test_extend_including_empty_side():
    c1, _, _ = _single_cache(9, seed=10)
    a = BatchQuantizedKVCache.merge([c1])
    b = BatchQuantizedKVCache([0, 0], group_size=GS, bits=BITS)  # empty, 2 rows
    a.extend(b)
    # Empty rows are right-justified: 9 slots of pure left padding, 0 tokens.
    assert a.offset.tolist() == [9, 0, 0]
    assert a.left_padding.tolist() == [0, 9, 9]
    c2, _, _ = _single_cache(3, seed=11)
    other = BatchQuantizedKVCache.merge([c2])
    base = BatchQuantizedKVCache.merge([c1])
    base.extend(other)
    assert base._idx == 9
    assert base.left_padding.tolist() == [0, 6]
    rk, _ = _row_state(base, 1)
    assert mx.array_equal(rk, _dequant(c2.state[0]))


def test_extend_rejects_mismatched_quantization():
    a = BatchQuantizedKVCache([0], group_size=GS, bits=BITS)
    b = BatchQuantizedKVCache([0], group_size=GS, bits=8)
    with pytest.raises(ValueError):
        a.extend(b)


def test_make_mask_matches_batch_kv_cache():
    lp = [3, 0]
    qc = BatchQuantizedKVCache(lp, group_size=GS, bits=BITS)
    fc = BatchKVCache(lp)
    k = mx.zeros((2, H, 5, DK), dtype=mx.float16)
    v = mx.zeros((2, H, 5, DV), dtype=mx.float16)
    qc.update_and_fetch(k, v)
    fc.update_and_fetch(k, v)
    m_q = qc.make_mask(1)
    m_f = fc.make_mask(1)
    assert m_q.shape == m_f.shape
    assert mx.array_equal(m_q, m_f)


def test_state_meta_state_roundtrip():
    c1, _, _ = _single_cache(7, seed=12)
    c2, _, _ = _single_cache(7, seed=13)
    batch = BatchQuantizedKVCache.merge([c1, c2])
    restored = BatchQuantizedKVCache.from_state(batch.state, batch.meta_state)
    assert restored.group_size == GS and restored.bits == BITS
    assert restored._idx == batch._idx
    for i in range(2):
        rk, rv = _row_state(batch, i)
        sk, sv = _row_state(restored, i)
        assert mx.array_equal(rk, sk)
        assert mx.array_equal(rv, sv)


def test_trim():
    c1, _, _ = _single_cache(10, seed=14)
    batch = BatchQuantizedKVCache.merge([c1])
    assert batch.is_trimmable()
    assert batch.trim(4) == 4
    assert batch._idx == 6
    assert batch.offset.tolist() == [6]


def test_to_mergeable_quantized_conversion():
    fp = KVCache()
    k, v = _rand(8, seed=15)
    fp.update_and_fetch(k, v)
    plain_q = QuantizedKVCache(group_size=GS, bits=BITS)
    k2, v2 = _rand(8, seed=16)
    plain_q.update_and_fetch(k2, v2)
    out = to_mergeable_quantized([fp, plain_q, "passthrough"], group_size=GS, bits=BITS)
    assert isinstance(out[0], MergeableQuantizedKVCache)
    assert out[0].offset == 8
    assert isinstance(out[1], MergeableQuantizedKVCache)
    assert out[1].offset == 8
    assert out[2] == "passthrough"


def test_mask_patch_idempotent_and_fixes_gqa_broadcast():
    assert apply_batched_quantized_sdpa_mask_patch() is True
    patched = mlx_base.quantized_scaled_dot_product_attention
    assert apply_batched_quantized_sdpa_mask_patch() is True  # idempotent
    assert mlx_base.quantized_scaled_dot_product_attention is patched

    # GQA (n_repeats > 1) + batched 4-D bool mask: raises unpatched, works now.
    # B must differ from n_kv: when B == n_kv the unpatched broadcast succeeds
    # SILENTLY with the mask batch dim applied to the kv-head dim (worse).
    B, n_q, n_kv, L, S, D = 3, 4, 2, 1, 6, DK
    mx.random.seed(17)
    q = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
    keys = mx.random.normal((B, n_kv, S, D)).astype(mx.float16)
    values = mx.random.normal((B, n_kv, S, D)).astype(mx.float16)
    qk = mx.quantize(keys, group_size=GS, bits=BITS)
    qv = mx.quantize(values, group_size=GS, bits=BITS)
    mask = mx.random.uniform(shape=(B, 1, L, S)) > 0.3
    orig = patched._polar_llama_wrapped
    with pytest.raises(ValueError):
        orig(q[:], qk, qv, scale=1.0, mask=mask, group_size=GS, bits=BITS)
    out = patched(q[:], qk, qv, scale=1.0, mask=mask, group_size=GS, bits=BITS)
    assert out.shape == (B, n_q, L, D)


def test_batched_quantized_attention_matches_per_row():
    """End equivalence: batched quantized SDPA over a merged, left-padded
    cache equals per-row single-sequence quantized SDPA."""
    apply_batched_quantized_sdpa_mask_patch()
    n_q = 4  # GQA: n_repeats = 2 over H = 2 kv heads
    lens = [10, 6]
    singles = [_single_cache(n, seed=20 + i) for i, n in enumerate(lens)]
    batch = MergeableQuantizedKVCache.merge([s[0] for s in singles])

    mx.random.seed(30)
    q_new = mx.random.normal((2, n_q, 1, DK)).astype(mx.float16)
    k_new = mx.random.normal((2, H, 1, DK)).astype(mx.float16)
    v_new = mx.random.normal((2, H, 1, DV)).astype(mx.float16)

    mask = batch.make_mask(1)
    keys, values = batch.update_and_fetch(k_new, v_new)
    # NOTE: upstream quantized SDPA does ``queries *= scale`` which rebinds
    # the *passed-in* python object; always hand it a fresh slice object so
    # subsequent uses of q_new in this test see unscaled values.
    out_b = mlx_base.scaled_dot_product_attention(
        q_new[:], keys, values, cache=batch, scale=DK**-0.5, mask=mask
    )

    for i, (c, _, _) in enumerate(singles):
        keys_s, values_s = c.update_and_fetch(
            k_new[i : i + 1], v_new[i : i + 1]
        )
        out_s = mlx_base.scaled_dot_product_attention(
            q_new[i : i + 1],
            keys_s,
            values_s,
            cache=c,
            scale=DK**-0.5,
            mask=None,  # single-sequence decode step: no mask needed
        )
        assert mx.allclose(
            out_b[i : i + 1].astype(mx.float32),
            out_s.astype(mx.float32),
            atol=2e-3,
            rtol=2e-2,
        ), f"row {i} batched vs single mismatch"


def test_make_quantized_prompt_cache_layer_types():
    class _FakeModel:
        layers = [object(), object(), object()]

    caches = make_quantized_prompt_cache(_FakeModel(), group_size=GS, bits=BITS)
    assert len(caches) == 3
    assert all(isinstance(c, MergeableQuantizedKVCache) for c in caches)
    assert caches[0].group_size == GS and caches[0].bits == BITS
