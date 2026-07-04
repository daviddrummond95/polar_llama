"""Batched quantized KV cache for mlx-lm (``BatchQuantizedKVCache``).

The gap (mlx-lm <= 0.31.3)
--------------------------
mlx-lm ships ``QuantizedKVCache`` for *single* sequences only:

* ``BatchKVCache`` / ``BatchRotatingKVCache`` store fp16 keys/values -- there
  is no batched quantized variant, so batched inference (``batch_generate`` /
  ``BatchGenerator`` / ``mlx_lm.server``) always pays the full fp16 KV cost.
  On a 24 GB machine this is the memory wall: Qwen3-8B-4bit with 32 rows of
  ~3K-token context needs ~17 GB of fp16 KV alone, on top of ~4.6 GB of
  weights -- past Metal's max recommended working set.
* ``QuantizedKVCache`` has no ``merge`` classmethod, so
  ``mlx_lm.generate._merge_caches`` raises "does not yet support batching".
* ``mlx_lm.models.base.quantized_scaled_dot_product_attention`` cannot accept
  the batched boolean masks that batch caches produce: with GQA
  (``n_repeats > 1``) the scores are 5-D ``(B, n_kv, R, L, S)`` while a batch
  cache's mask is 4-D ``(B, 1, L, S)``, and the broadcast raises
  ``[broadcast_shapes] (B,1,L,S) and (B,n_kv,R,L,S) cannot be broadcast``
  for any ``B > 1``. (Upstream never hits this because nothing batched is
  ever quantized.)

The fix (pure mx-ops, no Metal kernel needed)
---------------------------------------------
``BatchQuantizedKVCache`` combines the two upstream designs:

* Storage and quantize-on-write exactly like ``QuantizedKVCache``: keys and
  values are ``(packed_uint32, scales, biases)`` tuples produced by
  ``mx.quantize`` (groups run along ``head_dim``, so slicing / rolling /
  padding along the *position* axis never crosses a quantization group).
* Batch bookkeeping exactly like ``BatchKVCache``: per-row ``left_padding``,
  per-row ``offset`` (an ``mx.array`` consumed by RoPE), scalar write index
  ``_idx``, plus the full batch protocol used by ``BatchGenerator``:
  ``merge`` / ``update_and_fetch`` / ``extract`` and
  ``prepare`` / ``finalize`` / ``filter`` / ``extend`` / ``trim`` /
  ``make_mask``.
* Attention dispatch is inherited for free: ``base.scaled_dot_product_
  attention`` routes any cache exposing ``.bits`` to the existing (already
  batch-capable) ``quantized_scaled_dot_product_attention`` built on
  ``mx.quantized_matmul``. Only the 4-D-mask broadcast bug above needs a
  one-line, guarded, idempotent patch:
  :func:`apply_batched_quantized_sdpa_mask_patch` expands the batch mask to
  ``(B, 1, 1, L, S)`` when the model uses GQA.

Wiring into mlx-lm batching (no upstream edits)
-----------------------------------------------
``mlx_lm.generate`` builds batch caches by calling
``per_sequence_cache.merge([...])`` on the caches passed to
``BatchGenerator.insert(..., caches=...)``. So:

* :func:`make_quantized_prompt_cache` returns per-layer
  ``MergeableQuantizedKVCache`` seeds (a ``QuantizedKVCache`` subclass whose
  ``merge`` returns a ``BatchQuantizedKVCache``).
* :func:`batch_generate_quantized` is a drop-in ``mlx_lm.batch_generate``
  with ``kv_bits`` / ``kv_group_size``.
* :class:`QuantizedBatchGenerator` is a drop-in ``BatchGenerator`` whose
  internally-created caches are quantized.

Usage::

    from polar_llama.local.batch_quantized_kv import batch_generate_quantized
    out = batch_generate_quantized(
        model, tokenizer, prompt_token_lists, kv_bits=4, max_tokens=128
    )

Scope: models whose ``make_cache``-less default is ``KVCache`` per layer
(e.g. Qwen2/Qwen3/Llama-family full-attention models). For hybrid models the
non-``KVCache`` layers keep their stock fp16 batch caches; rotating
(sliding-window) layers are *not* quantized (upstream marks that NYI too).

This module is import-guarded (safe to import without ``mlx``) and fully
self-contained: it can be loaded via ``importlib`` from the file path without
importing ``polar_llama``. It monkeypatches nothing unless
``apply_batched_quantized_sdpa_mask_patch()`` is called (the public entry
points call it for you; it is guarded and idempotent).
"""

from __future__ import annotations

from typing import Any, List, Optional

try:  # pragma: no cover - exercised only where mlx is installed
    import mlx.core as mx
    from mlx.utils import tree_map, tree_reduce

    from mlx_lm.models import base as _mlx_base
    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.cache import (
        KVCache,
        QuantizedKVCache,
        _BaseCache,
        dynamic_roll,
    )

    MLX_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLX_AVAILABLE = False

__all__ = [
    "MLX_AVAILABLE",
    "is_available",
    "BatchQuantizedKVCache",
    "MergeableQuantizedKVCache",
    "make_quantized_prompt_cache",
    "to_mergeable_quantized",
    "apply_batched_quantized_sdpa_mask_patch",
    "batch_generate_quantized",
    "QuantizedBatchGenerator",
]

_MASK_PATCH_FLAG = "_polar_llama_batched_qkv_mask_patch"

DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_BITS = 8


def is_available() -> bool:
    """True when mlx + mlx-lm are importable (Apple silicon, [local] extra)."""
    return MLX_AVAILABLE


def _require_mlx() -> None:
    if not MLX_AVAILABLE:
        raise ImportError(
            "batch_quantized_kv requires mlx and mlx-lm (Apple silicon, "
            "`pip install polar-llama[local]`)."
        )


if MLX_AVAILABLE:

    class BatchQuantizedKVCache(_BaseCache):
        """Batched, left-padded, quantized KV cache.

        Semantics mirror ``BatchKVCache`` (left padding, per-row ``offset``
        used for RoPE, boolean batch masks) with the storage / attention
        format of ``QuantizedKVCache`` (``mx.quantize`` tuples; the
        ``bits`` / ``group_size`` attributes route attention through
        ``quantized_scaled_dot_product_attention``).
        """

        step = 256

        def __init__(
            self,
            left_padding: List[int],
            group_size: int = DEFAULT_KV_GROUP_SIZE,
            bits: int = DEFAULT_KV_BITS,
        ):
            self.keys = None  # tuple(packed, scales, biases) or None
            self.values = None
            self.left_padding = mx.array(left_padding)
            self.offset = mx.array([-l for l in left_padding])
            self._idx = 0
            self.group_size = group_size
            self.bits = bits
            self._right_padding = None

        # -- write path -----------------------------------------------------
        def update_and_fetch(self, keys, values):
            B, n_kv_heads, num_steps, k_head_dim = keys.shape
            v_head_dim = values.shape[-1]
            prev = self._idx

            if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
                el_per_int = 8 * mx.uint32.size // self.bits
                new_steps = (self.step + num_steps - 1) // self.step * self.step
                shape = (B, n_kv_heads, new_steps)

                def init_quant(dim):
                    return (
                        mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                        mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                        mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    )

                def expand_quant(x):
                    new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, new_x], axis=-2)

                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys, self.values = tree_map(
                            lambda x: x[..., :prev, :], (self.keys, self.values)
                        )
                    self.keys, self.values = tree_map(
                        expand_quant, (self.keys, self.values)
                    )
                else:
                    self.keys = init_quant(k_head_dim)
                    self.values = init_quant(v_head_dim)

            self.offset += num_steps
            self._idx += num_steps

            q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
            q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
            for i in range(len(self.keys)):
                self.keys[i][..., prev : self._idx, :] = q_keys[i]
                self.values[i][..., prev : self._idx, :] = q_values[i]

            return tree_map(
                lambda x: x[..., : self._idx, :], (self.keys, self.values)
            )

        # -- batch protocol (mirrors BatchKVCache) ---------------------------
        def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
            if left_padding is not None:
                if self.keys is not None:
                    raise ValueError(
                        "Left padding can only be added to an empty "
                        "BatchQuantizedKVCache"
                    )
                left_padding = mx.array(left_padding)
                self.left_padding += left_padding
                self.offset -= left_padding

            if right_padding is not None and max(right_padding) > 0:
                self._right_padding = mx.array(right_padding)

        def finalize(self):
            if self._right_padding is not None:
                padding = self._right_padding

                def roll(x):
                    return dynamic_roll(x, padding[:, None], axis=2)

                self.keys = tree_map(roll, self.keys)
                self.values = tree_map(roll, self.values)
                self.offset -= padding
                self.left_padding += padding
                self._right_padding = None

        def make_mask(self, N: int, return_array: bool = False, **kwargs):
            return create_causal_mask(
                N, offset=self._idx, left_padding=self.left_padding, **kwargs
            )

        def is_trimmable(self):
            return True

        def trim(self, n):
            n = min(self._idx, n)
            self._idx -= n
            self.offset -= n
            return n

        def filter(self, batch_indices):
            """In-place filter to keep just the given batch rows."""
            if self.keys is not None:
                self.keys = tree_map(lambda x: x[batch_indices], self.keys)
                self.values = tree_map(lambda x: x[batch_indices], self.values)
            self.offset = self.offset[batch_indices]
            self.left_padding = self.left_padding[batch_indices]

            # Shift left to reduce padding
            min_left_pad = self.left_padding.min().item()
            if min_left_pad > 0:
                if self.keys is not None:
                    self.keys = tree_map(
                        lambda x: x[..., min_left_pad:, :], self.keys
                    )
                    self.values = tree_map(
                        lambda x: x[..., min_left_pad:, :], self.values
                    )
                self._idx -= min_left_pad
                self.left_padding -= min_left_pad

        def extend(self, other):
            """In-place extend this cache with another BatchQuantizedKVCache."""
            if (self.group_size, self.bits) != (other.group_size, other.bits):
                raise ValueError(
                    "Cannot extend BatchQuantizedKVCache with mismatched "
                    f"quantization: ({self.group_size}, {self.bits}) vs "
                    f"({other.group_size}, {other.bits})"
                )
            if self.keys is None and other.keys is None:
                self.left_padding = mx.concatenate(
                    [self.left_padding, other.left_padding]
                )
                self.offset = mx.concatenate([self.offset, other.offset])
                return

            max_idx = max(self._idx, other._idx)
            template_k = self.keys if self.keys is not None else other.keys
            template_v = self.values if self.values is not None else other.values
            L1 = self.keys[0].shape[2] if self.keys is not None else 0
            L2 = other.keys[0].shape[2] if other.keys is not None else 0
            max_size = max(L1, L2)

            def pad(c):
                k, v = c.keys, c.values
                if k is None:
                    Bc = c.offset.shape[0]
                    k = tuple(
                        mx.zeros((Bc, t.shape[1], 0, t.shape[3]), dtype=t.dtype)
                        for t in template_k
                    )
                    v = tuple(
                        mx.zeros((Bc, t.shape[1], 0, t.shape[3]), dtype=t.dtype)
                        for t in template_v
                    )
                left = max_idx - c._idx
                right = max_size - k[0].shape[2] - left
                if right < 0:
                    k = tree_map(lambda x: x[..., :right, :], k)
                    v = tree_map(lambda x: x[..., :right, :], v)
                    right = 0
                if left != 0 or right != 0:
                    pad_widths = [(0, 0), (0, 0), (left, right), (0, 0)]
                    k = tree_map(lambda x: mx.pad(x, pad_widths), k)
                    v = tree_map(lambda x: mx.pad(x, pad_widths), v)
                return k, v, c.offset, c.left_padding + left

            (k1, v1, o1, lp1), (k2, v2, o2, lp2) = pad(self), pad(other)
            self.keys = tuple(mx.concatenate([a, b]) for a, b in zip(k1, k2))
            self.values = tuple(mx.concatenate([a, b]) for a, b in zip(v1, v2))
            self.offset = mx.concatenate([o1, o2])
            self.left_padding = mx.concatenate([lp1, lp2])
            self._idx = max_idx

        def extract(self, idx):
            """Extract row ``idx`` as a single-sequence quantized cache."""
            cache = MergeableQuantizedKVCache(
                group_size=self.group_size, bits=self.bits
            )
            padding = self.left_padding[idx].item()
            cache.keys = tree_map(
                lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]),
                self.keys,
            )
            cache.values = tree_map(
                lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]),
                self.values,
            )
            cache.offset = cache.keys[0].shape[2]
            return cache

        @classmethod
        def merge(cls, caches, group_size=None, bits=None):
            """Merge per-sequence caches into one batched quantized cache.

            Accepts ``QuantizedKVCache`` (used as-is; all must share
            ``group_size`` / ``bits``) and fp16 ``KVCache`` (quantized on
            merge). Shorter sequences are left-padded, mirroring
            ``BatchKVCache.merge``.
            """
            for c in caches:
                if isinstance(c, QuantizedKVCache):
                    if group_size is None:
                        group_size, bits = c.group_size, c.bits
                    elif (c.group_size, c.bits) != (group_size, bits):
                        raise ValueError(
                            "All quantized caches in a merge must share "
                            "group_size/bits; got "
                            f"({c.group_size}, {c.bits}) vs ({group_size}, {bits})"
                        )
                elif not isinstance(c, KVCache):
                    raise ValueError(
                        f"{type(c)} cannot be merged into a BatchQuantizedKVCache"
                    )
            if group_size is None:
                group_size, bits = DEFAULT_KV_GROUP_SIZE, DEFAULT_KV_BITS

            lengths = [c.offset for c in caches]
            max_length = max(lengths)
            if max_length == 0:
                return cls([0] * len(caches), group_size=group_size, bits=bits)

            padding = [max_length - l for l in lengths]
            B = len(caches)

            per_row = []
            for c in caches:
                if c.keys is None or c.offset == 0:
                    per_row.append(None)
                    continue
                off = c.offset
                if isinstance(c, QuantizedKVCache):
                    qk = tree_map(lambda x: x[..., :off, :], c.keys)
                    qv = tree_map(lambda x: x[..., :off, :], c.values)
                else:  # fp16 KVCache: quantize its contents now
                    qk = mx.quantize(
                        c.keys[..., :off, :], group_size=group_size, bits=bits
                    )
                    qv = mx.quantize(
                        c.values[..., :off, :], group_size=group_size, bits=bits
                    )
                per_row.append((qk, qv))

            template = next(q for q in per_row if q is not None)

            def alloc(tpl):
                return tuple(
                    mx.zeros(
                        (B, t.shape[1], max_length, t.shape[3]), dtype=t.dtype
                    )
                    for t in tpl
                )

            keys, values = alloc(template[0]), alloc(template[1])
            for i, (p, q) in enumerate(zip(padding, per_row)):
                if q is None:
                    continue
                qk, qv = q
                n = qk[0].shape[2]
                for j in range(len(keys)):
                    keys[j][i : i + 1, :, p : p + n] = qk[j]
                    values[j][i : i + 1, :, p : p + n] = qv[j]

            cache = cls(padding, group_size=group_size, bits=bits)
            cache.keys = keys
            cache.values = values
            cache.offset += max_length
            cache._idx = max_length
            return cache

        # -- serialization / introspection -----------------------------------
        @property
        def state(self):
            k, v = self.keys, self.values
            if self._idx < k[0].shape[2]:
                k = tree_map(lambda x: x[..., : self._idx, :], k)
                v = tree_map(lambda x: x[..., : self._idx, :], v)
            return (*k, *v, self.offset, self.left_padding)

        @state.setter
        def state(self, s):
            self.keys = tuple(s[0:3])
            self.values = tuple(s[3:6])
            self.offset = s[6]
            self.left_padding = s[7]
            self._idx = self.keys[0].shape[2]
            self._right_padding = None

        @property
        def meta_state(self):
            return tuple(map(str, (self.group_size, self.bits)))

        @meta_state.setter
        def meta_state(self, v):
            self.group_size, self.bits = map(int, v)

        def size(self):
            return self._idx

        def empty(self):
            return self.keys is None

        @property
        def nbytes(self):
            if self.keys is None:
                return 0
            return tree_reduce(
                lambda acc, x: acc + x.nbytes, (self.keys, self.values), 0
            )

    class MergeableQuantizedKVCache(QuantizedKVCache):
        """``QuantizedKVCache`` + the ``merge`` hook mlx-lm batching expects.

        ``mlx_lm.generate._merge_caches`` calls ``caches[0][layer].merge(...)``
        to build the batch cache; upstream ``QuantizedKVCache`` lacks
        ``merge``, which is why quantized caches "do not yet support
        batching". Instances are otherwise plain ``QuantizedKVCache`` (usable
        directly with ``generate_step``, ``save_prompt_cache``, ...).
        """

        @classmethod
        def merge(cls, caches):
            return BatchQuantizedKVCache.merge(caches)

    def make_quantized_prompt_cache(
        model,
        *,
        group_size: int = DEFAULT_KV_GROUP_SIZE,
        bits: int = DEFAULT_KV_BITS,
    ) -> List[Any]:
        """Per-sequence prompt cache with quantized KV for full-attn layers.

        Drop-in replacement for ``mlx_lm.models.cache.make_prompt_cache`` for
        use with ``BatchGenerator.insert(..., caches=...)`` /
        ``batch_generate(..., prompt_caches=...)``. Plain ``KVCache`` layers
        become ``MergeableQuantizedKVCache``; any other layer type (rotating /
        SSM / CacheList hybrids) is passed through unchanged and will batch
        with its stock fp16 path.
        """
        if hasattr(model, "make_cache"):
            base_cache = model.make_cache()
        else:
            num_layers = len(model.layers)
            base_cache = [KVCache() for _ in range(num_layers)]

        return [
            MergeableQuantizedKVCache(group_size=group_size, bits=bits)
            if type(c) is KVCache
            else c
            for c in base_cache
        ]

    def to_mergeable_quantized(
        layer_caches: List[Any],
        *,
        group_size: int = DEFAULT_KV_GROUP_SIZE,
        bits: int = DEFAULT_KV_BITS,
    ) -> List[Any]:
        """Convert one sequence's per-layer caches for quantized batching.

        fp16 ``KVCache`` layers are quantized via ``to_quantized``; existing
        ``QuantizedKVCache`` layers are re-classed (zero-copy) so they carry
        the ``merge`` hook. Other layer types pass through.
        """
        out = []
        for c in layer_caches:
            if isinstance(c, MergeableQuantizedKVCache):
                out.append(c)
            elif isinstance(c, QuantizedKVCache):
                m = MergeableQuantizedKVCache(
                    group_size=c.group_size, bits=c.bits
                )
                m.keys, m.values, m.offset = c.keys, c.values, c.offset
                out.append(m)
            elif type(c) is KVCache:
                q = c.to_quantized(group_size=group_size, bits=bits)
                m = MergeableQuantizedKVCache(group_size=group_size, bits=bits)
                m.keys, m.values, m.offset = q.keys, q.values, q.offset
                out.append(m)
            else:
                out.append(c)
        return out

    def apply_batched_quantized_sdpa_mask_patch() -> bool:
        """Fix quantized SDPA for batched (4-D) masks under GQA.

        ``quantized_scaled_dot_product_attention`` reshapes queries to
        ``(B, n_kv, R, L, D)`` when ``n_repeats > 1``, making the scores 5-D,
        but a batch cache's boolean mask is ``(B, 1, L, S)`` -- for ``B > 1``
        the broadcast raises. This wrapper inserts the missing repeat axis
        (``(B, 1, 1, L, S)``). Behavior is unchanged for ``None`` / string /
        2-D masks and for ``B == 1`` (the only case upstream ever exercised).

        Guarded and idempotent; patches the ``mlx_lm.models.base`` module
        attribute only (models resolve the symbol through that namespace at
        call time). Returns True when the patch is active.
        """
        if not MLX_AVAILABLE:
            return False
        if getattr(_mlx_base, _MASK_PATCH_FLAG, False):
            return True

        _orig = _mlx_base.quantized_scaled_dot_product_attention

        def quantized_sdpa_with_batched_mask(
            queries, q_keys, q_values, scale, mask, group_size=64, bits=8
        ):
            if (
                isinstance(mask, mx.array)
                and mask.ndim == 4
                and mask.shape[1] == 1
            ):
                n_kv_heads = q_keys[0].shape[-3]
                if queries.shape[1] // n_kv_heads > 1:
                    # scores will be (B, n_kv, R, L, S); mask (B, 1, L, S)
                    # must become (B, 1, 1, L, S) to broadcast.
                    mask = mx.expand_dims(mask, -3)
            return _orig(
                queries,
                q_keys,
                q_values,
                scale=scale,
                mask=mask,
                group_size=group_size,
                bits=bits,
            )

        quantized_sdpa_with_batched_mask._polar_llama_wrapped = _orig
        _mlx_base.quantized_scaled_dot_product_attention = (
            quantized_sdpa_with_batched_mask
        )
        setattr(_mlx_base, _MASK_PATCH_FLAG, True)
        return True

    def batch_generate_quantized(
        model,
        tokenizer,
        prompts: List[List[int]],
        *,
        kv_bits: int = DEFAULT_KV_BITS,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        prompt_caches: Optional[List[List[Any]]] = None,
        **kwargs,
    ):
        """``mlx_lm.batch_generate`` over a quantized batched KV cache.

        Identical call shape to ``mlx_lm.generate.batch_generate`` (prompts
        are token-id lists) plus ``kv_bits`` (4 or 8) and ``kv_group_size``.
        If ``prompt_caches`` (per-row prefix caches) are provided, fp16
        entries are quantized on entry.
        """
        from mlx_lm.generate import batch_generate

        apply_batched_quantized_sdpa_mask_patch()

        if prompt_caches is None:
            prompt_caches = [
                make_quantized_prompt_cache(
                    model, group_size=kv_group_size, bits=kv_bits
                )
                for _ in prompts
            ]
        else:
            prompt_caches = [
                to_mergeable_quantized(
                    pc, group_size=kv_group_size, bits=kv_bits
                )
                for pc in prompt_caches
            ]

        return batch_generate(
            model, tokenizer, prompts, prompt_caches=prompt_caches, **kwargs
        )

    def _make_quantized_batch_generator_class():
        from mlx_lm.generate import BatchGenerator

        class QuantizedBatchGenerator(BatchGenerator):
            """``BatchGenerator`` whose internally-created caches are quantized.

            Adds ``kv_bits`` / ``kv_group_size``. ``max_kv_size`` (rotating
            caches) is not supported with quantization -- upstream marks
            ``RotatingKVCache.to_quantized`` NYI as well.
            """

            def __init__(
                self,
                model,
                *,
                kv_bits: int = DEFAULT_KV_BITS,
                kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
                **kwargs,
            ):
                if kwargs.get("max_kv_size") is not None:
                    raise ValueError(
                        "max_kv_size (rotating KV) is not supported with a "
                        "quantized batched KV cache"
                    )
                self._kv_bits = kv_bits
                self._kv_group_size = kv_group_size
                apply_batched_quantized_sdpa_mask_patch()
                super().__init__(model, **kwargs)

            def _make_new_cache(self):
                return make_quantized_prompt_cache(
                    self.model,
                    group_size=self._kv_group_size,
                    bits=self._kv_bits,
                )

        return QuantizedBatchGenerator

    # Materialized lazily on first attribute access (PEP 562) to keep import
    # light and tolerate older mlx-lm without BatchGenerator.
    _quantized_batch_generator_cls = None

    def __getattr__(name):  # module-level PEP 562
        global _quantized_batch_generator_cls
        if name == "QuantizedBatchGenerator":
            if _quantized_batch_generator_cls is None:
                _quantized_batch_generator_cls = (
                    _make_quantized_batch_generator_class()
                )
            return _quantized_batch_generator_cls
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

else:  # mlx not installed: importable module, informative failures at use

    class BatchQuantizedKVCache:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            _require_mlx()

    class MergeableQuantizedKVCache:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            _require_mlx()

    class QuantizedBatchGenerator:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            _require_mlx()

    def make_quantized_prompt_cache(*a, **k):  # type: ignore[no-redef]
        _require_mlx()

    def to_mergeable_quantized(*a, **k):  # type: ignore[no-redef]
        _require_mlx()

    def apply_batched_quantized_sdpa_mask_patch() -> bool:  # type: ignore[no-redef]
        return False

    def batch_generate_quantized(*a, **k):  # type: ignore[no-redef]
        _require_mlx()
