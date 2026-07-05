"""Runtime workarounds for upstream mlx-lm bugs.

Currently contains a single patch:

``apply_gemma3n_batched_shared_kv_patch``
    Works around https://github.com/ml-explore/mlx-lm/issues/1384 (present in
    mlx-lm <= 0.31.3): on Gemma 3n (hybrid KV-shared architecture), batched
    generation (``mlx_lm.batch_generate`` and ``mlx_lm.server``, which batches
    internally) either crashes ("too many values to unpack") or silently
    produces garbage output.

    Two independent bugs in ``mlx_lm/models/gemma3n.py``, both triggered only
    by the batched caches (``BatchKVCache`` / ``BatchRotatingKVCache``):

    1. Crash: the KV-shared layers read ``keys, values = cache.state``, but
       ``state`` (a serialization API) is a 4-tuple
       ``(keys, values, offset, left_padding)`` for the batched caches.

    2. Garbage: the attention captures ``offset = cache.offset``, ropes keys,
       calls ``cache.update_and_fetch`` (which advances the offset with
       ``+=`` — an *in-place mutation* of the same ``mx.array`` object when
       the cache is batched), and only then ropes the queries with that same
       ``offset``. Keys get pre-update positions, queries get post-update
       (+L) positions, on every layer. With non-batched caches ``offset`` is
       an immutable ``int``, so sequential generation never shows this.

    The patch snapshots the offset (``mx.array(cache.offset)``) and mirrors
    the KV-sharing design mlx-lm itself uses for ``gemma4_text``: each
    concrete layer passes the exact ``(keys, values)`` it attended over (the
    return value of ``cache.update_and_fetch``) and the RoPE offset it used
    for the current tokens down to the layers that share its KV. This never
    touches ``cache.state`` and is therefore correct for every cache type,
    batched or not. See ``docs/mlx_lm_1384_fix.md`` for the full analysis and
    parity validation.

    The patch is guarded (no-op if mlx-lm is missing or already fixed) and
    idempotent. It is *not* applied automatically on import of polar_llama;
    call it explicitly before loading a Gemma 3n model:

        from polar_llama.local._mlx_patches import (
            apply_gemma3n_batched_shared_kv_patch,
        )
        apply_gemma3n_batched_shared_kv_patch()
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

_PATCH_FLAG = "_polar_llama_1384_patched"


def apply_gemma3n_batched_shared_kv_patch() -> bool:
    """Apply the Gemma 3n batched shared-KV fix to an installed mlx-lm.

    Returns:
        ``True`` if the patch is active (just applied, or applied earlier in
        this process), ``False`` if it was not needed (mlx-lm absent, model
        module changed, or upstream already fixed).
    """
    try:
        from mlx_lm.models import gemma3n
    except ImportError:
        return False

    # Idempotent: never patch twice.
    if getattr(gemma3n, _PATCH_FLAG, False):
        return True

    # No-op if upstream already fixed the shared-KV path: the bug is
    # identified by the attention layer reading ``cache.state`` directly.
    try:
        src = inspect.getsource(gemma3n.Gemma3nAttention.__call__)
    except (OSError, TypeError):
        return False
    if "cache.state" not in src:
        return False

    base = gemma3n  # the module namespace carries mx / nn / helpers
    mx = base.mx
    nn = base.nn
    scaled_dot_product_attention = base.scaled_dot_product_attention
    create_attention_mask = base.create_attention_mask

    def attention_call(
        self,
        x,
        mask=None,
        cache=None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ):
        B, L, _ = x.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, -1, self.head_dim)
        queries = self.q_norm(queries)

        if self.is_kv_shared_layer and shared_kv is not None:
            # Reuse the exact keys/values the concrete layer attended over,
            # and the RoPE offset it used for the current tokens.
            keys, values = shared_kv
        else:
            # Snapshot the offset. For batched caches ``cache.offset`` is an
            # ``mx.array`` which ``update_and_fetch`` updates in place (via
            # ``+=``), and the queries below must be roped with the
            # *pre-update* offset, like the keys.
            offset = mx.array(cache.offset) if cache is not None else 0
            keys = self.k_proj(x).reshape(B, L, -1, self.head_dim)
            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_proj(x).reshape(B, L, -1, self.head_dim)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values), offset

    def decoder_layer_call(
        self,
        x,
        mask=None,
        cache=None,
        per_layer_input=None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ):
        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn, shared_kv, offset = self.self_attn(
            active_prediction_normed,
            mask,
            cache,
            shared_kv=shared_kv,
            offset=offset,
        )

        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) * (2.0**-0.5)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = first_prediction * self.altup.correct_output_scale

        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = nn.gelu_approx(first_prediction)

        first_prediction = mx.multiply(first_prediction, per_layer_input)

        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        corrected_predictions[1:] = corrected_predictions[1:] + first_prediction

        return corrected_predictions, shared_kv, offset

    def language_model_call(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
    ):
        if input_embeddings is None:
            h = self.embed_tokens(inputs) * (self.hidden_size**0.5)
        else:
            h = input_embeddings

        per_layer_inputs = self.get_per_layer_inputs(inputs)
        per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(
            h,
            cache[self.first_full_idx],
        )
        sliding_window_mask = create_attention_mask(
            h,
            cache[self.first_sliding_idx],
            window_size=self.sliding_window,
        )
        h0 = h

        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5

        h_list = [h0]
        h_list.extend([proj(h0) for proj in self.altup_projections])
        h = mx.stack(h_list, axis=0)
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h[1:] = h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))

        # Keys/values (and the RoPE offset used for the current tokens) are
        # saved per layer so that the KV-shared layers can reuse them from
        # the layer they share with, independently of the cache type.
        intermediates = [(None, None)] * len(self.layers)
        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, i, :]

            is_global = self.config.layer_types[i] == "full_attention"

            if is_global:
                mask = global_mask
            else:
                mask = sliding_window_mask

            cache_idx = self.layer_idx_to_cache_idx[i]
            c = cache[cache_idx]
            if i >= self.first_kv_shared_layer_idx and c is not None:
                shared_kv, offset = intermediates[cache_idx]
            else:
                shared_kv, offset = None, None

            h, kv, offset = layer(
                h,
                mask,
                c,
                per_layer_input,
                shared_kv=shared_kv,
                offset=offset,
            )
            intermediates[i] = (kv, offset)

        target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5
        for i, proj in enumerate(self.altup_unembed_projections):
            h[i + 1] = proj(h[i + 1])
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h[1:] = h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))

        h = mx.mean(h, axis=0)

        out = self.norm(h)
        out = self.embed_tokens.as_linear(out)
        if self.final_logit_softcapping is not None:
            out = base.logit_softcap(self.final_logit_softcapping, out)
        return out

    gemma3n.Gemma3nAttention.__call__ = attention_call
    gemma3n.Gemma3nDecoderLayer.__call__ = decoder_layer_call
    gemma3n.LanguageModel.__call__ = language_model_call
    setattr(gemma3n, _PATCH_FLAG, True)
    return True
