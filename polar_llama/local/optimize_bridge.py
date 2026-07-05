"""Bridge :mod:`polar_llama.optimize` (DSPy-style prompt tuning) onto the local
MLX engine.

``optimize.py``'s ``Predict`` runs inference through its Rust async fan-out,
which only speaks to remote providers. It also exposes an ``inference_fn`` seam,
though, so a program can be driven by any backend. :func:`make_local_inference_fn`
returns such an ``inference_fn`` backed by on-device gemma-3n (mlx-lm), letting
you optimize prompts entirely locally::

    import polar_llama.optimize as po
    from polar_llama.local import make_local_inference_fn

    fn = make_local_inference_fn("mlx-community/gemma-3n-E4B-it-lm-4bit")
    module = po.Predict(sig, inference_fn=fn)
    tuned = po.InstructionOptimizer(metric=my_metric).compile(module, trainset)

Collapsed prefill (``collapse=True``, the default) computes the shared token
prefix once instead of re-prefilling it for every row. During prompt tuning the
few-shot demos make that shared system+demos prefix a large fraction of every
prompt (~80% in practice), and it is otherwise recomputed for every row of every
candidate evaluation -- so collapsing it is the dominant speedup for the tuning
schedule (~3.4x measured on a demo-laden tagging eval), with verified output
parity (benchmarks/validate_collapsed_prefill.py).

The model is loaded once via the process-global engine registry
(:func:`polar_llama.local.engine.get_engine`), so it shares weights with any
``inference_local(engine="in_process")`` calls on the same model.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, Type

__all__ = ["make_local_inference_fn"]


# ---------------------------------------------------------------------------
# JSON extraction: an inference_fn must return one JSON string (or None) per row
# matching the signature's output fields.
# ---------------------------------------------------------------------------
def _find_json_object(text: str) -> Optional[dict]:
    """Return the first balanced ``{...}`` object in ``text`` as a dict, or None.

    String-aware: braces (and quotes) inside a JSON string value are ignored, so
    a completion like ``{"answer": "a closing brace } here"}`` still parses.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : i + 1])
                except ValueError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def _extract(text: str, fields: List[str]) -> Optional[str]:
    """Coerce a model completion into a JSON string over ``fields``.

    Prefers an embedded JSON object (tolerating markdown fences / surrounding
    prose). List-valued fields are flattened to newline-joined strings so
    optimize.py's downstream parsing (which expects scalars) stays happy. Falls
    back, for a single-output signature, to wrapping the first non-empty line.
    """
    obj = _find_json_object(text)
    if obj is not None:
        obj = {
            k: ("\n".join(str(x) for x in v) if isinstance(v, list) else v)
            for k, v in obj.items()
        }
        return json.dumps(obj)

    cleaned = text.strip().strip("`").removeprefix("json").strip()
    if len(fields) == 1:
        first = cleaned.splitlines()[0].strip() if cleaned else None
        return json.dumps({fields[0]: first})
    return None


def make_local_inference_fn(
    model: str,
    *,
    engine: str = "in_process",
    collapse: bool = True,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_prefix_tokens: Optional[int] = None,
    apply_1384_patch: bool = True,
):
    """Build an ``optimize.py``-compatible ``inference_fn`` backed by local MLX.

    Parameters
    ----------
    model:
        mlx-lm model path/repo (e.g. ``mlx-community/gemma-3n-E4B-it-lm-4bit``).
    engine:
        Registry engine key (``"in_process"``); the loaded weights are shared
        with ``inference_local(engine="in_process")`` for the same model.
    collapse:
        Use collapsed prefill (share the common prompt prefix across rows). This
        is the dominant speedup for prompt tuning; set ``False`` for the plain
        ``batch_generate`` path (e.g. to A/B the two).
    max_tokens, temperature, top_p:
        Sampling parameters. ``temperature=0.0`` (default) is greedy/deterministic.
    min_prefix_tokens:
        Minimum shared-prefix length before collapsing kicks in (forwarded to
        ``collapsed_batch_generate``); ``None`` uses its default.
    apply_1384_patch:
        Apply the gemma-3n batched shared-KV fix (mlx-lm #1384) before loading.
        Required for correct *batched* gemma-3n generation; harmless otherwise.

    Returns
    -------
    Callable[[List[str], Type], List[Optional[str]]]
        The ``inference_fn`` to pass to :class:`polar_llama.optimize.Predict`.
    """
    from polar_llama.local import require_mlx

    require_mlx()

    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler

    from polar_llama.local.collapsed_prefill import collapsed_batch_generate
    from polar_llama.local.engine import get_engine

    if apply_1384_patch:
        from polar_llama.local._mlx_patches import (
            apply_gemma3n_batched_shared_kv_patch,
        )

        apply_gemma3n_batched_shared_kv_patch()

    eng = get_engine(model, engine)
    if not hasattr(eng, "get_model_and_tokenizer"):
        raise TypeError(
            "make_local_inference_fn requires the real MlxBatchEngine; got "
            f"{type(eng).__name__}. Unset POLAR_LLAMA_LOCAL_ENGINE=fake."
        )

    def inference_fn(
        messages: List[str], output_model: Type
    ) -> List[Optional[str]]:
        model_obj, tokenizer = eng.get_model_and_tokenizer()
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))

        # Each message is a JSON conversation array [{role, content}, ...];
        # apply the chat template to the FULL conversation (system + demos +
        # user) so few-shot demos are honored as real turns.
        prompt_ids = [
            tokenizer.apply_chat_template(
                json.loads(msg), add_generation_prompt=True
            )
            for msg in messages
        ]

        if collapse:
            kwargs: dict[str, Any] = {"max_tokens": max_tokens, "sampler": sampler}
            if min_prefix_tokens is not None:
                kwargs["min_prefix_tokens"] = min_prefix_tokens
            response = collapsed_batch_generate(
                model_obj, tokenizer, prompt_ids, **kwargs
            )
        else:
            response = batch_generate(
                model_obj,
                tokenizer,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

        texts = getattr(response, "texts", response)
        fields = list(output_model.model_fields)
        return [_extract(str(t), fields) for t in texts]

    return inference_fn
