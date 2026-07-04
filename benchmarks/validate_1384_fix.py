#!/usr/bin/env python
"""Parity validator for the mlx-lm issue #1384 fix (Gemma 3n batched shared-KV).

Definitive check: greedy, chat-templated generation must be identical between
the sequential path (``mlx_lm.stream_generate``, one prompt at a time) and the
batched path (``mlx_lm.generate.BatchGenerator`` — the same machinery
``mlx_lm.batch_generate`` and ``mlx_lm.server`` use).

Comparison is at token level. A prompt PASSES iff either:
  - the token streams are identical (STRICT MATCH), or
  - they are identical up to a first divergent step at which the batched
    distribution rates the two candidate tokens within ``--tie-eps`` nats of
    each other (NEAR-TIE: an argmax flip on a floating-point tie, which
    batched kernels with padded lanes can legitimately produce; the pre-fix
    bug produced immediate garbage, not ties).

Models validated:
  - mlx-community/gemma-3n-E4B-it-lm-4bit    (hybrid KV-shared: reproduces #1384)
  - mlx-community/Qwen2.5-0.5B-Instruct-4bit (full-attention control: no regression)

Usage:
    python benchmarks/validate_1384_fix.py                # installed mlx-lm as-is
    python benchmarks/validate_1384_fix.py --monkeypatch  # apply polar_llama runtime patch first
    python benchmarks/validate_1384_fix.py --strict       # require token-identical output
    python benchmarks/validate_1384_fix.py --cpu          # run on CPU (slow; leaves GPU alone)

Exit code 0 iff every prompt passes on every model.
"""

import argparse
import sys
from pathlib import Path

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "List three primary colors.",
    "What is 2+2?",
    "Name a color of the sky.",
]

MODELS = [
    ("hybrid (gemma3n, #1384 repro)", "mlx-community/gemma-3n-E4B-it-lm-4bit"),
    ("control (full attention)", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
]


def _apply_monkeypatch():
    try:
        from polar_llama.local._mlx_patches import (
            apply_gemma3n_batched_shared_kv_patch,
        )
    except Exception:
        # polar_llama may not be importable as a package in this venv
        # (compiled extension); load the patch module directly from source.
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "polar_llama/local/_mlx_patches.py"
        spec = importlib.util.spec_from_file_location("_mlx_patches", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        apply_gemma3n_batched_shared_kv_patch = (
            mod.apply_gemma3n_batched_shared_kv_patch
        )
    applied = apply_gemma3n_batched_shared_kv_patch()
    print(f"[monkeypatch] apply_gemma3n_batched_shared_kv_patch() -> {applied}")


def validate_model(label, path, max_tokens, tie_eps, strict):
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import stream_generate
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    print(f"\n=== {label}: {path} ===")
    model, tok = mlx_lm.load(path)
    msgs = [[{"role": "user", "content": q}] for q in PROMPTS]
    toks = [
        tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
        for m in msgs
    ]
    sampler = make_sampler(temp=0.0)

    # Sequential reference: tokens + per-step logprobs
    seq_steps = []
    eos = set(tok.eos_token_ids)
    for t in toks:
        steps = []
        for r in stream_generate(model, tok, t, max_tokens=max_tokens, sampler=sampler):
            steps.append((r.token, r.logprobs))
        if steps and steps[-1][0] in eos:  # mirror batch_generate: no EOS in output
            steps.pop()
        seq_steps.append(steps)

    # Batched: drive BatchGenerator (the mlx_lm.server machinery) manually so
    # we keep per-step logprobs.
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tok.eos_token_ids],
        sampler=sampler,
        max_tokens=max_tokens,
    )
    uids = gen.insert(toks, [max_tokens] * len(toks))
    bat_steps = {u: [] for u in uids}
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != "stop":  # mirror batch_generate's text
                bat_steps[r.uid].append((r.token, r.logprobs))
    gen.close()

    ok = True
    for i, u in enumerate(uids):
        s_steps, b_steps = seq_steps[i], bat_steps[u]
        s_toks = [t for t, _ in s_steps]
        b_toks = [t for t, _ in b_steps]
        text = tok.decode(s_toks).strip()
        if s_toks == b_toks:
            print(f"[MATCH] prompt {i}: {PROMPTS[i]!r}")
            print(f"    output: {text!r}")
            continue

        n = min(len(s_toks), len(b_toks))
        div = next((j for j in range(n) if s_toks[j] != b_toks[j]), n)
        verdict = "MISMATCH"
        if not strict and div < min(len(s_steps), len(b_steps)):
            st, bt = s_toks[div], b_toks[div]
            blp = b_steps[div][1]
            gap = abs((blp[bt] - blp[st]).item())
            if gap <= tie_eps:
                verdict = "NEAR-TIE"
            print(
                f"[{verdict}] prompt {i}: diverges at token {div} "
                f"(seq {tok.decode([st])!r} vs bat {tok.decode([bt])!r}, "
                f"batched logprob gap {gap:.4f} nats)"
            )
        else:
            print(f"[{verdict}] prompt {i}: diverges at token {div}")
        print(f"    sequential: {text!r}")
        print(f"    batched   : {tok.decode(b_toks).strip()!r}")
        ok &= verdict == "NEAR-TIE"

    del model
    mx.clear_cache()
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkeypatch", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--tie-eps", type=float, default=0.1)
    parser.add_argument("--models", choices=["hybrid", "control", "both"], default="both")
    args = parser.parse_args()

    import mlx.core as mx

    if args.cpu:
        mx.set_default_device(mx.cpu)
        mx.metal.is_available = lambda: False  # keep wired_limit a no-op

    if args.monkeypatch:
        _apply_monkeypatch()

    models = [
        (label, path)
        for label, path in MODELS
        if args.models == "both"
        or (args.models == "hybrid" and "gemma" in path)
        or (args.models == "control" and "Qwen" in path)
    ]

    all_ok = all(
        [
            validate_model(label, path, args.max_tokens, args.tie_eps, args.strict)
            for label, path in models
        ]
    )
    print(f"\n{'PASS' if all_ok else 'FAIL'}: sequential vs batched parity")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
