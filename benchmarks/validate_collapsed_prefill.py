#!/usr/bin/env python
"""Correctness + speedup validator for the collapsed shared-prefix prefill.

Proves two things about ``polar_llama.local.collapsed_prefill`` on the gate
workload (N rows sharing one ~5 KB system prompt):

  (a) CORRECTNESS -- greedy token parity vs the sequential reference
      (``mlx_lm.stream_generate`` row by row). A row PASSES iff its token
      stream is identical, or first diverges on a floating-point NEAR-TIE:
      at the divergent step the batched distribution rates the two candidate
      tokens within ``--tie-eps`` nats of each other (same rule as
      ``benchmarks/validate_1384_fix.py``; batched kernels with padded lanes
      can legitimately flip an argmax tie, while a positions/cache bug
      produces immediate garbage, not ties). Validated on BOTH the hybrid
      gemma-3n E4B (with the #1384 runtime patch) and the full-attention
      Qwen2.5-0.5B control.

  (b) SPEEDUP -- wall-clock and effective tok/s for
        sequential      : one stream_generate per row (all prefill repeated)
        naive_batched   : batch_generate on the FULL prompts (batched, but
                          the shared prefix is still prefilled per row)
        collapsed       : collapsed_batch_generate (shared prefix prefilled
                          ONCE, per-row suffixes batched on top)
      plus the measured prefill-token counts that prove the collapse.

The speedup table is only printed as PASS if the parity gate passes; exit
code is 0 iff every row of every model passes parity.

Usage (Apple silicon, models must be cached):
    python benchmarks/validate_collapsed_prefill.py                 # both models
    python benchmarks/validate_collapsed_prefill.py --models hybrid
    python benchmarks/validate_collapsed_prefill.py --rows 8 --max-tokens 32  # quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MODELS = [
    ("hybrid (gemma3n, needs #1384 patch)", "mlx-community/gemma-3n-E4B-it-lm-4bit"),
    ("control (full attention)", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolve cls.__module__ via sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Workload (shared 5 KB system prompt) reused from the gate benchmark, and
# the collapsed-prefill module loaded from source so this script runs in a
# bare mlx venv without polar_llama installed.
_gate = _load_module("local_mlx_gate", REPO / "benchmarks/local_mlx_gate.py")
cp = _load_module(
    "collapsed_prefill", REPO / "polar_llama/local/collapsed_prefill.py"
)
SYSTEM_PROMPT = _gate.SYSTEM_PROMPT
build_prompts = _gate.build_prompts


def _apply_1384_patch() -> bool:
    mod = _load_module("_mlx_patches", REPO / "polar_llama/local/_mlx_patches.py")
    return mod.apply_gemma3n_batched_shared_kv_patch()


def _chats(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ],
            add_generation_prompt=True,
            tokenize=True,
        )
        for p in prompts
    ]


def run_sequential(model, tok, chats, max_tokens):
    """Reference: per-row greedy stream_generate. Returns (tokens, wall, peak)."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    eos = set(tok.eos_token_ids)
    mx.reset_peak_memory()
    seq_toks = []
    t0 = time.perf_counter()
    for chat in chats:
        toks = [
            r.token
            for r in stream_generate(
                model, tok, chat, max_tokens=max_tokens, sampler=sampler
            )
        ]
        if toks and toks[-1] in eos:  # mirror batch_generate: no EOS in output
            toks.pop()
        seq_toks.append(toks)
    wall = time.perf_counter() - t0
    return seq_toks, wall, mx.get_peak_memory() / 1e9


def run_naive_batched(model, tok, chats, max_tokens):
    """batch_generate on FULL prompts: batched decode, no prefix collapse."""
    import mlx.core as mx
    from mlx_lm import batch_generate

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    resp = batch_generate(
        model, tok, [list(c) for c in chats], max_tokens=max_tokens, verbose=False
    )
    wall = time.perf_counter() - t0
    return resp, wall, mx.get_peak_memory() / 1e9


def run_collapsed(model, tok, chats, max_tokens):
    """The deliverable path: shared prefix prefilled once, suffixes batched."""
    import mlx.core as mx

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    resp = cp.collapsed_batch_generate(model, tok, chats, max_tokens=max_tokens)
    wall = time.perf_counter() - t0
    return resp, wall, mx.get_peak_memory() / 1e9


def run_parity_drive(model, tok, chats, max_tokens, seq_toks, tie_eps):
    """Untimed collapsed run through the same primitives, with an ONLINE
    near-tie check against the sequential reference (no logprob retention:
    the gap is computed the moment a row first diverges).

    Returns (verdicts, details, batch_token_lists).
    """
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    plan = cp.plan_collapsed_prefill([list(c) for c in chats])
    assert plan is not None, "workload must have a collapsible prefix"
    prefix_cache = cp.prefill_prompt_cache(model, plan.prefix_tokens)
    caches = cp.replicate_prefix_caches(prefix_cache, plan.rows)

    eos_ids = list(tok.eos_token_ids)
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in eos_ids],
        sampler=make_sampler(temp=0.0),
        max_tokens=max_tokens,
    )
    uids = gen.insert(plan.suffix_lists(), [max_tokens] * plan.rows, caches=caches)
    row_of = {u: i for i, u in enumerate(uids)}

    n = plan.rows
    b_toks = [[] for _ in range(n)]
    diverged = [None] * n  # (step, seq_token_or_None, bat_token, gap_nats)

    def _gap(logprobs, a, b):
        return abs((logprobs[a] - logprobs[b]).item())

    while responses := gen.next_generated():
        for r in responses:
            i = row_of[r.uid]
            j = len(b_toks[i])
            include = r.finish_reason != "stop"
            if include:
                b_toks[i].append(r.token)
            if diverged[i] is not None:
                continue
            s = seq_toks[i]
            if include:
                if j < len(s):
                    if r.token != s[j]:
                        diverged[i] = (j, s[j], r.token, _gap(r.logprobs, r.token, s[j]))
                else:
                    # Sequential ended (EOS) at step j; batch kept going.
                    gap = min(_gap(r.logprobs, r.token, e) for e in eos_ids)
                    diverged[i] = (j, None, r.token, gap)
            else:
                if j < len(s):
                    # Batch chose EOS at step j; sequential continued.
                    diverged[i] = (j, s[j], r.token, _gap(r.logprobs, r.token, s[j]))
    gen.close()

    verdicts, details = [], []
    for i in range(n):
        if diverged[i] is None:
            if len(b_toks[i]) == len(seq_toks[i]):
                verdicts.append("MATCH")
                details.append(None)
            else:  # defensive; the online branches should make this unreachable
                verdicts.append("MISMATCH")
                details.append(("length", len(seq_toks[i]), len(b_toks[i]), None))
        else:
            j, s_t, b_t, gap = diverged[i]
            verdicts.append("NEAR-TIE" if gap <= tie_eps else "MISMATCH")
            details.append((j, s_t, b_t, gap))
    return verdicts, details, b_toks


def validate_model(label, path, rows, max_tokens, tie_eps, show_rows):
    import mlx.core as mx
    import mlx_lm

    print(f"\n=== {label}: {path} ===")
    print(f"rows={rows} max_tokens={max_tokens} tie_eps={tie_eps}")
    t0 = time.perf_counter()
    model, tok = mlx_lm.load(path)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    prompts = build_prompts(rows)
    chats = _chats(tok, prompts)
    plan = cp.plan_collapsed_prefill([list(c) for c in chats])
    assert plan is not None
    suffix_lens = [len(s) for s in plan.suffixes]
    print(
        f"prompt tokens/row ~{len(chats[0])}; shared token prefix (LCP) = "
        f"{plan.prefix_len}; suffix tokens/row {min(suffix_lens)}..{max(suffix_lens)}"
    )
    print(
        f"prefill tokens: naive {plan.naive_prefill_tokens} -> collapsed "
        f"{plan.collapsed_prefill_tokens} "
        f"({plan.naive_prefill_tokens / plan.collapsed_prefill_tokens:.1f}x fewer)"
    )

    # -- timed modes --------------------------------------------------------
    seq_toks, seq_wall, seq_peak = run_sequential(model, tok, chats, max_tokens)
    seq_texts = [tok.decode(t) for t in seq_toks]
    naive, naive_wall, naive_peak = run_naive_batched(model, tok, chats, max_tokens)
    coll, coll_wall, coll_peak = run_collapsed(model, tok, chats, max_tokens)
    assert coll.used_collapse, "collapse must engage on this workload"

    # -- parity gate (untimed, same machinery as the collapsed path) --------
    verdicts, details, b_toks = run_parity_drive(
        model, tok, chats, max_tokens, seq_toks, tie_eps
    )
    n_match = verdicts.count("MATCH")
    n_tie = verdicts.count("NEAR-TIE")
    n_bad = verdicts.count("MISMATCH")

    print(f"\nparity vs sequential: {n_match} MATCH, {n_tie} NEAR-TIE, {n_bad} MISMATCH")
    for i, v in enumerate(verdicts):
        if v == "MATCH":
            continue
        j, s_t, b_t, gap = details[i]
        s_txt = "<EOS>" if s_t is None else repr(tok.decode([s_t]))
        b_txt = repr(tok.decode([b_t]))
        print(
            f"  [{v}] row {i}: diverges at token {j} "
            f"(seq {s_txt} vs collapsed {b_txt}, logprob gap "
            f"{'-' if gap is None else f'{gap:.4f}'} nats)"
        )
        if v == "MISMATCH":
            print(f"      sequential: {seq_texts[i].strip()!r}")
            print(f"      collapsed : {tok.decode(b_toks[i]).strip()!r}")

    # Cross-checks (informative, non-gating)
    coll_vs_drive = sum(
        1 for a, b in zip(coll.texts, (tok.decode(t) for t in b_toks)) if a == b
    )
    naive_vs_seq = sum(1 for a, b in zip(naive.texts, seq_texts) if a == b)
    print(
        f"cross-checks: collapsed_batch_generate text == parity-drive text on "
        f"{coll_vs_drive}/{rows} rows; naive_batched text == sequential text on "
        f"{naive_vs_seq}/{rows} rows"
    )

    for i in range(min(show_rows, rows)):
        print(f"  sample row {i} [{verdicts[i]}]: {coll.texts[i].strip()[:110]!r}")

    # -- table --------------------------------------------------------------
    seq_out = sum(len(t) for t in seq_toks)
    naive_out = naive.stats.generation_tokens
    coll_out = coll.stats.generation_tokens
    coll_prefill = coll.stats.prompt_tokens + coll.plan.prefix_len

    print(f"\n{'mode':<14} | {'wall_s':>8} | {'out_tok':>7} | {'eff_tok/s':>9} | "
          f"{'prefill_tok':>11} | {'peak_gb':>7}")
    print("-" * 72)
    for name, wall, out, prefill, peak in (
        ("sequential", seq_wall, seq_out, "~" + str(plan.naive_prefill_tokens), seq_peak),
        ("naive_batched", naive_wall, naive_out, str(naive.stats.prompt_tokens), naive_peak),
        ("collapsed", coll_wall, coll_out, str(coll_prefill), coll_peak),
    ):
        print(f"{name:<14} | {wall:>8.2f} | {out:>7} | {out / wall:>9.1f} | "
              f"{prefill:>11} | {peak:>7.2f}")
    print(
        f"collapsed speedup: {seq_wall / coll_wall:.2f}x vs sequential, "
        f"{naive_wall / coll_wall:.2f}x vs naive_batched "
        f"(prefix prefilled once in {coll.prefix_prefill_s:.2f}s)"
    )
    measured_collapse = naive.stats.prompt_tokens / max(coll_prefill, 1)
    print(f"measured prefill collapse: {measured_collapse:.1f}x fewer prompt tokens")

    result = {
        "model": path,
        "rows": rows,
        "max_tokens": max_tokens,
        "prefix_len": plan.prefix_len,
        "parity": {"match": n_match, "near_tie": n_tie, "mismatch": n_bad},
        "wall_s": {
            "sequential": seq_wall,
            "naive_batched": naive_wall,
            "collapsed": coll_wall,
        },
        "eff_tok_s": {
            "sequential": seq_out / seq_wall,
            "naive_batched": naive_out / naive_wall,
            "collapsed": coll_out / coll_wall,
        },
        "peak_gb": {
            "sequential": seq_peak,
            "naive_batched": naive_peak,
            "collapsed": coll_peak,
        },
        "prompt_tokens": {
            "naive_batched": naive.stats.prompt_tokens,
            "collapsed_incl_prefix": coll_prefill,
        },
        "speedup_vs_sequential": seq_wall / coll_wall,
        "speedup_vs_naive_batched": naive_wall / coll_wall,
        "prefix_prefill_s": coll.prefix_prefill_s,
        "sample_outputs": [t.strip() for t in coll.texts[:3]],
    }

    del model
    mx.clear_cache()
    return n_bad == 0, result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", choices=["hybrid", "control", "both"], default="both")
    ap.add_argument("--rows", type=int, default=32, help="rows for the hybrid model")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--control-rows", type=int, default=8)
    ap.add_argument("--control-max-tokens", type=int, default=64)
    ap.add_argument("--tie-eps", type=float, default=0.1)
    ap.add_argument("--show-rows", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        print("mlx / mlx-lm not installed -- validator skipped (needs Apple silicon).")
        return 0

    # Required for any batched run on the hybrid model; guarded no-op elsewhere.
    print(f"[#1384 patch] applied = {_apply_1384_patch()}")

    todo = []
    for label, path in MODELS:
        if args.models == "both" or (
            args.models == "hybrid" and "gemma" in path
        ) or (args.models == "control" and "Qwen" in path):
            hybrid = "gemma" in path
            todo.append(
                (
                    label,
                    path,
                    args.rows if hybrid else args.control_rows,
                    args.max_tokens if hybrid else args.control_max_tokens,
                )
            )

    all_ok = True
    results = []
    for label, path, rows, max_tokens in todo:
        ok, res = validate_model(
            label, path, rows, max_tokens, args.tie_eps, args.show_rows
        )
        all_ok &= ok
        results.append(res)

    print(f"\n{'PASS' if all_ok else 'FAIL'}: collapsed-prefill greedy parity vs sequential")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"pass": all_ok, "results": results}, f, indent=2)
        print(f"wrote {args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
