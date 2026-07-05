#!/usr/bin/env python
"""Validate the batched quantized KV cache (polar_llama.local.batch_quantized_kv).

Proves two things on real hardware:

PARITY (``--stage parity``)
    (a) *Batching exactness*: greedy batched generation over the quantized KV
        cache matches upstream single-sequence generation over
        ``QuantizedKVCache`` (same bits; the exact math this cache batches).
        The fp16 pair (BatchKVCache vs single KVCache) is measured as the
        ceiling -- upstream fp16 batching itself flips the odd near-tie -- and
        the quantized pair must be within EXACTNESS_SLACK of that ceiling.
    (b) *Quality vs fp16*: teacher-forced next-token argmax agreement, scored
        over the fp16-batched continuations. Teacher forcing removes the
        divergence cascade (one near-tie flip changes every later token of a
        free-running comparison), so this is the honest per-step metric.
        Gates: Q8 >= 0.95, Q4 >= 0.85 absolute -- Q8 KV is near-lossless and
        Q4 KV flips only low-margin tokens (published 4-bit KV results:
        ~0.98/~0.90 typical agreement) -- and both must be within
        TF_SLACK of the fp16 teacher-forced ceiling measured in the same run.
    Free-running fp16-vs-quantized divergence is also reported (info only).

MEMORY (``--stage memory``)
    Each config (kv x batch) runs in a subprocess; peak MLX memory is compared
    against the Metal max recommended working set (the practical budget:
    beyond it macOS pages GPU memory and earlier gate runs hard-OOMed).
    PASS requires the fp16 wall config to exceed the budget (or die) while
    every quantized config at the same batch fits under it with non-empty
    output.

Run with the mlx smoke venv python, e.g.:
    $SM/bin/python benchmarks/validate_quantized_kv.py --stage parity
    $SM/bin/python benchmarks/validate_quantized_kv.py --stage memory
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BQKV_PATH = REPO / "polar_llama" / "local" / "batch_quantized_kv.py"

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"

# --- parity gates (rationale in the module docstring / docs) ---------------
# Faithfulness (batched-quantized vs single-sequence-quantized, teacher-forced
# argmax agreement) is the property this work adds: it is cascade-free and,
# per the atol-2e-3 tensor-level unit test, any token disagreement is a genuine
# near-tie (logit margin < 2e-3) flipped by padded-lane reduction order. That
# near-tie *density* scales with quantization noise, so Q4 gets a larger slack
# vs the fp16 batched-vs-single ceiling than Q8.
FAITHFULNESS_SLACK = {8: 0.03, 4: 0.06}
TF_GATES = {8: 0.95, 4: 0.85}  # absolute teacher-forced quality gates vs fp16
TF_SLACK = {8: 0.04, 4: 0.15}  # allowed drop vs measured fp16 TF-quality ceiling

PARITY_PROMPTS = [
    "Give me three facts about the Moon.",
    "Write a haiku about mountains.",
    "Explain, step by step, why the sky is blue. Keep it under 120 words.",
    "List the first 12 prime numbers.",
    ("Here is some context. " + "The quick brown fox jumps over the lazy dog. " * 40
     + "Question: how many words are in the repeated sentence, and what animal jumps?"),
    ("Summarize the following in one sentence. " + "Solar panels convert sunlight "
     "into electricity using photovoltaic cells made of semiconductor materials. "
     * 25),
    "Translate 'good morning, my friend' into French and Spanish.",
    "What is 17 * 23? Show your work.",
]


def load_bqkv():
    spec = importlib.util.spec_from_file_location("batch_quantized_kv", BQKV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encode_prompts(tokenizer, raw):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )
        for p in raw
    ]


def run_batched(model, tokenizer, prompts, max_tokens, caches=None,
                prefill_step_size=2048):
    """Drive BatchGenerator (same loop as batch_generate) collecting token ids."""
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        prefill_step_size=prefill_step_size,
    )
    try:
        uids = gen.insert(list(prompts), [max_tokens] * len(prompts), caches=caches)
        results = {u: [] for u in uids}
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
    finally:
        gen.close()
    return [results[u] for u in uids]


def run_single(model, tokenizer, prompt, max_tokens, cache_factory):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    pc = [cache_factory() for _ in range(len(model.layers))]
    eos = set(tokenizer.eos_token_ids)
    toks = []
    for tok, _ in generate_step(
        mx.array(prompt), model, max_tokens=max_tokens, prompt_cache=pc
    ):
        if tok in eos:
            break
        toks.append(tok)
    return toks


def matched_prefix_fraction(a, b):
    if a == b:
        return 1.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    denom = max(1, min(len(a), len(b)))
    return n / denom


def tf_predictions_batched(model, prompts, continuations, cache_factory,
                           chunk=256):
    """Teacher-forced argmax predictions over the continuation positions.

    The full sequences (prompt + continuation) are left-padded into a batch
    and run through fresh caches from ``cache_factory(left_pads)``; at every
    continuation position we record the model's argmax prediction given the
    exact reference history. Returns one predicted-token list per row (aligned
    to that row's continuation), so callers can compare it either with the
    reference (quality) or with another cache's predictions (faithfulness) --
    both cascade-free, unlike free-running generation.
    """
    import mlx.core as mx

    seqs = [list(p) + list(c) for p, c in zip(prompts, continuations)]
    maxlen = max(len(s) for s in seqs)
    left_pads = [maxlen - len(s) for s in seqs]
    tokens = mx.array([[0] * lp + s for lp, s in zip(left_pads, seqs)])
    caches = cache_factory(left_pads)

    preds = []
    for st in range(0, maxlen, chunk):
        logits = model(tokens[:, st : st + chunk], cache=caches)
        p = mx.argmax(logits, axis=-1)
        mx.eval(p)
        preds.append(p)
        mx.clear_cache()
    preds = mx.concatenate(preds, axis=1)  # (B, maxlen): pred for next position

    out = []
    for i, (p, c) in enumerate(zip(prompts, continuations)):
        start = left_pads[i] + len(p)  # absolute pos of first continuation token
        out.append([preds[i, start + j - 1].item() for j in range(len(c))])
    return out


def tf_predictions_single(model, prompt, continuation, cache_factory,
                          chunk=256):
    """Single-sequence teacher-forced predictions (no batch, no padding).

    The genuine upstream reference path: one sequence through plain per-layer
    caches from ``cache_factory()``.
    """
    import mlx.core as mx

    seq = list(prompt) + list(continuation)
    tokens = mx.array([seq])
    pc = [cache_factory() for _ in range(len(model.layers))]
    preds = []
    for st in range(0, len(seq), chunk):
        logits = model(tokens[:, st : st + chunk], cache=pc)
        p = mx.argmax(logits, axis=-1)
        mx.eval(p)
        preds.append(p)
        mx.clear_cache()
    preds = mx.concatenate(preds, axis=1)
    start = len(prompt)
    return [preds[0, start + j - 1].item() for j in range(len(continuation))]


def agreement(rows_a, rows_b):
    """Mean per-position token agreement between two per-row prediction lists."""
    agree = total = 0
    for a, b in zip(rows_a, rows_b):
        for x, y in zip(a, b):
            agree += int(x == y)
            total += 1
    return agree / max(1, total)


def stage_parity(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import BatchKVCache, KVCache, QuantizedKVCache

    bqkv = load_bqkv()
    bqkv.apply_batched_quantized_sdpa_mask_patch()

    model, tokenizer = load(args.model)
    prompts = encode_prompts(tokenizer, PARITY_PROMPTS[: args.rows])
    print(f"[parity] model={args.model} rows={len(prompts)} "
          f"prompt lens={[len(p) for p in prompts]} max_tokens={args.max_tokens} "
          f"group_size={args.group_size}")

    t0 = time.perf_counter()
    # Free-running fp16 batched reference: these continuations are the
    # teacher-forcing targets for every variant (and the fp16 texts).
    fp16_batched = run_batched(model, tokenizer, prompts, args.max_tokens)

    # Faithfulness ceiling: teacher-forced argmax predictions from fp16
    # *batched* vs fp16 *single-sequence* caches, compared to each other.
    # Cascade-free -- any gap is purely padded-lane reduction-order noise.
    fp16_pred_batched = tf_predictions_batched(
        model, prompts, fp16_batched,
        lambda lps: [BatchKVCache(lps) for _ in model.layers],
    )
    fp16_pred_single = [
        tf_predictions_single(model, p, c, KVCache)
        for p, c in zip(prompts, fp16_batched)
    ]
    faith_ceiling = agreement(fp16_pred_batched, fp16_pred_single)
    # Quality reference is fp16-batched predicting the fp16 reference tokens;
    # ~1.0 by construction (it predicted them), a sanity anchor.
    tf_fp16 = agreement(fp16_pred_batched, fp16_batched)
    print(f"[parity] fp16 ceiling: batched-vs-single faithfulness="
          f"{faith_ceiling:.4f} teacher-forced-quality={tf_fp16:.4f} "
          f"({time.perf_counter()-t0:.0f}s)")

    report = {
        "model": args.model,
        "rows": len(prompts),
        "max_tokens": args.max_tokens,
        "group_size": args.group_size,
        "fp16": {
            "faithfulness_ceiling": faith_ceiling,
            "teacher_forced_quality": tf_fp16,
            "texts": [tokenizer.decode(t) for t in fp16_batched],
        },
        "bits": {},
    }

    all_pass = True
    for bits in args.kv_bits:
        t0 = time.perf_counter()
        gs = args.group_size

        # Faithfulness: batched-quantized vs single-sequence-quantized argmax
        # predictions over the same reference history (the property this work
        # adds). Cascade-free; must ~match the fp16 ceiling.
        q_pred_batched = tf_predictions_batched(
            model, prompts, fp16_batched,
            lambda lps: [
                bqkv.BatchQuantizedKVCache(lps, group_size=gs, bits=bits)
                for _ in model.layers
            ],
        )
        q_pred_single = [
            tf_predictions_single(
                model, p, c,
                lambda: QuantizedKVCache(group_size=gs, bits=bits),
            )
            for p, c in zip(prompts, fp16_batched)
        ]
        faithfulness = agreement(q_pred_batched, q_pred_single)

        # Quality vs fp16: batched-quantized predictions vs the fp16 reference
        # tokens (teacher-forced, so cascade-free).
        tf = agreement(q_pred_batched, fp16_batched)

        # Free-running batched-quantized generation, reported for context
        # (cascade-dominated: one near-tie flip changes the whole tail).
        seeds = [
            [bqkv.MergeableQuantizedKVCache(group_size=gs, bits=bits)
             for _ in model.layers]
            for _ in prompts
        ]
        q_batched = run_batched(model, tokenizer, prompts, args.max_tokens,
                                caches=seeds)
        vs_fp16 = sum(
            matched_prefix_fraction(a, b) for a, b in zip(q_batched, fp16_batched)
        ) / len(prompts)

        faith_gate = faith_ceiling - FAITHFULNESS_SLACK[bits]
        ok_faith = faithfulness >= faith_gate
        ok_tf = tf >= TF_GATES[bits] and tf >= tf_fp16 - TF_SLACK[bits]
        nonempty = all(len(t) > 0 for t in q_batched)
        ok = ok_faith and ok_tf and nonempty
        all_pass &= ok

        print(f"[parity] q{bits}(g{gs}): faithfulness={faithfulness:.4f} "
              f"(gate >= {faith_gate:.4f}) "
              f"teacher-forced-quality={tf:.4f} (gate >= {TF_GATES[bits]:.2f}) "
              f"free-run-vs-fp16={vs_fp16:.4f} "
              f"[{'PASS' if ok else 'FAIL'}] ({time.perf_counter()-t0:.0f}s)")

        report["bits"][bits] = {
            "faithfulness": faithfulness,
            "faithfulness_gate": faith_gate,
            "teacher_forced_quality": tf,
            "teacher_forced_gate": TF_GATES[bits],
            "free_run_vs_fp16": vs_fp16,
            "nonempty": nonempty,
            "pass": ok,
            "texts": [tokenizer.decode(t) for t in q_batched],
        }

    report["pass"] = all_pass
    report["peak_gb"] = mx.get_peak_memory() / 1e9
    return report


# --------------------------------------------------------------------------
# kvbytes stage -- deterministic resident-KV measurement (no model, CPU-safe)
# --------------------------------------------------------------------------

# Qwen3-8B-4bit architecture (config.json): the KV footprint depends only on
# these, not on the weights, so we can measure it without loading the model.
QWEN3_8B = {"n_kv_heads": 8, "head_dim": 128, "n_layers": 36}


def _model_kv_dims(model_id):
    """Read (n_kv_heads, head_dim, n_layers) from the HF config if available."""
    try:
        import glob
        import os

        pats = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}"
            "/snapshots/*/config.json"
        )
        cfg = json.load(open(sorted(glob.glob(pats))[0]))
        n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        return n_kv, hd, cfg["num_hidden_layers"]
    except Exception:
        d = QWEN3_8B
        return d["n_kv_heads"], d["head_dim"], d["n_layers"]


def stage_kvbytes(args):
    """Measure exact resident KV bytes for fp16 vs Q8/Q4, per the real cache
    classes' ``nbytes``, without loading the model or running a huge batch.

    One transformer layer's batched cache is built at the target (batch,
    context) for each dtype and its ``nbytes`` read; the model total is
    ``per_layer * n_layers``. Runs on the CPU stream, so it is immune to the
    GPU-memory pressure that stops the end-to-end memory stage. This is the
    storage that drives the OOM, measured deterministically.
    """
    import mlx.core as mx

    # Read the Metal working-set budget BEFORE switching the default device to
    # CPU (device_info() reflects the default device and drops the Metal key
    # once it is CPU). --budget-gb overrides for other machines.
    if args.budget_gb:
        budget = args.budget_gb
    elif mx.metal.is_available():
        budget = mx.device_info().get(
            "max_recommended_working_set_size", 0) / 1e9 or 19.07
    else:
        budget = 19.07

    mx.set_default_device(mx.cpu)
    from mlx_lm.models.cache import BatchKVCache

    bqkv = load_bqkv()
    n_kv, hd, n_layers = _model_kv_dims(args.model)

    def one_layer_nbytes(cache, B, L):
        k = mx.zeros((B, n_kv, L, hd), dtype=mx.float16)
        v = mx.zeros((B, n_kv, L, hd), dtype=mx.float16)
        cache.update_and_fetch(k, v)
        mx.eval(getattr(cache, "keys"), getattr(cache, "values"))
        return cache.nbytes

    rows = []
    for B in args.kvbytes_batches:
        L = args.prompt_tokens
        fp16 = one_layer_nbytes(BatchKVCache([0] * B), B, L) * n_layers / 1e9
        q8 = one_layer_nbytes(
            bqkv.BatchQuantizedKVCache([0] * B, group_size=args.group_size, bits=8),
            B, L) * n_layers / 1e9
        q4 = one_layer_nbytes(
            bqkv.BatchQuantizedKVCache([0] * B, group_size=args.group_size, bits=4),
            B, L) * n_layers / 1e9
        rows.append({"batch": B, "context": L, "fp16_gb": fp16,
                     "q8_gb": q8, "q4_gb": q4,
                     "q8_ratio": q8 / fp16, "q4_ratio": q4 / fp16})
        mx.clear_cache()

    print(f"[kvbytes] model={args.model} n_kv_heads={n_kv} head_dim={hd} "
          f"layers={n_layers} group_size={args.group_size} "
          f"budget≈{budget:.1f} GB")
    print(f"\n| batch | ctx | fp16 KV GB | Q8 KV GB | Q4 KV GB | Q8/fp16 | Q4/fp16 |")
    print("|-------|-----|-----------|----------|----------|---------|---------|")
    for r in rows:
        print(f"| {r['batch']:>5} | {r['context']:>4} | {r['fp16_gb']:>9.2f} "
              f"| {r['q8_gb']:>8.2f} | {r['q4_gb']:>8.2f} | {r['q8_ratio']:>7.2f} "
              f"| {r['q4_ratio']:>7.2f} |")

    # Weights (~4.6 GB for 8B-4bit) + KV must fit the working-set budget.
    weights = args.weights_gb
    print(f"\n[kvbytes] with ~{weights:.1f} GB weights, total (weights+KV) vs "
          f"{budget:.1f} GB budget:")
    verdict_rows = []
    for r in rows:
        fp16_tot = weights + r["fp16_gb"]
        q4_tot = weights + r["q4_gb"]
        q8_tot = weights + r["q8_gb"]
        print(f"  batch {r['batch']:>3}: fp16 {fp16_tot:>5.1f} GB "
              f"({'FITS' if fp16_tot <= budget else 'OOM'}) | "
              f"q8 {q8_tot:>5.1f} GB ({'FITS' if q8_tot <= budget else 'OOM'}) | "
              f"q4 {q4_tot:>5.1f} GB ({'FITS' if q4_tot <= budget else 'OOM'})")
        verdict_rows.append((r["batch"], fp16_tot, q8_tot, q4_tot))

    # PASS: some batch where fp16 total exceeds budget but Q4 (and/or Q8) fits.
    ok = any(f > budget and q4 <= budget for _, f, _, q4 in verdict_rows)
    print(f"\n[kvbytes] {'PASS' if ok else 'FAIL'} "
          f"(a batch where fp16 KV overflows the budget but Q4 KV fits)")
    return {"rows": rows, "budget_gb": budget, "weights_gb": weights, "pass": ok}


# --------------------------------------------------------------------------
# memory stage
# --------------------------------------------------------------------------

def synth_prompts(tokenizer, n_rows, n_tokens):
    base = tokenizer.encode(
        "The history of computing spans mechanical calculators, vacuum tubes, "
        "transistors, integrated circuits and modern accelerators; each era "
        "reshaped what software could do and who could afford to run it. "
    )
    prompts = []
    for i in range(n_rows):
        want = max(16, n_tokens - (i * 17) % 64)  # slight length variety
        reps = want // len(base) + 1
        prompts.append((base * reps)[:want])
    return prompts


def stage_mem_worker(args):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import batch_generate

    bqkv = load_bqkv()
    budget = mx.device_info()["max_recommended_working_set_size"]

    model, tokenizer = load(args.model)
    prompts = synth_prompts(tokenizer, args.worker_batch, args.prompt_tokens)
    kv = args.worker_kv
    t0 = time.perf_counter()
    try:
        if kv == "fp16":
            r = batch_generate(model, tokenizer, prompts,
                               max_tokens=args.max_tokens,
                               prefill_step_size=args.prefill_step_size)
        else:
            bits = int(kv[1:])
            r = bqkv.batch_generate_quantized(
                model, tokenizer, prompts, kv_bits=bits,
                kv_group_size=args.group_size, max_tokens=args.max_tokens,
                prefill_step_size=args.prefill_step_size)
        peak = mx.get_peak_memory()
        out = {
            "ok": True,
            "kv": kv,
            "batch": args.worker_batch,
            "prompt_tokens": args.prompt_tokens,
            "peak_gb": peak / 1e9,
            "budget_gb": budget / 1e9,
            "fits_budget": bool(peak <= budget),
            "wall_s": time.perf_counter() - t0,
            "gen_tps": r.stats.generation_tps,
            "nonempty_rows": sum(bool(t.strip()) for t in r.texts),
            "sample": r.texts[0][:80],
        }
    except Exception as e:  # metal OOM etc.
        out = {
            "ok": False,
            "kv": kv,
            "batch": args.worker_batch,
            "prompt_tokens": args.prompt_tokens,
            "peak_gb": mx.get_peak_memory() / 1e9,
            "budget_gb": budget / 1e9,
            "fits_budget": False,
            "error": f"{type(e).__name__}: {e}",
        }
    print("MEMJSON " + json.dumps(out))
    return out


def stage_memory(args):
    results = []
    for cfg in args.mem_configs.split(","):
        kv, batch = cfg.split(":")
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--stage", "mem-worker",
            "--model", args.model,
            "--worker-kv", kv,
            "--worker-batch", batch,
            "--prompt-tokens", str(args.prompt_tokens),
            "--max-tokens", str(args.mem_max_tokens),
            "--group-size", str(args.group_size),
            # Same prefill step for every config so peak differences are pure
            # KV storage, not different prompt-logit / attention-score
            # transients. (The unfused quantized path materializes an extra
            # ~1.5 GB attention-scores transient per layer at this step/batch;
            # keeping the step equal makes fp16 vs quantized apples-to-apples.)
            "--prefill-step-size", str(args.prefill_step_size),
        ]
        print(f"[memory] running {cfg} ...", flush=True)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.worker_timeout
            )
            line = next(
                (l for l in proc.stdout.splitlines() if l.startswith("MEMJSON ")),
                None,
            )
            if line:
                res = json.loads(line[len("MEMJSON "):])
            else:
                res = {
                    "ok": False, "kv": kv, "batch": int(batch),
                    "fits_budget": False,
                    "error": f"worker died rc={proc.returncode}: "
                             f"{proc.stderr.strip()[-300:]}",
                }
        except subprocess.TimeoutExpired:
            res = {"ok": False, "kv": kv, "batch": int(batch),
                   "fits_budget": False,
                   "error": f"timeout>{args.worker_timeout}s (thrash/OOM)"}
        results.append(res)
        status = ("OOM/died: " + res.get("error", "")[:120]) if not res["ok"] else (
            f"peak={res['peak_gb']:.2f} GB "
            f"({'fits' if res['fits_budget'] else 'EXCEEDS'} budget "
            f"{res.get('budget_gb', 0):.2f} GB)")
        print(f"[memory] {cfg}: {status}", flush=True)

    # ---- table + verdict ----
    budget = next((r.get("budget_gb") for r in results if r.get("budget_gb")), 0)
    print(f"\n| kv    | batch | prompt_toks | peak GB | fits {budget:.1f} GB budget |")
    print("|-------|-------|-------------|---------|----------------------|")
    for r in results:
        peak = f"{r['peak_gb']:.2f}" if r.get("peak_gb") else "died"
        fits = "yes" if r.get("fits_budget") else ("NO" if r["ok"] else "NO (died/timeout)")
        print(f"| {r['kv']:<5} | {r['batch']:>5} | {r.get('prompt_tokens','-'):>11} "
              f"| {peak:>7} | {fits:<20} |")

    fp16_walls = [r for r in results if r["kv"] == "fp16" and not r["fits_budget"]]
    quant_fits = [
        r for r in results
        if r["kv"] != "fp16" and r["ok"] and r["fits_budget"]
        and r.get("nonempty_rows", 0) == r["batch"]
    ]
    wall_batch = max((r["batch"] for r in fp16_walls), default=None)
    ok = bool(fp16_walls) and any(
        r["batch"] >= (wall_batch or 0) for r in quant_fits
    )
    print(f"\n[memory] fp16 exceeds budget at batch: "
          f"{[r['batch'] for r in fp16_walls] or 'never (increase batch/context)'}")
    print(f"[memory] quantized fits at batch: "
          f"{[(r['kv'], r['batch']) for r in quant_fits] or 'none'}")
    print(f"[memory] {'PASS' if ok else 'FAIL'}")
    return {"results": results, "pass": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["parity", "memory", "kvbytes", "all", "mem-worker"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--kv-bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=2816,
                    help="memory stage: per-row synthetic prompt length")
    ap.add_argument("--mem-max-tokens", type=int, default=8,
                    help="memory stage: decode steps (peak is hit early)")
    ap.add_argument("--mem-configs", default="fp16:16,fp16:32,q8:32,q4:32",
                    help="comma list of kv:batch subprocess configs")
    ap.add_argument("--prefill-step-size", type=int, default=256)
    ap.add_argument("--worker-timeout", type=int, default=1800)
    ap.add_argument("--kvbytes-batches", type=int, nargs="+",
                    default=[8, 16, 24, 32, 48, 64],
                    help="kvbytes stage: batches to tabulate resident KV for")
    ap.add_argument("--weights-gb", type=float, default=4.6,
                    help="kvbytes stage: model weight footprint (8B-4bit≈4.6)")
    ap.add_argument("--budget-gb", type=float, default=0.0,
                    help="kvbytes stage: override the Metal working-set budget")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full report to this path")
    # internal (mem-worker)
    ap.add_argument("--worker-kv", default=None)
    ap.add_argument("--worker-batch", type=int, default=None)
    args = ap.parse_args()

    if args.stage == "mem-worker":
        out = stage_mem_worker(args)
        sys.exit(0 if out.get("ok") is not None else 1)

    report = {}
    ok = True
    if args.stage in ("parity", "all"):
        report["parity"] = stage_parity(args)
        ok &= report["parity"]["pass"]
        # memory stage in the same process would inherit parity's peak; keep
        # the stages separate (memory uses subprocesses anyway).
    if args.stage in ("kvbytes", "all"):
        report["kvbytes"] = stage_kvbytes(args)
        ok &= report["kvbytes"]["pass"]
    if args.stage in ("memory", "all"):
        report["memory"] = stage_memory(args)
        ok &= report["memory"]["pass"]

    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
        print(f"[report] wrote {args.json}")

    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
