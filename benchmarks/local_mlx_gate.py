#!/usr/bin/env python3
"""Build-vs-buy gate benchmark for the local MLX backend.

Compares three ways of running N prompts that share one long system prompt
against a local MLX model on Apple silicon:

  a) ``sequential``  -- one ``mlx_lm.generate`` call per row (naive baseline).
  b) ``server``      -- the "buy" path: polar_llama's existing Rust async
                        fan-out pointed at a local OpenAI-compatible server
                        (``mlx_lm.server`` or vllm-mlx) via ``OPENAI_BASE_URL``.
  c) ``in_process``  -- the "build" path: ``mlx_lm.batch_generate`` with the
                        shared system-prompt prefix cache replicated per row.

Metrics per mode (see docs/local_mlx_gate_decision.md for the decision rule):

  * wall_s            -- total wall-clock for the whole column, including
                         model load amortisation excluded (load is timed
                         separately) but including ALL prefill.
  * ttft_s            -- per-row time-to-first-token (sequential / in_process:
                         measured via streaming callbacks where available;
                         server: not per-row observable through the fan-out,
                         reported as null).
  * effective_tok_s   -- total OUTPUT tokens / total wall-clock. Never
                         decode-only throughput.
  * peak_mem_gb       -- ``mx.get_peak_memory()`` for in-process modes; RSS
                         delta for the server mode (server process not
                         instrumented -- noted in JSON).

The script degrades gracefully: with no mlx installed it prints what it
*would* run and exits 0 (so it is CI-importable and CI-runnable as a no-op).

Usage
-----
Tiny default model (~300 MB download, good for validating the harness).
NOTE: mlx-lm 0.31.x is incompatible with transformers>=5 (import fails in
``AutoTokenizer.register``), so pin transformers<5:

    uv run --with mlx-lm --with 'transformers<5' \
        python benchmarks/local_mlx_gate.py --rows 32

Real gate measurement on the target models (Apple M4 Pro, 24 GB):

    uv run --with mlx-lm --with 'transformers<5' \
        python benchmarks/local_mlx_gate.py \
        --model mlx-community/gemma-4-E4B-it-4bit --rows 64 --max-tokens 256
    uv run --with mlx-lm --with 'transformers<5' \
        python benchmarks/local_mlx_gate.py \
        --model mlx-community/Ornith-1.0-9B-4bit --rows 64 --max-tokens 256

Server mode needs a local OpenAI-compatible server started separately, e.g.:

    uv run --with mlx-lm python -m mlx_lm server \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8080

then pass ``--base-url http://127.0.0.1:8080``. Without ``--base-url`` the
server mode is skipped (not failed).

Output: a comparison table on stdout plus ``--json PATH`` machine-readable
results.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------

# Long shared system prompt: the whole point of the gate is prefix-cache
# behaviour, so this must dominate per-row input tokens.
SYSTEM_PROMPT = (
    "You are a meticulous data-labelling assistant embedded in a Polars "
    "pipeline. Follow these rules exactly. "
    + " ".join(
        f"Rule {i}: when classifying record {i}, consider tone, factuality, "
        f"and audience, and never emit more than one label per record."
        for i in range(1, 41)
    )
    + " Respond concisely and never repeat the question."
)

USER_TEMPLATE = (
    "Record {i}: 'Customer says the {noun} arrived {state} and support was "
    "{quality}.' Classify the sentiment as positive, negative, or mixed, and "
    "give one short justification."
)

_NOUNS = ["package", "laptop", "invoice", "refund", "headset", "manual"]
_STATES = ["late", "damaged", "early", "as described", "incomplete"]
_QUALITY = ["helpful", "slow", "rude", "excellent", "unreachable"]


def build_prompts(rows: int) -> list[str]:
    return [
        USER_TEMPLATE.format(
            i=i,
            noun=_NOUNS[i % len(_NOUNS)],
            state=_STATES[i % len(_STATES)],
            quality=_QUALITY[i % len(_QUALITY)],
        )
        for i in range(rows)
    ]


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    mode: str
    ok: bool
    wall_s: Optional[float] = None
    load_s: Optional[float] = None
    output_tokens: Optional[int] = None
    effective_tok_s: Optional[float] = None
    ttft_s: Optional[list] = None  # per-row seconds, or None if unobservable
    peak_mem_gb: Optional[float] = None
    rows: int = 0
    note: str = ""
    completions: list = field(default_factory=list, repr=False)

    def finalize(self) -> None:
        if self.ok and self.wall_s and self.output_tokens is not None:
            self.effective_tok_s = self.output_tokens / self.wall_s

    def to_json(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "completions"}
        d["ttft_p50_s"] = _p50(self.ttft_s)
        return d


def _p50(xs: Optional[list]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(x for x in xs if x is not None)
    return s[len(s) // 2] if s else None


def _rss_gb() -> float:
    # ru_maxrss is bytes on macOS, KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3) if sys.platform == "darwin" else rss / (1024**2)


def _count_tokens(tokenizer, texts: list[str]) -> int:
    return sum(len(tokenizer.encode(t)) for t in texts if t)


# ---------------------------------------------------------------------------
# Mode (a): sequential mlx_lm.generate
# ---------------------------------------------------------------------------


def run_sequential(model, tokenizer, prompts: list[str], args) -> ModeResult:
    import mlx.core as mx
    from mlx_lm import stream_generate

    res = ModeResult(mode="sequential", ok=True, rows=len(prompts), ttft_s=[])
    mx.reset_peak_memory()
    completions: list[str] = []
    total_out = 0
    t_all = time.perf_counter()
    for prompt in prompts:
        chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=True,
        )
        t0 = time.perf_counter()
        ttft = None
        pieces = []
        n_out = 0
        for chunk in stream_generate(
            model, tokenizer, chat, max_tokens=args.max_tokens
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            pieces.append(chunk.text)
            n_out += 1
        res.ttft_s.append(ttft)
        completions.append("".join(pieces))
        total_out += n_out
    res.wall_s = time.perf_counter() - t_all
    res.output_tokens = total_out
    res.peak_mem_gb = mx.get_peak_memory() / 1024**3
    res.completions = completions
    res.finalize()
    return res


# ---------------------------------------------------------------------------
# Mode (b): server -- existing polar_llama async fan-out via OPENAI_BASE_URL
# ---------------------------------------------------------------------------


def run_server(prompts: list[str], args) -> ModeResult:
    res = ModeResult(mode="server", ok=False, rows=len(prompts))
    if not args.base_url:
        res.note = (
            "skipped: no --base-url. Start `python -m mlx_lm server --model "
            f"{args.model} --port 8080` and pass --base-url "
            "http://127.0.0.1:8080"
        )
        return res

    import urllib.request

    # Fail fast with a clear message if nothing is listening.
    try:
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/models", method="GET"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:  # noqa: BLE001 - report any connectivity failure
        res.note = f"skipped: server not reachable at {args.base_url}: {e}"
        return res

    try:
        import polars as pl

        import polar_llama  # noqa: F401 - registers the .llama namespace
        from polar_llama.local.server_backend import inference_local_server
    except ImportError as e:
        res.note = f"skipped: polar_llama not importable ({e})"
        return res

    os.environ["OPENAI_BASE_URL"] = args.base_url
    # mlx_lm.server ignores auth but the OpenAI client requires a key header.
    os.environ.setdefault("OPENAI_API_KEY", "local-not-used")

    df = pl.DataFrame({"prompt": prompts})
    rss0 = _rss_gb()
    t0 = time.perf_counter()
    # Exercise the SHIPPED buy-path (polar_llama.local.server_backend), which
    # builds an explicit [system, user] message array so the full system prompt
    # is sent on every row. inference_async's system_prompt kwarg is gated on
    # cache=True and silently drops the prompt otherwise (only ~53 tokens/row
    # reached the server, making earlier server numbers unfairly fast).
    out = df.with_columns(
        completion=inference_local_server(
            pl.col("prompt"),
            model=args.server_model or args.model,
            system=SYSTEM_PROMPT,
            base_url=args.base_url,
        )
    )
    res.wall_s = time.perf_counter() - t0
    res.completions = out["completion"].to_list()
    errors = [
        c for c in res.completions if c and c.startswith('{"_error"')
    ]
    if errors:
        res.note = f"{len(errors)}/{len(prompts)} rows returned error JSON"
    res.ok = len(errors) < len(prompts)
    res.ttft_s = None  # not observable per-row through the Rust fan-out
    res.peak_mem_gb = None  # model lives in the server process
    res.note = (res.note + " " if res.note else "") + (
        f"client RSS delta {_rss_gb() - rss0:.2f} GB; server memory not "
        "instrumented -- read it from `mlx_lm.server` logs or Activity Monitor"
    )
    return res


def count_server_tokens(res: ModeResult, tokenizer) -> None:
    """Token-count server completions with the local tokenizer (approximate:
    same tokenizer family, but the server may inject its own template)."""
    if res.ok and tokenizer is not None and res.completions:
        good = [
            c
            for c in res.completions
            if c and not c.startswith('{"_error"')
        ]
        res.output_tokens = _count_tokens(tokenizer, good)
        res.finalize()


# ---------------------------------------------------------------------------
# Mode (c): in-process mlx_lm.batch_generate with replicated prefix caches
# ---------------------------------------------------------------------------


def run_in_process(model, tokenizer, prompts: list[str], args) -> ModeResult:
    """Verified against mlx-lm 0.31.3 (smoke test, 2026-07-04):

    - ``batch_generate(model, tokenizer, prompts: List[List[int]], ...)``
      takes TOKEN IDS, not strings (strings die with ZeroDivisionError).
    - Returns ``BatchResponse`` with ``.texts``, ``.caches`` and ``.stats``
      (``BatchStats``: prompt_tokens/generation_tokens/peak_memory/...).
    - ``return_prompt_caches=True`` surfaces caches on ``.caches`` (there is
      NO ``.prompt_caches`` attribute); each entry is a per-layer list of
      KVCache objects and ``copy.deepcopy`` replication round-trips.
    """
    import copy

    import mlx.core as mx
    from mlx_lm import batch_generate

    res = ModeResult(mode="in_process", ok=True, rows=len(prompts))
    mx.reset_peak_memory()

    chats = [
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

    t0 = time.perf_counter()
    if not args.no_prefix_cache:
        # Warm the shared system-prompt prefix once, then replicate the
        # resulting KV cache across all rows so per-row prefill only covers
        # the short user suffix.
        sys_prefix = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}],
            add_generation_prompt=False,
            tokenize=True,
        )
        warm = batch_generate(
            model,
            tokenizer,
            [sys_prefix],
            max_tokens=1,
            verbose=False,
            return_prompt_caches=True,
        )
        base_cache = warm.caches[0]
        caches = [base_cache] + [
            copy.deepcopy(base_cache) for _ in range(len(chats) - 1)
        ]
        result = batch_generate(
            model,
            tokenizer,
            chats,
            max_tokens=args.max_tokens,
            verbose=False,
            prompt_caches=caches,
        )
        res.note = "replicated system-prefix prompt_caches"
    else:
        result = batch_generate(
            model, tokenizer, chats, max_tokens=args.max_tokens, verbose=False
        )
        res.note = "prefix cache disabled; plain batch_generate"
    res.wall_s = time.perf_counter() - t0

    res.completions = list(result.texts)
    stats = getattr(result, "stats", None)
    res.output_tokens = (
        stats.generation_tokens
        if stats is not None
        else _count_tokens(tokenizer, res.completions)
    )
    res.peak_mem_gb = mx.get_peak_memory() / 1024**3
    # batch_generate interleaves all rows; the first decode step lands after
    # the (batched) prefill, so a single shared TTFT is the honest number.
    res.ttft_s = None
    res.note += "; per-row TTFT needs BatchGenerator streaming (not measured)"
    res.finalize()
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(results: list[ModeResult]) -> None:
    cols = [
        ("mode", 12),
        ("ok", 4),
        ("rows", 5),
        ("wall_s", 9),
        ("eff_tok/s", 10),
        ("ttft_p50_s", 11),
        ("peak_gb", 8),
    ]
    header = " | ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        vals = [
            r.mode,
            "yes" if r.ok else "no",
            str(r.rows),
            f"{r.wall_s:.2f}" if r.wall_s else "-",
            f"{r.effective_tok_s:.1f}" if r.effective_tok_s else "-",
            f"{_p50(r.ttft_s):.3f}" if _p50(r.ttft_s) else "-",
            f"{r.peak_mem_gb:.2f}" if r.peak_mem_gb else "-",
        ]
        print(" | ".join(v.ljust(w) for v, (_, w) in zip(vals, cols)))
    for r in results:
        if r.note:
            print(f"  [{r.mode}] {r.note}")

    done = [r for r in results if r.ok and r.wall_s]
    if len(done) >= 2:
        by = {r.mode: r for r in done}
        if "server" in by and "in_process" in by:
            ratio = by["server"].wall_s / by["in_process"].wall_s
            print(
                f"\nGATE: server wall-clock is {ratio:.2f}x in_process "
                f"({'within' if ratio <= 1.20 else 'OUTSIDE'} the 20% "
                "buy-threshold; see docs/local_mlx_gate_decision.md)"
            )


def check_parity(results: list[ModeResult]) -> Optional[str]:
    """Row-order sanity: every mode must return one completion per row."""
    for r in results:
        if r.ok and len(r.completions) != r.rows:
            return f"{r.mode}: {len(r.completions)} completions for {r.rows} rows"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Full docs in the module docstring (python -c 'import ...').",
    )
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="MLX model id (default: tiny harness-validation model; run the "
        "real gate with mlx-community Gemma 4 E4B / Ornith-1.0-9B 4-bit)",
    )
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--base-url",
        default=os.environ.get("POLAR_LLAMA_BENCH_BASE_URL"),
        help="local OpenAI-compatible server for the 'server' mode "
        "(e.g. http://127.0.0.1:8080); omitted => server mode skipped",
    )
    ap.add_argument(
        "--server-model",
        default=None,
        help="model name the server expects, if it differs from --model",
    )
    ap.add_argument(
        "--modes",
        default="sequential,server,in_process",
        help="comma list of modes to run",
    )
    ap.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="disable prompt_caches replication in in_process mode",
    )
    ap.add_argument("--json", default=None, help="write results JSON here")
    ap.add_argument(
        "--patch-1384",
        action="store_true",
        help="apply polar_llama's runtime monkeypatch for mlx-lm #1384 "
        "(fixes hybrid Gemma batched shared-KV garbage) before loading",
    )
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    prompts = build_prompts(args.rows)
    results: list[ModeResult] = []

    try:
        import mlx.core  # noqa: F401
        import mlx_lm

        have_mlx = True
    except ImportError:
        have_mlx = False

    if args.patch_1384 and have_mlx:
        import importlib.util
        from pathlib import Path

        _pp = Path(__file__).resolve().parent.parent / "polar_llama/local/_mlx_patches.py"
        _spec = importlib.util.spec_from_file_location("_mlx_patches", _pp)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        print(f"[#1384 patch] applied = {_mod.apply_gemma3n_batched_shared_kv_patch()}")

    if not have_mlx and "server" not in modes:
        print(
            "mlx / mlx-lm not installed -- benchmark skipped.\n"
            "This harness needs Apple silicon and the optional local extra:\n"
            "    pip install polar-llama[local]   (or: uv pip install mlx mlx-lm)\n"
            f"It would have run modes {modes} over {args.rows} rows against "
            f"{args.model}.",
            file=sys.stderr,
        )
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"skipped": "mlx not installed"}, f)
        return 0

    print(f"platform: {platform.platform()} / {platform.machine()}")
    print(f"model: {args.model}  rows: {args.rows}  max_tokens: {args.max_tokens}")

    model = tokenizer = None
    load_s = None
    if have_mlx:
        # tokenizer is also used to token-count server completions
        t0 = time.perf_counter()
        try:
            model, tokenizer = mlx_lm.load(args.model)
            load_s = time.perf_counter() - t0
            print(f"loaded in {load_s:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(
                f"could not load {args.model}: {e}\n"
                "Pass a downloadable mlx-community model id.",
                file=sys.stderr,
            )
            return 0 if "server" not in modes else 1
    else:
        # server-only run without mlx: wall-clock still measured, but
        # effective tok/s is unavailable (no local tokenizer to count with).
        skipped = {"sequential", "in_process"} & set(modes)
        if skipped:
            print(
                f"mlx not installed: skipping modes {sorted(skipped)}; "
                "running server mode only (no effective tok/s)",
                file=sys.stderr,
            )
        modes = ["server"]

    def _safe(mode: str, fn):
        # Isolate per-mode failures: a crash in one mode (e.g. an upstream
        # mlx-lm bug on a hybrid model's batched cache) must not discard the
        # other modes' results or the summary table/JSON.
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc().strip().splitlines()
            r = ModeResult(mode=mode, ok=False, rows=len(prompts))
            r.note = f"FAILED: {type(e).__name__}: {e} | at {tb[-2] if len(tb) > 1 else ''}"
            print(f"{mode} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            return r

    if "sequential" in modes and model is not None:
        r = _safe("sequential", lambda: run_sequential(model, tokenizer, prompts, args))
        r.load_s = load_s
        results.append(r)

    if "in_process" in modes and model is not None:
        r = _safe("in_process", lambda: run_in_process(model, tokenizer, prompts, args))
        r.load_s = load_s
        results.append(r)

    if "server" in modes:
        r = _safe("server", lambda: run_server(prompts, args))
        if r.ok:
            count_server_tokens(r, tokenizer)
        results.append(r)

    parity = check_parity(results)
    if parity:
        print(f"PARITY FAILURE: {parity}", file=sys.stderr)

    print_table(results)

    if args.json:
        payload = {
            "model": args.model,
            "rows": args.rows,
            "max_tokens": args.max_tokens,
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "platform": platform.platform(),
            "parity_failure": parity,
            "results": [r.to_json() for r in results],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")

    return 1 if parity else 0


if __name__ == "__main__":
    sys.exit(main())
