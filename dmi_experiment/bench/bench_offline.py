#!/usr/bin/env python3
"""bench_offline.py — DMI vs vLLM-baseline offline batch benchmark.

Runs the same set of prompts through ONE of two backends per invocation:
  --mode vanilla   — vanilla vLLM 0.17 (no observability). Reference run.
  --mode dmi       — DMI with DMXGPUWorker. Captures all hooks via the ring.
                     Use --null-mode to skip ClickHouse insert (transport-only).

Output: appends one row to results.csv per invocation.
Designed to be sweep-driven from an sbatch script (see bench.sbatch).

The vLLM PR #33736 (extract_hidden_states) benchmark lives in a SEPARATE env
(vllm-extract) and is owned by the teammate — not handled here.

Reference for the LLM(...) incantation: DMI/README.md §9a, plus
DMI/tests/vllm_monitored_runner.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

# Per DMI README + run_dmi.sh, these env vars must be set before importing vllm.
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")


_PROMPT_FRAGMENT = (
    "Below is a brief description of a topic. The history of computer "
    "science begins in the 19th century with Charles Babbage's Difference "
    "Engine and Ada Lovelace's pioneering programming. In the 20th century "
    "Alan Turing formalized computation through the Turing machine, "
    "establishing what is and is not computable. World War II accelerated "
    "computer development with machines like Colossus and ENIAC, used for "
    "code-breaking and ballistic computations. The transistor revolution "
    "in the 1950s shrank room-sized computers to manageable sizes. The "
    "integrated circuit and Moore's Law drove exponential growth in "
    "computing power throughout the latter half of the twentieth century. "
    "Personal computing arrived in the 1970s with the Altair 8800, Apple "
    "II, and IBM PC. The internet transformed communication and commerce "
    "in the 1990s. Today, machine learning models trained on vast datasets "
    "are reshaping software development. Considering this history, the "
    "question is: "
)


def make_prompts(num_prompts: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    out = []
    for i in range(num_prompts):
        n = rng.randint(0, 10**6)
        out.append(
            f"{_PROMPT_FRAGMENT}What is the answer to question {n} for prompt index {i}?"
        )
    return out


def build_llm(args):
    from vllm import LLM

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    if args.mode == "dmi":
        kwargs["worker_cls"] = "monitoring.vllm_integration.DMXGPUWorker"
        ac = {
            "dmx_hook_selection": args.hook_selection,
            "dmx_ring_payload_mb": args.ring_payload_mb,
            "dmx_ring_pinned_mb": args.ring_pinned_mb,
        }
        if args.null_mode:
            ac["dmx_null_mode"] = True
        kwargs["additional_config"] = ac

    return LLM(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["vanilla", "dmi"], required=True)
    p.add_argument("--model", required=True, help="HF model id or local path (snapshot dir)")

    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--num-warmups", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1,
                   help="Per CLAUDE.md §3, max_tokens=1 is the apples-to-apples "
                        "constraint for the DMI vs vLLM-extract comparison.")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="vLLM scheduler concurrency cap (functions as batch size).")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graph capture (debug only; on by default for fair comp).")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=42)

    # DMI-only args (ignored when --mode vanilla)
    p.add_argument("--hook-selection", default="vllm-full",
                   help="DMI hook selection. 'vllm-full' = all hooks except attn weight "
                        "matrices (FlashAttention can't materialize those). 'full' "
                        "is HF-only.")
    p.add_argument("--null-mode", action="store_true",
                   help="DMI null mode — capture + ring transport, skip ClickHouse "
                        "insert. Storage parity decision per CLAUDE.md §3.")
    p.add_argument("--ring-payload-mb", type=int, default=4096)
    p.add_argument("--ring-pinned-mb", type=int, default=4096)

    p.add_argument("--output-csv", default="results.csv")
    p.add_argument("--tag", default="", help="optional row label for grouping runs")
    args = p.parse_args()

    print(f"[bench_offline] start  mode={args.mode}  model={args.model}", flush=True)
    print(f"[bench_offline] num_prompts={args.num_prompts}  max_num_seqs={args.max_num_seqs}  "
          f"max_tokens={args.max_tokens}  warmups={args.num_warmups}", flush=True)
    if args.mode == "dmi":
        print(f"[bench_offline] hook_selection={args.hook_selection}  "
              f"null_mode={args.null_mode}  ring_payload_mb={args.ring_payload_mb}  "
              f"ring_pinned_mb={args.ring_pinned_mb}", flush=True)

    prompts = make_prompts(args.num_prompts, seed=args.seed)

    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    print("[bench_offline] building LLM ...", flush=True)
    t_b0 = time.perf_counter()
    llm = build_llm(args)
    build_seconds = time.perf_counter() - t_b0
    print(f"[bench_offline] LLM ready in {build_seconds:.1f}s", flush=True)

    # Warmup — captures CUDA graphs, exercises hook installation.
    if args.num_warmups > 0:
        warm_n = min(args.num_prompts, max(8, args.max_num_seqs or 8))
        warm = prompts[:warm_n]
        print(f"[bench_offline] warmup: {args.num_warmups} pass(es) of {warm_n} prompts", flush=True)
        t_w0 = time.perf_counter()
        for _ in range(args.num_warmups):
            _ = llm.generate(warm, params, use_tqdm=False)
        warmup_seconds = time.perf_counter() - t_w0
        print(f"[bench_offline] warmup done in {warmup_seconds:.2f}s", flush=True)
    else:
        warmup_seconds = 0.0

    # Timed run
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    bench_seconds = time.perf_counter() - t0

    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    avg_prompt_len = total_prompt_tokens / max(1, len(outputs))

    # Best-effort clean shutdown for DMI (flushes ring before vLLM's 8s kill timer).
    if args.mode == "dmi":
        try:
            llm.collective_rpc("stop_monitoring")
        except Exception as e:
            print(f"[bench_offline] stop_monitoring failed (non-fatal): {e}", flush=True)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompts_per_s = args.num_prompts / bench_seconds
    prompt_tok_per_s = total_prompt_tokens / bench_seconds

    print(f"[bench_offline] DONE  bench={bench_seconds:.3f}s  "
          f"prompts/s={prompts_per_s:.2f}  prefill_tok/s={prompt_tok_per_s:.0f}  "
          f"avg_prompt_len={avg_prompt_len:.0f}", flush=True)

    out_path = Path(args.output_csv)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "mode", "tag", "model",
                "num_prompts", "max_num_seqs", "max_tokens",
                "hook_selection", "null_mode",
                "dtype", "tp_size", "seed", "max_model_len",
                "num_warmups", "build_seconds", "warmup_seconds", "bench_seconds",
                "total_prompt_tokens", "total_gen_tokens", "avg_prompt_len",
                "prompts_per_s", "prefill_tokens_per_s",
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            args.mode,
            args.tag,
            args.model,
            args.num_prompts,
            args.max_num_seqs if args.max_num_seqs is not None else "",
            args.max_tokens,
            args.hook_selection if args.mode == "dmi" else "",
            args.null_mode if args.mode == "dmi" else "",
            args.dtype,
            args.tensor_parallel_size,
            args.seed,
            args.max_model_len,
            args.num_warmups,
            f"{build_seconds:.2f}",
            f"{warmup_seconds:.3f}",
            f"{bench_seconds:.3f}",
            total_prompt_tokens,
            total_gen_tokens,
            f"{avg_prompt_len:.1f}",
            f"{prompts_per_s:.3f}",
            f"{prompt_tok_per_s:.1f}",
        ])

    print(f"[bench_offline] appended row to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
