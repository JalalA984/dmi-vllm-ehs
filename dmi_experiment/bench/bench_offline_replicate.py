#!/usr/bin/env python3
"""bench_offline_replicate.py — DMI vs vLLM-baseline offline batch benchmark.

Variant of bench_offline.py for the replicate experiment:
  * 256 ShareGPT prompts sampled with the SAME logic + SAME seed as the
    teammate's vLLM-extract bench. Source file at
    /scratch/zt1/project/zaoxing-prj/shared/ShareGPT_V3_unfiltered_cleaned_split.json
    (sampling logic mirrors shared/make_prompts_test.py).
  * Designed to be invoked once per (mode, batch_size, trial) cell. The
    sbatch (bench_replicate.sbatch) wraps the full sweep in a 4-trial
    outer loop so we end up with 4 samples per cell -> mean +/- stddev.
  * Output goes to results_replicate.csv by default (separate file from
    the original results.csv so the previously-published numbers are
    untouched).

Same vLLM/DMI plumbing as bench_offline.py; only the prompt source and
defaults differ. Trial number is propagated through the --tag field so
we don't have to modify the CSV schema.
"""

from __future__ import annotations

from transformers import PretrainedConfig
if not hasattr(PretrainedConfig, "standardize_rope_params"):
    PretrainedConfig.standardize_rope_params = lambda self: None
if not hasattr(PretrainedConfig, "validate_rope"):
    PretrainedConfig.validate_rope = lambda self: None


import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

# Per DMI README + run_dmi.sh, these env vars must be set before importing vllm.
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")


def load_sharegpt_prompts(json_path: str, num_prompts: int, seed: int = 42) -> list[str]:
    """Mirror shared/make_prompts_test.py exactly so DMI and vLLM-extract see
    byte-identical prompt lists. Tested locally (per teammate) to be
    deterministic across runs at seed=42."""
    with open(json_path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    prompts: list[str] = []
    while len(prompts) < num_prompts:
        conv = data[rng.randint(0, len(data) - 1)]
        if len(conv["conversations"]) > 0:
            turn = conv["conversations"][rng.randint(0, len(conv["conversations"]) - 1)]
            if turn["from"] == "human":
                prompts.append(turn["value"])
    return prompts


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

    # Prompt source — defaults match the shared/make_prompts_test.py setup.
    p.add_argument("--prompts-json",
                   default="/scratch/zt1/project/zaoxing-prj/shared/"
                           "ShareGPT_V3_unfiltered_cleaned_split.json",
                   help="ShareGPT JSON to sample from. Sampling logic exactly "
                        "mirrors shared/make_prompts_test.py.")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-warmups", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1,
                   help="Apples-to-apples with vLLM-extract (prompt-only capture).")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="vLLM scheduler concurrency cap (functions as batch size).")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="Bumped from 4096 vs original bench because ShareGPT "
                        "human turns are more variable in length than the "
                        "synthetic prompts. 8192 covers the bulk of ShareGPT "
                        "without inflating KV cache budget too much. If a "
                        "rare long prompt errors, bump this and rerun.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graph capture (debug only).")
    p.add_argument("--dtype", default="auto")

    # DMI-only args (ignored when --mode vanilla)
    p.add_argument("--hook-selection", default="vllm-full",
                   help="DMI hook selection. 'vllm-full' = all hooks except "
                        "attn weight matrices (FlashAttention can't materialize "
                        "those). 'full' is HF-only.")
    p.add_argument("--null-mode", action="store_true",
                   help="DMI null mode — capture + ring transport, skip "
                        "ClickHouse insert.")
    p.add_argument("--ring-payload-mb", type=int, default=4096)
    p.add_argument("--ring-pinned-mb", type=int, default=4096)

    p.add_argument("--output-csv", default="results_replicate.csv")
    p.add_argument("--tag", default="",
                   help="Row label for grouping runs. We use 'trialN_bsM' "
                        "from the sbatch so trial+batch are recoverable from "
                        "the CSV without a schema change.")
    args = p.parse_args()

    print(f"[bench_offline_replicate] start  mode={args.mode}  model={args.model}", flush=True)
    print(f"[bench_offline_replicate] num_prompts={args.num_prompts}  "
          f"max_num_seqs={args.max_num_seqs}  max_tokens={args.max_tokens}  "
          f"warmups={args.num_warmups}  tag={args.tag}", flush=True)
    print(f"[bench_offline_replicate] prompts_json={args.prompts_json}", flush=True)
    if args.mode == "dmi":
        print(f"[bench_offline_replicate] hook_selection={args.hook_selection}  "
              f"null_mode={args.null_mode}", flush=True)

    # Load prompts BEFORE building the LLM so a missing file fails fast
    # (saves the ~2 min model-build wall time on a typo).
    if not Path(args.prompts_json).is_file():
        raise SystemExit(f"prompts JSON not found: {args.prompts_json}")
    prompts = load_sharegpt_prompts(args.prompts_json, args.num_prompts, seed=args.seed)
    char_lens = [len(p) for p in prompts]
    print(f"[bench_offline_replicate] sampled {len(prompts)} prompts  "
          f"char-len min/avg/max = {min(char_lens)}/{sum(char_lens)//len(prompts)}/{max(char_lens)}",
          flush=True)
    # Quick guard: warn if any prompt's char count is suspiciously close to
    # max_model_len * ~4 chars/token. Tokenization happens later in vLLM; this
    # is a heuristic so we know to expect a 'too long' error.
    char_threshold = args.max_model_len * 4
    long_prompts = [i for i, n in enumerate(char_lens) if n > char_threshold]
    if long_prompts:
        print(f"[bench_offline_replicate] WARN: {len(long_prompts)} prompts "
              f"have > {char_threshold} chars; may exceed --max-model-len "
              f"{args.max_model_len}. Indices: {long_prompts[:5]}...", flush=True)

    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    print("[bench_offline_replicate] building LLM ...", flush=True)
    t_b0 = time.perf_counter()
    llm = build_llm(args)
    build_seconds = time.perf_counter() - t_b0
    print(f"[bench_offline_replicate] LLM ready in {build_seconds:.1f}s", flush=True)

    # Warmup — captures CUDA graphs, exercises hook installation.
    if args.num_warmups > 0:
        warm_n = min(args.num_prompts, max(8, args.max_num_seqs or 8))
        warm = prompts[:warm_n]
        print(f"[bench_offline_replicate] warmup: {args.num_warmups} pass(es) "
              f"of {warm_n} prompts", flush=True)
        t_w0 = time.perf_counter()
        for _ in range(args.num_warmups):
            _ = llm.generate(warm, params, use_tqdm=False)
        warmup_seconds = time.perf_counter() - t_w0
        print(f"[bench_offline_replicate] warmup done in {warmup_seconds:.2f}s", flush=True)
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
            print(f"[bench_offline_replicate] stop_monitoring failed (non-fatal): {e}",
                  flush=True)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompts_per_s = args.num_prompts / bench_seconds
    prompt_tok_per_s = total_prompt_tokens / bench_seconds

    print(f"[bench_offline_replicate] DONE  bench={bench_seconds:.3f}s  "
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

    print(f"[bench_offline_replicate] appended row to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
