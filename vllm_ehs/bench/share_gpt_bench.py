import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm import LLM, SamplingParams
import torch
import tempfile
import json

KVConnectorFactory.register_connector(
    "NoWritesConnector",
    "NoWritesConnector",
    "NoWritesConnector",
)

def make_prompts(num_prompts: int, seed: int = 42) -> list[str]:
    with open("/shared/ShareGPT_V3_unfiltered_cleaned_split.json") as f:
        data = json.load(f)

    rng = random.Random(seed)
    prompts = []
    while len(prompts) < num_prompts:
            conv = data[rng.randint(0, len(data) - 1)]
            if len(conv['conversations']) > 0:      
                    turn = conv['conversations'][rng.randint(0, len(conv['conversations']) - 1)]
                    if turn['from'] == 'human':
                            prompts.append(turn['value'])

    return prompts


def build_llm(tmpdirname, args):
    

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [  # Target model layer indices
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
                    ],
                }
            },
        },
        kv_transfer_config={
            "kv_connector": ("NoWritesConnector" if args.no_write else "ExampleHiddenStatesConnector"),
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmpdirname,
            },
        },
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    return LLM(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HF model id or local path (snapshot dir)")

    p.add_argument("--num-prompts", type=int, default=256)
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

    p.add_argument("--output-csv", default="results.csv")
    p.add_argument("--tag", default="", help="optional row label for grouping runs")
    p.add_argument("--no-write", action='store_true',
                   help="If True, uses NoWritesConnector and does nor write anything to disk. " \
                   "By default, writes to disk using ExampleHiddenStatesConnector")
    args = p.parse_args()

    print(f"[bench_offline] start model={args.model}", flush=True)
    print(f"[bench_offline] num_prompts={args.num_prompts}  max_num_seqs={args.max_num_seqs}  "
          f"max_tokens={args.max_tokens}  warmups={args.num_warmups}", flush=True)

    prompts = make_prompts(args.num_prompts, seed=args.seed)

    
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print("[bench_offline] building LLM ...", flush=True)
        t_b0 = time.perf_counter()
        llm = build_llm(tmpdirname, args)
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
                "dtype", "tp_size", "seed", "max_model_len",
                "num_warmups", "build_seconds", "warmup_seconds", "bench_seconds",
                "total_prompt_tokens", "total_gen_tokens", "avg_prompt_len",
                "prompts_per_s", "prefill_tokens_per_s",
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            "vllm-no-write" if args.no_write else "vllm-plain",
            args.tag,
            args.model,
            args.num_prompts,
            args.max_num_seqs if args.max_num_seqs is not None else "",
            args.max_tokens,
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