#!/bin/bash
# bench.sbatch — sweep batch sizes, run vanilla vLLM + DMI null-mode for each.
#
# Submit with:   sbatch bench.sbatch
# Live log:      tail -f bench-<JOBID>.out
# Result CSV:    ~/scratch.zaoxing-prj/results.csv (appended each invocation)
#
# Per CLAUDE.md decisions:
#   - Model:           Qwen3-4B
#   - max_tokens:      1 (apples-to-apples for vLLM-extract's prompt-only capture)
#   - DMI hook mode:   vllm-full (Option B unmodified; vllm-full = full minus
#                                 attn_scores/pattern, which FlashAttention
#                                 cannot materialize anyway)
#   - DMI storage:     null-mode (transport-only; ClickHouse off, per asymmetric
#                                 storage parity decision)
#   - Batch size sweep: {1, 4, 8, 16, 32}

#SBATCH --job-name=dmi-bench
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=bench-%j.out
#SBATCH --error=bench-%j.err

set -euo pipefail

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date) ==="

# ── Toolchain ───────────────────────────────────────────────────────
module load cuda/12.3.0/gcc cmake/3.26.3/gcc gcc/11.3.0
nvidia-smi
echo

# ── Conda env (vllm-exp built by setup_env.sh baseline) ─────────────
source ~/scratch.zaoxing-prj/miniconda3/etc/profile.d/conda.sh
conda activate vllm-exp
which python && python -V

# ── DMI repo + paths (mirrors run_dmi.sh) ───────────────────────────
DMI=~/scratch.zaoxing-prj/DMI
WORK_ROOT=~/scratch.zaoxing-prj
cd "$WORK_ROOT"

export PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/integration/transformers/src${PYTHONPATH:+:$PYTHONPATH}

# ── Force torch's bundled NVIDIA CUDA libs (12.8.x) ahead of the cuda/12.3.0
# module's libs.  monitoring_native_backend.so transitively loads torch's
# libtorch_cuda → libcusparse → libnvJitLink and torch's libc10_cuda → libcudart.
# Both call CUDA 12.5+/12.8 symbols (cudaGetDriverEntryPointByVersion,
# __nvJitLinkCreate_12_8) that the system cuda 12.3 libs don't have.
# We've also patchelf'd the .so to use DT_RUNPATH (not DT_RPATH) so this
# LD_LIBRARY_PATH actually wins.
TORCH_NV_LIBS=$(python -c "
import os, sys
nv = os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia')
if os.path.isdir(nv):
    print(':'.join(os.path.join(nv, d, 'lib') for d in sorted(os.listdir(nv))
                   if os.path.isdir(os.path.join(nv, d, 'lib'))))
")
# $CONDA_PREFIX/lib must come BEFORE the gcc/11.3.0 module's lib64, otherwise
# /cvmfs/.../gcc-11.3.0/lib64/libstdc++.so.6 wins — and that lib (gcc 11.3) is
# missing GLIBCXX_3.4.30, which torch's libs (built with gcc 12+) require.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$TORCH_NV_LIBS:$DMI/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VLLM_TARGET_DEVICE=cuda
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_MODULE_LOADING=EAGER
export HF_HOME=$WORK_ROOT/hf_cache              # vLLM scratch only; model loaded from local path
export HF_HUB_OFFLINE=1                         # defensive — compute nodes can't reach HF anyway
export XDG_CACHE_HOME=$WORK_ROOT/.cache
export VLLM_CACHE_ROOT=$WORK_ROOT/vllm_cache

# ── Qwen3-4B local snapshot (downloaded via Globus to shared/) ──────
# Both teammates point at this same path — no per-user download.
MODEL_PATH=/scratch/zt1/project/zaoxing-prj/shared/qwen3-4b
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Qwen3-4B not found at $MODEL_PATH"
    echo "Re-run the Globus transfer or check the path."
    exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: $MODEL_PATH exists but config.json is missing — partial download?"
    exit 1
fi
echo "MODEL_PATH=$MODEL_PATH"
ls -lh "$MODEL_PATH"/*.safetensors

# ── Sweep parameters ────────────────────────────────────────────────
BATCH_SIZES=(1 4 8 16 32)
NUM_PROMPTS=64
DRIVER="$WORK_ROOT/bench_offline.py"
RESULTS="$WORK_ROOT/results.csv"

if [ ! -f "$DRIVER" ]; then
    echo "ERROR: driver not found at $DRIVER"
    exit 1
fi

echo
echo "========== Sweep starts =========="
echo "Driver:  $DRIVER"
echo "Results: $RESULTS"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Num prompts per run: $NUM_PROMPTS"
echo

for bs in "${BATCH_SIZES[@]}"; do
    echo "---------- batch_size=$bs : VANILLA ----------"
    python "$DRIVER" \
        --mode vanilla \
        --model "$MODEL_PATH" \
        --max-num-seqs "$bs" \
        --num-prompts "$NUM_PROMPTS" \
        --max-tokens 1 \
        --num-warmups 2 \
        --gpu-memory-utilization 0.85 \
        --output-csv "$RESULTS" \
        --tag "sweep_bs${bs}"

    echo
    echo "---------- batch_size=$bs : DMI null-mode (vllm-full) ----------"
    python "$DRIVER" \
        --mode dmi \
        --model "$MODEL_PATH" \
        --max-num-seqs "$bs" \
        --num-prompts "$NUM_PROMPTS" \
        --max-tokens 1 \
        --num-warmups 2 \
        --gpu-memory-utilization 0.85 \
        --hook-selection vllm-full \
        --null-mode \
        --output-csv "$RESULTS" \
        --tag "sweep_bs${bs}"
    echo
done

echo "========== Sweep complete: $(date) =========="
echo "Results CSV: $RESULTS"
echo "Final 12 rows:"
tail -n 12 "$RESULTS"
