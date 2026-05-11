#!/bin/bash
# bench_replicate.sbatch — replicate sweep: 256 ShareGPT prompts, 4 trials.
#
# Variant of bench.sbatch for the replicate experiment. Differences vs the
# original:
#   * Outer loop on trial (1..4) wraps the existing batch-size sweep, so we
#     end up with 4 independent samples per (mode, batch_size) cell. Outer
#     trial loop (rather than inner) so each trial spans a full sweep cycle
#     -> trials are statistically independent rather than time-correlated.
#   * Driver: bench_offline_replicate.py (loads 256 ShareGPT prompts via the
#     same logic as shared/make_prompts_test.py, seed=42).
#   * Output CSV: results_replicate.csv (separate from results.csv so the
#     previously-published numbers stay untouched).
#   * --time bumped to 04:00:00 (4 trials * 5 batches * 2 modes = 40 python
#     invocations; each rebuilds the LLM, ~2 min build + warmup + bench, so
#     conservatively ~100 min total).
#   * --max-model-len bumped to 8192 (ShareGPT prompts are more variable in
#     length than the synthetic prompts; 8192 covers the bulk).

#SBATCH --job-name=dmi-bench-rep
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=bench-rep-%j.out
#SBATCH --error=bench-rep-%j.err

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

# Force torch's bundled NVIDIA CUDA libs (12.8.x) ahead of the cuda/12.3.0
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
export HF_HOME=$WORK_ROOT/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$WORK_ROOT/.cache
export VLLM_CACHE_ROOT=$WORK_ROOT/vllm_cache

# ── Qwen3-4B local snapshot ─────────────────────────────────────────
MODEL_PATH=/scratch/zt1/project/zaoxing-prj/shared/qwen3-4b
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Qwen3-4B not found at $MODEL_PATH"
    exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: $MODEL_PATH exists but config.json is missing"
    exit 1
fi
echo "MODEL_PATH=$MODEL_PATH"

# ── ShareGPT prompts file (must exist; bench errors out fast if not) ─
PROMPTS_JSON=/scratch/zt1/project/zaoxing-prj/shared/ShareGPT_V3_unfiltered_cleaned_split.json
if [ ! -f "$PROMPTS_JSON" ]; then
    echo "ERROR: ShareGPT JSON not found at $PROMPTS_JSON"
    exit 1
fi
echo "PROMPTS_JSON=$PROMPTS_JSON"
echo

# ── Sweep parameters ────────────────────────────────────────────────
BATCH_SIZES=(1 4 8 16 32)
NUM_PROMPTS=256
MAX_MODEL_LEN=8192
NUM_TRIALS=4
DRIVER="$WORK_ROOT/bench_offline_replicate.py"
RESULTS="$WORK_ROOT/results_replicate.csv"

if [ ! -f "$DRIVER" ]; then
    echo "ERROR: driver not found at $DRIVER"
    exit 1
fi

echo "========== Replicate sweep starts =========="
echo "Driver:        $DRIVER"
echo "Results:       $RESULTS"
echo "Trials:        $NUM_TRIALS"
echo "Batch sizes:   ${BATCH_SIZES[*]}"
echo "Prompts/run:   $NUM_PROMPTS"
echo "Max model len: $MAX_MODEL_LEN"
echo

for trial in $(seq 1 "$NUM_TRIALS"); do
    echo
    echo "########################################"
    echo "########  TRIAL $trial / $NUM_TRIALS  ##########"
    echo "########################################"
    for bs in "${BATCH_SIZES[@]}"; do
        echo
        echo "---------- trial=$trial  batch_size=$bs : VANILLA ----------"
        python "$DRIVER" \
            --mode vanilla \
            --model "$MODEL_PATH" \
            --prompts-json "$PROMPTS_JSON" \
            --num-prompts "$NUM_PROMPTS" \
            --max-num-seqs "$bs" \
            --max-tokens 1 \
            --max-model-len "$MAX_MODEL_LEN" \
            --num-warmups 2 \
            --gpu-memory-utilization 0.85 \
            --output-csv "$RESULTS" \
            --tag "trial${trial}_bs${bs}"

        echo
        echo "---------- trial=$trial  batch_size=$bs : DMI null-mode (vllm-full) ----------"
        python "$DRIVER" \
            --mode dmi \
            --model "$MODEL_PATH" \
            --prompts-json "$PROMPTS_JSON" \
            --num-prompts "$NUM_PROMPTS" \
            --max-num-seqs "$bs" \
            --max-tokens 1 \
            --max-model-len "$MAX_MODEL_LEN" \
            --num-warmups 2 \
            --gpu-memory-utilization 0.85 \
            --hook-selection vllm-full \
            --null-mode \
            --output-csv "$RESULTS" \
            --tag "trial${trial}_bs${bs}"
    done
done

echo
echo "========== Replicate sweep complete: $(date) =========="
echo "Results CSV: $RESULTS"
echo "Total rows expected: $((NUM_TRIALS * ${#BATCH_SIZES[@]} * 2))"
echo "Rows written:"
wc -l "$RESULTS"
echo
echo "Last 12 rows:"
tail -n 12 "$RESULTS"
