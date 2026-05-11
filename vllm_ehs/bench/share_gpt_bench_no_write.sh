#!/bin/bash
# Result CSV:    ~/scratch.zaoxing-prj/results.csv (appended each invocation)
#
# Per CLAUDE.md decisions:
#   - Model:           Qwen3-4B
#   - max_tokens:      1 (apples-to-apples for vLLM-extract's prompt-only capture)
#   - Batch size sweep: {1, 4, 8, 16, 32}

#SBATCH --job-name=dmi-bench
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=bench-%j.out
#SBATCH --error=bench-%j.err

set -euo pipefail

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date) ==="

# ── Toolchain ───────────────────────────────────────────────────────
module load apptainer

# ── Sweep parameters ────────────────────────────────────────────────
BATCH_SIZES=(1 4 8 16 32)
NUM_PROMPTS=64
MAX_MODEL_LEN=8192
NUM_SWEEPS=4
SHARED="/scratch/zt1/project/zaoxing-prj/shared"
DRIVER="share_gpt_bench.py"
RESULTS="/shared/CMSC818Q_proj/results-share-no-write-${SLURM_JOB_ID}.csv"
touch "${SHARED}"/CMSC818Q_proj/results-share-no-write-"${SLURM_JOB_ID}".csv

# if [ ! -f "$DRIVER" ]; then
#     echo "ERROR: driver not found at $DRIVER"
#     exit 1
# fi

echo
echo "========== Sweep starts =========="
echo "Driver:  $DRIVER"
echo "Results: {$SHARED}/results.csv"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Num prompts per run: $NUM_PROMPTS"
echo

cd ~/scratch

for i in $(seq 1 ${NUM_SWEEPS});
do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "---------- batch_size=$bs ----------"
        apptainer exec --nv --bind "$SHARED":/shared \
            vllm-openai_v0.18.1.sif \
            python3 "$DRIVER" \
            --model /shared/qwen3-4b \
            --max-num-seqs "$bs" \
            --num-prompts "$NUM_PROMPTS" \
            --max-tokens 1 \
            --max-model-len "$MAX_MODEL_LEN" \
            --num-warmups 2 \
            --gpu-memory-utilization 0.85 \
            --output-csv "$RESULTS" \
            --tag "sweep_bs${bs}" \
            --write-to-disk False

        echo
    done
done

echo "========== Sweep complete: $(date) =========="
echo "Results CSV: $RESULTS"
echo "Final 12 rows:"
tail -n 12 "$RESULTS"