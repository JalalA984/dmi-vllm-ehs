#!/bin/bash
# rebuild_vllm.sbatch — source-build vLLM 0.17 against local glibc.
#
# Why: build_env_v3.sbatch used VLLM_USE_PRECOMPILED=1, which downloads
# prebuilt _C.abi3.so from vLLM's CDN. Those binaries link against
# GLIBC_2.34. Zaratan compute nodes have glibc 2.28. The build's smoke
# test (`import vllm`) didn't catch this because _C is lazy-loaded only
# on the cuda platform code path — never triggered on the CPU build node.
#
# Fix: rebuild from source with VLLM_USE_PRECOMPILED unset. The compiler
# links _C against the build node's glibc 2.28, which matches GPU nodes.
#
# Run after this completes: sbatch bench.sbatch

#SBATCH --job-name=vllm-rebuild
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=rebuild-%j.out
#SBATCH --error=rebuild-%j.err

set -uo pipefail

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date) ==="
ldd --version | head -1

# ── Toolchain ───────────────────────────────────────────────────────
module load cuda/12.3.0/gcc cmake/3.26.3/gcc gcc/11.3.0
nvcc --version
echo

# ── Build env vars ──────────────────────────────────────────────────
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}   # libcuda stub for link
export MAX_JOBS=12   # ninja parallelism — slightly under cpus-per-task to leave headroom for nvcc forks
export VLLM_TARGET_DEVICE=cuda
unset VLLM_USE_PRECOMPILED                                      # CRITICAL — true source build

# ── Pin to A100 (sm_80) only ────────────────────────────────────────
# Why: default arch list includes sm_89 (Ada/RTX 4090). vLLM 0.17 ships
# Marlin FP8 quant kernels (sm89_kernel_fe4m3fn_*.cu) whose ptx requires
# CUDA ≥12.4 ptxas; Zaratan only has CUDA 12.3, so those kernels fail
# with "Unexpected instruction types specified for 'mma'". A100 doesn't
# use them anyway — restrict to 8.0 so they're never compiled.
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_ARCHS="80"                                          # vllm-specific override (belt + suspenders)

# Backup channel — if vllm's setup.py ever honors CMAKE_ARGS, this also helps.
# The primary mechanism is the patch to CMakeLists.txt that sets
# FETCHCONTENT_FULLY_DISCONNECTED ON at the cmake-source level.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DFETCHCONTENT_FULLY_DISCONNECTED=ON"

echo "CUDA_HOME=$CUDA_HOME"
echo "MAX_JOBS=$MAX_JOBS"
echo "VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED:-<unset>}"
echo "CMAKE_ARGS=$CMAKE_ARGS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CUDA_ARCHS=$CUDA_ARCHS"
echo

# ── Sanity check: pre-fetched deps must exist before we even try ────
VLLM_SRC=~/scratch.zaoxing-prj/vllm-0.17.0
for d in cutlass-src vllm-flash-attn-src triton_kernels-src flashmla-src qutlass-src; do
    if [ ! -d "$VLLM_SRC/.deps/$d" ] || [ -z "$(ls -A "$VLLM_SRC/.deps/$d" 2>/dev/null)" ]; then
        echo "FATAL: $VLLM_SRC/.deps/$d missing or empty"
        echo "Compute nodes have no internet — cmake cannot recover by re-fetching."
        echo "Pre-clone on the login node before re-submitting this job."
        exit 1
    fi
done
echo "[rebuild] all .deps/*-src dirs present and non-empty"

# ── Verify FETCHCONTENT_FULLY_DISCONNECTED patch is in CMakeLists.txt ──
if ! grep -q "FETCHCONTENT_FULLY_DISCONNECTED" "$VLLM_SRC/CMakeLists.txt"; then
    echo "FATAL: $VLLM_SRC/CMakeLists.txt is missing the FETCHCONTENT_FULLY_DISCONNECTED patch."
    echo "Apply via: sed -i '2i\\set(FETCHCONTENT_FULLY_DISCONNECTED ON)' $VLLM_SRC/CMakeLists.txt"
    exit 1
fi
echo "[rebuild] CMakeLists.txt has FETCHCONTENT_FULLY_DISCONNECTED patch"
echo

# ── Conda env (already exists from build_env_v3) ────────────────────
source ~/scratch.zaoxing-prj/miniconda3/etc/profile.d/conda.sh
conda activate vllm-exp
which python && python -V

DMI=~/scratch.zaoxing-prj/DMI
VLLM_SRC=~/scratch.zaoxing-prj/vllm-0.17.0
echo "VLLM_SRC=$VLLM_SRC"

# ── Wipe previous vllm install + build artifacts ────────────────────
echo "[rebuild] uninstalling existing vllm ..."
pip uninstall -y vllm || true

echo "[rebuild] cleaning source build dir ..."
rm -rf "$VLLM_SRC/build" "$VLLM_SRC/dist" "$VLLM_SRC/vllm.egg-info"
find "$VLLM_SRC" -name "*.so" -delete 2>/dev/null || true

# ── Source build (the slow step — ~30-60 min on 4 cores) ───────────
echo "[rebuild] starting source build ..."
time pip install --no-cache-dir "$VLLM_SRC"

# ── Re-symlink .so files into the integration source tree ──────────
echo "[rebuild] re-symlinking .so files into integration/vllm/ ..."
VLLM_SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
VLLM_INTEG="$DMI/integration/vllm/vllm"
for so in "$VLLM_SITE"/*.so; do
    [ -f "$so" ] && ln -sf "$so" "$VLLM_INTEG/$(basename "$so")" && echo "  linked $(basename "$so")"
done
if [ -d "$VLLM_SITE/vllm_flash_attn" ]; then
    mkdir -p "$VLLM_INTEG/vllm_flash_attn"
    for so in "$VLLM_SITE"/vllm_flash_attn/*.so; do
        [ -f "$so" ] && ln -sf "$so" "$VLLM_INTEG/vllm_flash_attn/$(basename "$so")"
    done
fi
[ -f "$VLLM_SITE/_version.py" ] && ln -sf "$VLLM_SITE/_version.py" "$VLLM_INTEG/_version.py"

# ── Smoke test — THIS time actually load _C ─────────────────────────
echo "[rebuild] verifying _C loads (the test build_env_v3 missed) ..."
python -c "
import vllm
print('vllm version:', vllm.__version__)
import vllm._C   # the failing import from the bench job
print('vllm._C imported OK')
from vllm import LLM, SamplingParams
print('LLM, SamplingParams imported OK')
from monitoring.vllm_integration import DMXGPUWorker
print('DMXGPUWorker imported OK')
"

echo "=== rebuild complete: $(date) ==="
echo "Next: sbatch bench.sbatch"
