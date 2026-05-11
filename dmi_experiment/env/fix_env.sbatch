#!/bin/bash
# fix_env.sbatch — finish the env that rebuild_vllm.sbatch left half-done.
#
# Picks up where setup_env.sh baseline would have continued after pip install
# vllm: pin huggingface-hub, install flashinfer, build clickhouse-cpp + DMI
# monitoring native backend, symlink vllm .so files into the forked tree,
# `pip install --no-deps -e .` for DMI, then smoke-test imports.
#
# RUNS ON THE STANDARD PARTITION (no GPU needed for any step):
#   - clickhouse-cpp build is pure C++.
#   - monitoring's nvcc step compiles for sm_80 statically; doesn't need a GPU.
#   - smoke test only does Python imports, no LLM init.
#
# DOES NOT VALIDATE: actual GPU runtime.  That can only be done on gpu-a100.
# But if every import in the smoke test succeeds, the bench is highly likely
# to work on the GPU side.
#
# Submit:    sbatch fix_env.sbatch
# Read:      tail -f fix-env-<JOBID>.out

#SBATCH --job-name=dmi-fix-env
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=fix-env-%j.out
#SBATCH --error=fix-env-%j.err

set -uo pipefail

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-local}  start $(date) ==="

# ── Toolchain ────────────────────────────────────────────────────────
module load cuda/12.3.0/gcc cmake/3.26.3/gcc gcc/11.3.0
which nvcc && nvcc --version | head -n4
echo

# ── Conda env ────────────────────────────────────────────────────────
source ~/scratch.zaoxing-prj/miniconda3/etc/profile.d/conda.sh
conda activate vllm-exp
which python && python -V

# ── Paths ────────────────────────────────────────────────────────────
DMI=~/scratch.zaoxing-prj/DMI
WORK_ROOT=~/scratch.zaoxing-prj
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
echo "CUDA_HOME=$CUDA_HOME"
# libcuda stub on the link path so the .so links cleanly on a node with no driver.
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"

step() { echo; echo "================ STEP: $* ================"; }

# ── 1. huggingface-hub pin (root cause of bench-19190114) ────────────
step "1/6 pin huggingface-hub==1.0.0rc2"
pip install "huggingface-hub==1.0.0rc2" || { echo "FAIL: hub pin"; exit 11; }

# ── 2. flashinfer (vLLM 0.17 attention backend) ──────────────────────
step "2/6 install flashinfer-python==0.6.4"
pip install flashinfer-python==0.6.4 || { echo "FAIL: flashinfer install"; exit 12; }

# ── 3. clickhouse-cpp static lib ─────────────────────────────────────
step "3/6 build clickhouse-cpp"
CH_LIB="$DMI/libs/clickhouse-cpp/build/clickhouse/libclickhouse-cpp-lib.a"
if [ -f "$CH_LIB" ]; then
    echo "already built: $CH_LIB"
else
    cd "$DMI"
    cmake -S libs/clickhouse-cpp -B libs/clickhouse-cpp/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        || { echo "FAIL: clickhouse-cpp cmake configure"; exit 13; }
    cmake --build libs/clickhouse-cpp/build -j 8 \
        || { echo "FAIL: clickhouse-cpp build"; exit 13; }
    test -f "$CH_LIB" || { echo "FAIL: clickhouse-cpp lib not produced"; exit 13; }
fi

# ── 4. monitoring_native_backend.so ──────────────────────────────────
step "4/6 build monitoring_native_backend.so (SM_ARCH=sm_80, A100)"
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SO_FINAL="$DMI/monitoring_native_backend${EXT_SUFFIX}"
SO_LOCAL="$DMI/monitoring/monitoring_native_backend${EXT_SUFFIX}"
echo "SO target: $SO_FINAL"
if [ -f "$SO_FINAL" ] && [ -f "$SO_LOCAL" ]; then
    echo "already built"
    ls -la "$SO_FINAL" "$SO_LOCAL"
else
    cd "$DMI"
    # SM_ARCH must be explicit (sm_80 = A100); default 'native' fails on a
    # CPU-only standard node because nvcc can't query a GPU.
    make -C monitoring \
        NVCC="$CUDA_HOME/bin/nvcc" \
        SM_ARCH=sm_80 \
        -j 8 \
        || { echo "FAIL: monitoring make"; exit 14; }
    test -f "$SO_FINAL" || { echo "FAIL: $SO_FINAL not produced"; exit 14; }
fi

# ── 5. Symlink pip-vllm .so into integration/vllm/vllm/ ──────────────
step "5/6 symlink vllm site-packages .so files into forked tree"
VLLM_SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "VLLM_SITE=$VLLM_SITE"
VLLM_INTEG="$DMI/integration/vllm/vllm"
test -d "$VLLM_INTEG" || { echo "FAIL: $VLLM_INTEG missing (submodule not init?)"; exit 15; }

LINKED=0
for so in "$VLLM_SITE"/*.so; do
    [ -f "$so" ] || continue
    ln -sf "$so" "$VLLM_INTEG/$(basename "$so")"
    LINKED=$((LINKED+1))
    echo "  + $(basename "$so")"
done

# vllm_flash_attn subpkg
mkdir -p "$VLLM_INTEG/vllm_flash_attn"
for so in "$VLLM_SITE"/vllm_flash_attn/*.so; do
    [ -f "$so" ] || continue
    ln -sf "$so" "$VLLM_INTEG/vllm_flash_attn/$(basename "$so")"
    LINKED=$((LINKED+1))
    echo "  + vllm_flash_attn/$(basename "$so")"
done

# _version.py from pip vllm so the forked tree can resolve vllm.__version__
if [ -f "$VLLM_SITE/_version.py" ]; then
    ln -sf "$VLLM_SITE/_version.py" "$VLLM_INTEG/_version.py"
    echo "  + _version.py"
fi
echo "linked $LINKED .so files"
test "$LINKED" -ge 2 || { echo "FAIL: too few .so symlinks"; exit 15; }

# ── 6. pip install --no-deps -e DMI ──────────────────────────────────
step "6/6 pip install --no-deps -e DMI"
# --no-deps because TransformerLens dep would try to clone over the internet
# from a compute node and time out (we don't use the HF path, so it's OK to skip).
cd "$DMI"
pip install --no-deps -e . || { echo "FAIL: DMI editable install"; exit 16; }

# ── Smoke test: every import that bench_offline.py does, on the bench's PYTHONPATH ──
step "smoke-test imports (this is what bench-19190114 died on)"

cd "$WORK_ROOT"
export PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/integration/transformers/src${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$DMI/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python - <<'PY'
import sys
print(f"[smoke] python {sys.version.split()[0]}")
print(f"[smoke] sys.path[0:3] = {sys.path[0:3]}")
print()

print("[smoke] import transformers (forked) ...")
import transformers
print(f"  transformers {transformers.__version__}")
print(f"  loaded from  {transformers.__file__}")
assert "DMI/integration/transformers" in transformers.__file__, \
    "expected forked transformers, got upstream"

print("\n[smoke] import huggingface_hub ...")
import huggingface_hub
print(f"  huggingface_hub {huggingface_hub.__version__}")
# Pip installs as "1.0.0rc2" but the package reports __version__ as
# "1.0.0.rc2" (PEP 440 canonical form). Accept either spelling.
assert huggingface_hub.__version__ in ("1.0.0rc2", "1.0.0.rc2"), \
    f"expected 1.0.0rc2 (or canonical 1.0.0.rc2), got {huggingface_hub.__version__}"

print("\n[smoke] from vllm import SamplingParams  (the line that failed at bench-19190114:141) ...")
from vllm import SamplingParams
sp = SamplingParams(temperature=0.0, max_tokens=1)
print(f"  ok: {sp!r}")

print("\n[smoke] from vllm.config import ModelConfig ...")
from vllm.config import ModelConfig
print(f"  ok: {ModelConfig}")

print("\n[smoke] import vllm  (check we got the FORKED tree) ...")
import vllm, os
print(f"  vllm.__file__ = {vllm.__file__}")
assert "DMI/integration/vllm" in vllm.__file__, \
    f"expected forked vllm in DMI/integration/vllm, got {vllm.__file__}"

print("\n[smoke] from monitoring.config import HookSelection, MonitoringConfig ...")
from monitoring.config import HookSelection, MonitoringConfig
# HookSelection is a dataclass (not an Enum) — just verify imports succeeded.
# Mode validity gets checked at bench time when ring_transport.py looks
# up the requested mode in _HOOK_SELECTIONS.
print(f"  imported OK: HookSelection={HookSelection}, MonitoringConfig={MonitoringConfig}")

print("\n[smoke] import monitoring.vllm_integration ...")
import monitoring.vllm_integration as mvi
print(f"  DMXGPUWorker = {mvi.DMXGPUWorker}")

print("\n[smoke] ALL IMPORTS OK")
PY
RC=$?

echo
if [ "$RC" -eq 0 ]; then
    echo "================================================================"
    echo "  SMOKE TEST PASSED"
    echo "  Safe to resubmit:  sbatch bench.sbatch"
    echo "================================================================"
else
    echo "================================================================"
    echo "  SMOKE TEST FAILED (rc=$RC) — DO NOT resubmit bench yet"
    echo "================================================================"
fi
echo "=== done $(date) ==="
exit $RC
