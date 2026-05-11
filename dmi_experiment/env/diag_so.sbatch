#!/bin/bash
# diag_so.sbatch — diagnose why monitoring_native_backend.so fails to dlopen
# on the GPU compute node. The DMI loader at _native_engine.py:54-56 catches
# all exceptions silently and reports a misleading "not found" message;
# this script bypasses the loader and exposes the real error.
#
# Submit: sbatch diag_so.sbatch
# Tail:   tail -f diag-so-<JOBID>.out  diag-so-<JOBID>.err
#
# Short walltime to favor scheduling priority. No model loading, no vLLM,
# just import the .so and look at it.

#SBATCH --job-name=dmi-diag-so
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=diag-so-%j.out
#SBATCH --error=diag-so-%j.err

set -uo pipefail

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-local}  start $(date) ==="

# Toolchain — match what bench.sbatch uses
module load cuda/12.3.0/gcc cmake/3.26.3/gcc gcc/11.3.0
source ~/scratch.zaoxing-prj/miniconda3/etc/profile.d/conda.sh
conda activate vllm-exp

DMI=~/scratch.zaoxing-prj/DMI
SO_LOCAL="$DMI/monitoring/monitoring_native_backend.cpython-310-x86_64-linux-gnu.so"
SO_REPO="$DMI/monitoring_native_backend.cpython-310-x86_64-linux-gnu.so"

# Match bench's env exactly
export PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/integration/transformers/src${PYTHONPATH:+:$PYTHONPATH}

# Torch ships its own CUDA 12.8 libs (cusparse calls __nvJitLinkCreate_12_8);
# the cuda/12.3.0 module's lib64 has a 12.3 nvJitLink missing that symbol.
# Force torch's bundled libs to win.
TORCH_NV_LIBS=$(python -c "
import os, sys
nv = os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia')
if os.path.isdir(nv):
    print(':'.join(os.path.join(nv, d, 'lib') for d in sorted(os.listdir(nv))
                   if os.path.isdir(os.path.join(nv, d, 'lib'))))
")
echo "TORCH_NV_LIBS=$TORCH_NV_LIBS"
# $CONDA_PREFIX/lib must come BEFORE the gcc/11.3.0 module's lib64, otherwise
# /cvmfs/.../gcc-11.3.0/lib64/libstdc++.so.6 wins — and that lib (gcc 11.3) is
# missing GLIBCXX_3.4.30, which torch's libs (built with gcc 12+) require.
# The conda env has libstdc++ from libstdcxx-ng (gcc 13+) which has the symbol.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$TORCH_NV_LIBS:$DMI/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo
echo "=== state ==="
which python; python -V
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -la "$SO_LOCAL" "$SO_REPO" 2>&1
nvidia-smi | head -10

echo
echo "=== ldd $SO_LOCAL ==="
ldd "$SO_LOCAL" 2>&1
RC=$?
echo "(ldd rc=$RC; missing libs show as 'not found')"

echo
echo "=== ldd $SO_REPO ==="
ldd "$SO_REPO" 2>&1

echo
echo "=== explicit dlopen via Python with full traceback ==="
python <<'PY'
import importlib.util, glob, traceback, sys
from pathlib import Path

EXT = "monitoring_native_backend"
DMI = Path("/home/jahmad1/scratch.zaoxing-prj/DMI")

candidates = []
candidates.extend(glob.glob(str(DMI / "monitoring" / f"{EXT}*.so")))
candidates.extend(glob.glob(str(DMI / f"{EXT}*.so")))
print(f"candidates: {len(candidates)} found")
for c in candidates:
    print(f"  {c}")

print()
for so_path in candidates:
    print(f"--- attempting to load {so_path} ---")
    try:
        spec = importlib.util.spec_from_file_location(EXT, so_path)
        if spec is None or spec.loader is None:
            print("  no spec or loader")
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  OK — module loaded, attrs sample: {[a for a in dir(mod) if not a.startswith('_')][:10]}")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stdout)
    print()
PY

echo
echo "=== done $(date) ==="
