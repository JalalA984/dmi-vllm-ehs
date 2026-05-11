#!/bin/bash
# diagnose_env.sh — run on the LOGIN NODE.  Reports what's present in the
# vllm-exp env / DMI repo right now.  Read-only; takes ~10 seconds.
#
# Usage:  bash ~/scratch.zaoxing-prj/diagnose_env.sh

set -uo pipefail

echo "=== node $(hostname)  $(date) ==="
echo

source ~/scratch.zaoxing-prj/miniconda3/etc/profile.d/conda.sh
conda activate vllm-exp

DMI=~/scratch.zaoxing-prj/DMI

echo "--- python / env ---"
which python && python -V
echo

echo "--- pip pkgs of interest ---"
for pkg in vllm transformers huggingface-hub flashinfer-python flashinfer torch; do
    line=$(pip show "$pkg" 2>/dev/null | grep -E "^Version" || echo "  NOT INSTALLED")
    printf "  %-22s %s\n" "$pkg" "$line"
done
echo

echo "--- DMI native artifacts ---"
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null || echo ".so")
echo "  EXT_SUFFIX=$EXT_SUFFIX"
for f in \
    "$DMI/libs/clickhouse-cpp/build/clickhouse/libclickhouse-cpp-lib.a" \
    "$DMI/monitoring_native_backend${EXT_SUFFIX}" \
    "$DMI/monitoring/monitoring_native_backend${EXT_SUFFIX}" \
    ; do
    if [ -f "$f" ]; then
        printf "  PRESENT  %s  (%s)\n" "$f" "$(ls -la "$f" | awk '{print $5}') bytes"
    else
        printf "  MISSING  %s\n" "$f"
    fi
done
echo

echo "--- vllm site-packages symlinks into integration/vllm/vllm/ ---"
VLLM_INTEG="$DMI/integration/vllm/vllm"
if [ ! -d "$VLLM_INTEG" ]; then
    echo "  ERROR: $VLLM_INTEG does not exist (submodule not initialized?)"
else
    ls -la "$VLLM_INTEG"/_C*.so "$VLLM_INTEG"/_moe_C*.so "$VLLM_INTEG"/_version.py 2>/dev/null || echo "  no _C / _moe_C / _version.py symlinks present"
    if [ -d "$VLLM_INTEG/vllm_flash_attn" ]; then
        ls -la "$VLLM_INTEG/vllm_flash_attn"/*.so 2>/dev/null || echo "  vllm_flash_attn/ exists but no .so symlinks"
    else
        echo "  vllm_flash_attn/ dir missing"
    fi
fi
echo

echo "--- where vllm is actually being imported from (PYTHONPATH unset) ---"
python -c "import vllm, os; print('  ', os.path.dirname(vllm.__file__))"
echo

echo "--- the actual failing import from bench-19190114 ---"
echo "Trying: from vllm import SamplingParams  (with bench's PYTHONPATH)"
PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/integration/transformers/src \
python - <<'PY' 2>&1 | sed 's/^/  /'
try:
    from vllm import SamplingParams
    print("OK: SamplingParams imported")
except Exception as e:
    print(f"FAIL ({type(e).__name__}): {e}")
PY
echo

echo "=== done ==="
