# DMI side of the DMI vs vLLM-extract_hidden_states comparison

Class project (CMSC818Q, Spring 2026). DMI runs in `vllm-full` mode with
`dmx_null_mode=True` against vanilla vLLM 0.17, on Zaratan A100, with
Qwen3-4B and `max_tokens=1`. The other side of the comparison
(vLLM-extract / PR #33736) lives in `../vllm_ehs/`.

## Layout

```
dmi_experiment/
├── env/         scripts to (re)build the vllm-exp conda env on Zaratan
├── bench/       drivers + sbatch wrappers used to produce the results
└── results/     the CSVs and stdout/stderr from the actual runs
```

### env/

| file | what it does |
|---|---|
| `setup_env.sh` | DMI's upstream recipe with one patch: `vllm==0.17.0` → `vllm==0.17.0+cu123`. Modern pip refuses the unsuffixed pin because the published wheel has a local version identifier. Original lives at `experiments/online_serving/script/setup_env.sh`. |
| `rebuild_vllm.sbatch` | source-builds vLLM 0.17 because `VLLM_USE_PRECOMPILED=1` ships a `_C.abi3.so` linked against glibc 2.34 and Zaratan compute nodes have glibc 2.28. Source build links against the build node's glibc, which matches. |
| `fix_env.sbatch` | finishes the env after `rebuild_vllm.sbatch`: pins huggingface-hub, installs flashinfer, builds clickhouse-cpp, builds the monitoring native backend, symlinks vLLM's `.so`s into the forked tree, `pip install -e` DMI. Has a smoke-test at the end. |
| `diagnose_env.sh` | login-node read-only health check. Run this if anything stops working — it tells you which artifact is missing. |
| `diag_so.sbatch` | GPU-node diagnostic for `monitoring_native_backend.so` dlopen failures. Only useful if `fix_env.sbatch`'s smoke test fails. |
| `snapshots/conda_env_nobuilds.yml` | conda env yaml from the working state on 2026-05-06. **Reference only** — `conda env create -f` fails on editable installs. Useful for auditing what versions actually ran. |
| `snapshots/pip_freeze.txt` | `pip freeze` from the same state. Useful for diffing against a fresh build to see what drifted. |

### bench/

| file | what it does |
|---|---|
| `bench_offline.py` | driver. `--mode vanilla` runs vanilla vLLM; `--mode dmi --null-mode --hook-selection vllm-full` runs DMI with full capture and no ClickHouse. Appends one row to a CSV per invocation. |
| `bench.sbatch` | phase 1 sweep wrapper. Sweeps `max_num_seqs ∈ {1, 4, 8, 16, 32}` and runs vanilla + DMI for each. |
| `bench_offline_replicate.py` | same as `bench_offline.py` but loads 256 ShareGPT prompts (seed=42, deterministic) instead of 64 synthetic prompts. |
| `bench_replicate.sbatch` | phase 2 sweep wrapper. Same batch sweep, wrapped in an outer 4-trial loop for mean ± stddev. |

### results/

Phase 1 and phase 2 are two separate runs with different prompt sets:

| | phase 1 | phase 2 |
|---|---|---|
| prompts | 64 synthetic prompts (one prompt template) | 256 ShareGPT human turns (seed=42) |
| `max_model_len` | 4096 | 8192 |
| trials per cell | 1 | 4 (mean ± stddev) |
| job ID | 19203724 | 19219545 |
| date | 2026-05-04 | 2026-05-06 |

Both are batch-size sweeps `{1, 4, 8, 16, 32}` with vanilla vLLM and
DMI null-mode, on Qwen3-4B, A100, `max_tokens=1`.

## How to reproduce on Zaratan

1. Clone DMI on Zaratan into `~/scratch.zaoxing-prj/DMI` and check out
   `Cloud_final_project`.
2. Pre-clone vLLM 0.17 source into `~/scratch.zaoxing-prj/vllm-0.17.0` with
   the cmake patch + pre-fetched `.deps/*-src` dirs (rebuild_vllm.sbatch
   aborts loudly if either is missing — read the error message).
3. Create an empty Python 3.10 conda env. The sbatch scripts in step 4
   and 5 do all the actual installs, so this env starts blank:
   ```
   conda create -n vllm-exp python=3.10 -y
   ```
   Don't try `conda env create -f env/snapshots/conda_env_nobuilds.yml` —
   the yaml has editable-install entries (`hf-prometheus`, dev-version
   `transformers`) that aren't on PyPI and the create fails. The yaml is
   shipped for comparison, not reproduction (see snapshots note below).
4. `sbatch env/rebuild_vllm.sbatch` — replaces the precompiled vllm
   with a source build linked against the build node's glibc 2.28.
5. `sbatch env/fix_env.sbatch` — re-pins huggingface-hub, builds
   clickhouse-cpp + the monitoring native backend, symlinks vLLM's
   compiled `.so`s into the forked tree, `pip install -e` DMI, and
   smoke-tests imports at the end.
6. Bench scripts expect `bench_offline*.py` at `$WORK_ROOT`
   (`~/scratch.zaoxing-prj/`). Copy or symlink them there before
   submitting `bench.sbatch` / `bench_replicate.sbatch`.

## If something breaks

- Run `bash env/diagnose_env.sh` on the login node first. It reports
  PRESENT/MISSING for every load-bearing artifact (hub pin,
  clickhouse-cpp lib, monitoring `.so`, vllm symlinks) in ~10 seconds.
  Tells you what's broken before you go GPU-shopping.
- If the bench dies with a confusing `.so` / dlopen error, then run
  `sbatch env/diag_so.sbatch` on a GPU node. It bypasses DMI's silent
  exception-swallowing loader and prints the actual ABI/symbol error.

## Caveats worth knowing before touching anything

- **huggingface-hub MUST be `1.0.0rc2` exactly.** DMI's forked
  transformers does an `==` version check at import time. If hub gets
  bumped (transitive installs sometimes do this), every bench will die
  at first import. Re-pin with `pip install "huggingface-hub==1.0.0rc2"`
  after any pip activity. `diagnose_env.sh` flags it.
- **pip metadata `transformers` must stay `<5.0.0`.** The forked vLLM in
  `integration/vllm/vllm/transformers_utils/config.py` reads
  `importlib.metadata.version("transformers")` to decide v4 vs v5 code
  path. If something pollutes pip metadata to 5.x (e.g. a stray
  `transformers` install), forked vLLM enters the v5 branch and calls
  methods like `standardize_rope_params` that the forked source doesn't
  define. `bench_offline_replicate.py` includes a defensive monkey-patch
  for this at the top of the file. The cleaner fix is to realign with
  `pip uninstall -y transformers && pip install -e DMI/integration/transformers && pip install "huggingface-hub==1.0.0rc2"`.
- **`~/.local/lib/python3.10/site-packages/` can silently shadow the
  conda env.** If the bench picks up a stale package from there it
  ignores whatever's in conda. Worth checking with
  `python -c "import sys; print([p for p in sys.path if '.local' in p])"`
  if something is mysteriously wrong.
- **`bench_offline_replicate.py` has a monkey-patch for `PretrainedConfig`**
  at the top of the file (lines 23-27). It's a safety net for the
  transformers-v5 metadata problem above. Don't strip it unless you've
  confirmed pip metadata is clean.
- **`setup_env.sh`'s path math is broken from this location.** Inside
  the script, `REPO_ROOT="$SCRIPT_DIR/../../.."` was correct from the
  original location (`experiments/online_serving/script/`) but resolves
  to `cmsc818q_project/` from here. The file is shipped as a
  documentation snapshot — for actually building the env, use
  `rebuild_vllm.sbatch` + `fix_env.sbatch`.
- **`bench.sbatch` / `bench_replicate.sbatch` reference `$WORK_ROOT/bench_offline.py`.**
  `$WORK_ROOT` is `~/scratch.zaoxing-prj/`. Copy or symlink the bench
  drivers there before submitting (or edit the path; we left the
  scripts unmodified to keep them faithful to what was actually run).
