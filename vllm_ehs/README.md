# EHS side of the DMI vs vLLM-extract_hidden_states comparison

We benchmark EHS using vLLM 0.18.1 running in Apptainer with 
[this image](https://hub.docker.com/layers/vllm/vllm-openai/v0.18.1/images/sha256-6aaea8694609df39960a76961807c5956879797c503de9573815a984d0b26407).

### bench/

| File | What it Does |
|---|---|
| `share_gpt_bench.sh` | Batch script to evaluate the EHS vLLM feature. Runs 4 sweeps across batch sizes 1, 4, 8, 16, and 32, using 256 total prompts for each trial. |
| `share_gpt_bench_no_write.sh` | Batch script to evaluate EHS without writing any data to disk. Same sweeps as above. |
| `share_gpt_bench_no_extract.sh` | Batch script to benchmark vLLM 0.18.1 with no EHS. Gives us a baseline that we can compare with EHS. Same sweeps as above. |
| `share_gpt_bench.py` | Driver python script that times vLLM with EHS enabled. Called by `share_gpt_bench.sh` and `share_gpt_bench_no_write.sh`. Writes data to a temporary file unless flag `--no-write` is enabled. Appends one row to given CSV. |
| `share_gpt_bench_no_extract.py`| Driver python script that times vLLM without EHS. Called by `share_gpt_bench_no_extract.sh`. Appends one row to given CSV. |
| `NoWritesConnector.py` | Custom `KVConnector` class used by `share_gpt_bench.py` when `--no-write` is enabled to use EHS without writing states to the disk. | 
