# EHS side of the DMI vs vLLM-extract_hidden_states comparison

We benchmark EHS using vLLM 0.18.1 running in Apptainer with 
[this image](https://hub.docker.com/layers/vllm/vllm-openai/v0.18.1/images/sha256-6aaea8694609df39960a76961807c5956879797c503de9573815a984d0b26407).

### bench/

| File | What it Does |
|---|---|
| `share_gpt_bench.sh` | Batch script to evaluate the EHS vLLM feature. Runs 4 sweeps across batch sizes 1, 4, 8, 16, and 32, using 256 total prompts for each trial. |
| `share_gpt_bench_no_write.sh` | Batch script to evaluate EHS without writing any data to disk. Same sweeps as above. |
| `share_gpt_bench_no_extract.sh` | Batch script to benchmark vLLM 0.18.1 with no EHS. Gives us a baseline that we can compare with EHS. Same sweeps as above. |
| `share_gpt_bench.py` | Driver python script that times vLLM with EHS enabled. Called by `share_gpt_bench.sh` and `share_gpt_bench_no_write.sh`. Writes data to a temporary file unless flag `--no-write` is enabled. Takes CSV as input, appends one row. |
| `share_gpt_bench_no_extract.py`| Driver python script that times vLLM without EHS. Called by `share_gpt_bench_no_extract.sh`. Takes CSV as input, appends one row. |
| `NoWritesConnector.py` | Custom `KVConnector` class used by `share_gpt_bench.py` when `--no-write` is enabled. Lets us use EHS without writing states to the disk. | 

### results/

The results of our benchmarking are in the `results` directory. Results for EHS with writing to disk are in `results-share-19217997.csv`, EHS without writing to disk in `results-share-no-write-19222206.csv`, and for vLLM 0.18.1 without EHS in `results-share-no-extract-19217999.csv`. 

## How to Reproduce on Zaratan
This assumes access to the `/scratch/zt1/project/zaoxing-prj/shared` directory, where we have placed the model weights for Qwen3-4B and the ShareGPT dataset. The below steps give instructions with specific file locations, you can use different locations but will need to make minor changes to the batch scripts.
1. Install the Apptainer image linked above. Place it in `~/scratch`.
2. Place the python driver that you will use file in the home directory.
3. Submit the batch script that you will use with `sbatch`.
4. CSV files with results are configured to appear in `/scratch/zt1/project/zaoxing-prj/shared/CMSC818Q_proj/`.
