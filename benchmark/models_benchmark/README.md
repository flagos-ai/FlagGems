# Benchmark README

This directory contains scripts and configurations for benchmarking
the throughput of **[FlagScale](https://github.com/FlagOpen/FlagScale)**  **with and without the FlagGems** operator library.

## 1. Requirements

Install Python dependencies with:
```bash
       pip install -r requirements.txt
```
## 2. Running Base Benchmark (FlagScale only)

Use the provided script `online.sh`.
1. Edit the script and change the `MODEL` variable:
```bash
       MODEL="/Change/To/Your/Real/Path/Here/Qwen/Qwen3-8B"
```
   Replace with the actual path to your model.

2. Run the script:
```bash
       bash online.sh
```
3. Results will be written under a timestamped folder, e.g.:
```bash
       online-benchmark-Qwen3-8B/2025_09_01_12_00/result.txt
```
Each run includes multiple configurations of:
- Input length: 128, 512, 1024, 2048, 6144, 14336, 30720
- Output length: 128, 512, 1024, 2048
- Number of prompts: 1, 100, 1000, 2000

The script automatically starts a vLLM server, waits until it is ready, and then launches the benchmark.

## 3. Running Benchmark with FlagGems

⚠️ Before running the FlagGems benchmark, you need to verify that your version of **FlagScale** includes the FlagGems hook.

### 3.1 Verify Integration in FlagScale

Check that your FlagScale source code includes logic similar to the following (see [flagscale/backends/vllm/vllm/worker/model_runner.py#L79](https://github.com/FlagOpen/FlagScale/blob/v0.8.0/flagscale/backends/vllm/vllm/worker/model_runner.py#L79)):

   ```python
  # --- FLAGSCALE MODIFICATION BEG ---
  # Know more about FlagGems: https://github.com/FlagOpen/FlagGems
  import os
  if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
      try:
          print("Try to using FLAGGEMS...")
          import flag_gems
          flag_gems.enable(record=True, path="/tmp/gems_oplist.log.txt")
          flag_gems.apply_gems_patches_to_vllm(verbose=True)
          logger.info("Successfully enabled flag_gems as default ops implementation.")
      except ImportError as e:
          # Throw an exception directly if failure occurs
          raise ImportError("Failed to import 'flag_gems'. Please install flag_gems or set USE_FLAGGEMS=false to disable it.") from e
      except Exception as e:
          # Throw an exception directly if failure occurs
          raise RuntimeError(f"Failed to enable 'flag_gems': {e}. Please check your flag_gems installation or set USE_FLAGGEMS=false to disable it.") from e
  # --- FLAGSCALE MODIFICATION END ---
   ```
If such code exists in your FlagScale installation, then FlagGems will be injected correctly when you set `USE_FLAGGEMS=1`.

When the service starts, check logs for messages like:


```
  Overriding a previously registered kernel for the same operator...
  operator: aten::add.Tensor(...)
```

This indicates that FlagGems has been enabled.

---

### 3.2 Run the Benchmark with FlagGems

Use the provided script `online_with_gems.sh`.

1. Edit the script and change the `MODEL` variable to your real model path.
2. Run the script:

   ```bash
   bash online_with_gems.sh
	```
 > Note: The script automatically exports `USE_FLAGGEMS=1`.
You do not need to set it manually.

3. Results will be saved in the same format as the base benchmark, under a folder ending with `-with-gems`.

## 4. Comparing Results

After running both scripts, you will have three sets of results:

- Base FlagScale results (from `online.sh`)
- FlagScale + FlagGems results (from `online_with_gems.sh`)
- check the `/tmp/gems_oplist.log.txt` file generated during the run. This log records which operators were invoked through `flag_gems`.
