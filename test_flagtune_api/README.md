# FlagTune API Checks

Run from the repository root:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm0.17
./test_flagtune_api/run_checks.sh
```

The default check imports the local `src/flag_gems`, validates
`flag_gems.flagtune(include=...)`, checks the FlagTune op registry can be
extended, confirms unregistered ops are rejected, and checks that the selected
LibTuner objects switch to expanded configs without launching kernels.
