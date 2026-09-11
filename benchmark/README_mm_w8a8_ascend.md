# Ascend W8A8 MM (draft)

The Ascend backend implements `mm_w8a8_fp8`/`mm_w8a8_fp8_out` with INT8 inputs
and per-row/per-column FP32 scales. This backend uses INT8 Cube instructions;
it does not imply native FP8 matrix multiplication on Ascend 910B.

## Implementation

- Shape-dependent tiling, with NZ-prepacked inputs for wide N and selected narrow N cases.
- Exact INT32 accumulation plus a fused FP32 scale/cast Vector pass where appropriate.
- Guarded Cube-only `al.custom` paths: Fixpipe VDEQF16 into L1, row scaling on Cube,
  and Fixpipe BF16 output. Large short-K cases use 16-row scale blocks, batched
  Fixpipe output, and next-tile L1 prefetch.
- A device range check selects a software FP32 scaling fallback when the FP16
  intermediate is unsafe. That fallback is for correctness, not a performance guarantee.

## Local validation

The CI-preparation snapshot passed `pre-commit run --all-files`, all rule-check
scripts, and the repository `tools/test-op.sh` on CANN 9.0 with
Torch 2.10.0+cpu, torch-npu 2.10.0, and FlagTree 0.6.1+ascend3.5:
225 tests passed in normal mode and 225 passed with `--ref=cpu --quick`.
The built wheel contains the Python/C++ fragments; five installed-wheel smoke
checks passed. This was local reproduction, not a completed upstream CI run.
The final commit only additionally normalizes a header's line endings and adds
these documentation/reference files; final-commit validation is recorded in the PR.

### Known blockers before ready for review

1. The CANN 8.5 CI compiler (FlagTree 0.6.0+ascend3.2) lacks intermediate pipeline
   APIs required by the compatibility layer. Its backend tests have not passed.
2. Re-run all 433 performance shapes on the submitted CI-compatible source and
   pinned compiler. The reference measurements below belong to an earlier source.
3. Review the compiler compatibility hooks and mixed/Cube entry handling.

## Historical performance evidence (not this commit)

`reference_results/mm_w8a8_ascend_pr5972.csv` contains all 433 shapes from
FlagGems PR #5972, head `9fe20332b9407eaa88b14c613b316551e49d9ba2`.
On Ascend910B4-1, the measured operator source SHA256 was
`e376d807218f0f3b02286afaca7e4216218852d16e56216c46d7d387a413a0e1`.
The environment used Torch 2.10.0+cpu, torch-npu 2.10.0.post2,
FlagTree 0.6.0+ascend.gitf56cd1bd and CANN 9.0, with additional CANN 9.1 compiler
paths available. These are different from the CI-preparation environment above.

All 433 shapes passed their respective numerical checks and were faster by
six-sample median than Torch and default Ascend FlagGems BF16 MM. Geometric mean
speedups were 1.3307x versus Torch and 3.7976x versus default FlagGems.
Those gains have not yet been revalidated on the submitted source.

Timing used NPUGraph replay. MM, both output scales, and the final cast were
included; input quantization, padding, input/scale packing, compilation,
autotuning, and explicit allocations were excluded. Torch and FlagGems used the
faster of equivalent row/column-major B layouts. This is not end-to-end latency.
The quantized operator and BF16 baselines have different input precision semantics.

## Reproduce on a configured Ascend environment

```bash
# Source your matching CANN environment and activate the configured Python venv.
export FLAGTREE_BACKEND=ascend
export ASCEND_RT_VISIBLE_DEVICES=2  # choose an available device on your machine
export PYTHONPATH="$PWD/src"
python -m pytest -q tests/test_mm_w8a8_fp8.py
CHANGED_FILES=tests/test_mm_w8a8_fp8.py bash tools/test-op.sh local-mm

# Prepared-input kernel benchmark; the built-in list is a smaller smoke suite.
DTYPE=bf16 GEMS=1 python benchmark/bench_mm_w8a8_ascend.py mm-result.json
```

`SHAPES` can override the benchmark list with comma-separated `MxNxK` values.
The full shape list is in `mm_w8a8_ascend_shapes.json`. The standalone smoke
benchmark is not an upstream CI benchmark test and is not identical to the
historical shared-buffer three-way comparison harness.
