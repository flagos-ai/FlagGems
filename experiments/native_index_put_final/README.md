# Opt-in Native MUSA `index_put` experiment

This experiment adds MUSA C++/MUSA kernels for selected `index_put` and
`index_put_` workloads on MTT S5000. It is explicitly experimental. The public
path remains disabled unless `FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT=1` is set.

## Build

Build the extension from source with the active Torch MUSA installation:

```bash
bash experiments/native_index_put_final/build.sh
```

The generated shared object and build directory are local artifacts and are
not part of the source change. The build script imports `torch_musa` before
invoking its MUSA extension toolchain.

## Official correctness and benchmark

The helper scripts invoke the repository's unchanged official files:

```bash
bash experiments/native_index_put_final/run_correctness.sh both
bash experiments/native_index_put_final/run_benchmark.sh both
```

The benchmark helper runs `benchmark/test_index_put.py` in Comprehensive
Operator mode, serially, with the repository benchmark implementation. It
writes the original JSON files and compact audit CSVs under `results/`.
For `both`, it also runs the official workload once with a dispatch trace;
that traced run confirms which Cases entered Native code, but its timings are
not used as performance evidence.

For manual commands, expose both the extension and source tree before running
pytest, and import `torch_musa` so the MUSA device is registered:

```bash
export PYTHONPATH="experiments/native_ext:src${PYTHONPATH:+:${PYTHONPATH}}"
python -c 'import torch_musa, pytest; raise SystemExit(pytest.main(["-q", "tests/test_index_put.py"]))'
FLAGGEMS_MTHREADS_NATIVE_INDEX_PUT=1 python -c 'import torch_musa, pytest; raise SystemExit(pytest.main(["-q", "tests/test_index_put.py"]))'
```

## Scope and known limitations

The implementation uses row-tail, full-rank, atomic, and Bool-prefix paths
behind conservative shape, dtype, layout, and alias gates. Unsupported inputs
continue through the existing FlagGems implementation. The extension is loaded
once at module import and the native path is opt-in.

Native kernels mask invalid addresses so an out-of-range index does not issue
an invalid memory access. They do not currently propagate a recoverable
PyTorch-style out-of-range exception. CPU Torch reports `IndexError` for the
tested invalid indices, while the installed MUSA Torch path returned without
an error in the same probe. Until this contract is resolved, keep Native
disabled by default; this experiment is not production-ready.

The current official Comprehensive results remain below the 0.80x Torch goal.
See `results/performance_summary.md` and the paired raw JSON/CSV outputs for
the exact commit and environment used for each run. The existing PR #7163
contains a separate Triton N=1 change; this Native implementation is based on
upstream master and does not include that patch.
