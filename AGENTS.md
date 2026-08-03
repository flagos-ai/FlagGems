# AGENTS.md

This file is the shared guidance for AI coding agents (Claude Code, Gemini CLI, GitHub
Copilot, Cursor, OpenAI Codex, etc.) working in this repository. `CLAUDE.md`, `GEMINI.md`,
and `.github/copilot-instructions.md` are symlinks to this file.

## Project overview

FlagGems is a high-performance, backend-neutral operator library written in the
[Triton](https://github.com/openai/triton) language. It registers Triton kernels against
PyTorch's ATen dispatcher so that unmodified `torch.*` / `torch.nn.functional.*` calls are
transparently redirected to FlagGems implementations (no `torch.compile` required). A single
Python codebase targets 10+ hardware backends (NVIDIA, AMD, Cambricon, Ascend, Iluvatar,
Metax, Moore Threads, Kunlunxin, and more) through a vendor-dispatch runtime.

## Build & install

Development is an **editable install**; there is no separate "build" step for the pure-Python
path (kernels JIT-compile at runtime).

```shell
pip install -e .                      # pure-Python editable install (most common)
pip install -r requirements/requirements_nvidia.txt   # backend deps (pick your backend file)
```

- `setup.sh <backend>` bootstraps a full venv via `uv` for a named backend (e.g.
  `./setup.sh nvidia-cuda128`). Backend names and their pinned deps live in
  `src/flag_gems/backends.yaml` and `pyproject.toml` `[project.optional-dependencies]`.
- **C++ extensions** (experimental, off by default) are built through scikit-build-core +
  CMake by passing `CMAKE_ARGS`:
  ```shell
  CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DCMAKE_BUILD_TYPE=Release" pip install -v -e .
  ```
  Select a backend with `-DFLAGGEMS_BACKEND=CUDA|IX|MUSA|NPU`. The C++ path depends on the
  external [libtriton_jit](https://github.com/flagos-ai/libtriton_jit) library. Always set
  `-DCMAKE_BUILD_TYPE=Release` or the wrappers run un-optimized. See
  `docs/content/en/getting-started/install.md` for the full CMake option table.

## Lint & format

Formatting/pre-checks run through `pre-commit` (black, isort with the black profile, flake8,
clang-format for C/C++). Install once, then it runs on every `git commit`:

```shell
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

flake8 ignores `F405,E731,W503,E203` with `--max-line-length=120`.

## Tests & benchmarks

The two test kinds are split by directory, and an operator is only considered good when it
passes **both**:

- `tests/` — **accuracy only** (one `test_<op>.py` per op; compares against a PyTorch reference).
- `benchmark/` — **performance only** (mirrored `test_<op>.py`; measures speedup vs. PyTorch).

Both are driven by `pytest`; `pytest.ini` sets `pythonpath = src`, so `flag_gems` imports without
installing. Every test/benchmark function is decorated `@pytest.mark.<op_id>` (the operator's
inventory ID from `conf/operators.yaml`), which is what enables marker-based selection.

### Direct pytest (one op, single device)
A plain `pytest` invocation runs on the single active device — effectively **GPU 0** unless you
set the vendor's `*_VISIBLE_DEVICES` env var yourself.

```shell
pytest tests/test_add.py                 # accuracy for one op on the active device
pytest tests/test_add.py --ref cpu       # compare against a CPU reference instead of the device
pytest tests/test_add.py --quick         # reduced shape/dtype coverage (fast smoke run)
pytest -m add tests/                     # select by operator ID marker
pytest benchmark/test_add.py -s --level core   # perf benchmark (levels: core | comprehensive)
```

Benchmark `--mode` selects what latency is measured: `kernel` (default), `operator` (e2e),
`wrapper`, or `cudagraph`.

### Unified runner: `tools/run_tests.py` (accuracy + benchmark, multi-GPU)
This is the simplest way to confirm an op passes **both** accuracy and benchmark in one shot.
For each operator it runs the accuracy test (from `tests/`) **and then** the benchmark (from
`benchmark/`), and writes a combined, detailed report to `results/summary.json` (plus per-op
`results/<op>/accuracy_result.json` and `performance_result.json`; add `--dump-output` for raw
per-op stdout/stderr logs). It selects operators from the `conf/operators.yaml` inventory and
pins each op to a GPU via the vendor's `*_VISIBLE_DEVICES`, so — unlike direct `pytest`, which is
limited to GPU 0 — it spreads work across multiple devices in parallel (one worker process per GPU).

```shell
python tools/run_tests.py --ops add                 # one op: accuracy + benchmark
python tools/run_tests.py --ops add,mul,softmax      # several ops (comma-separated IDs)
python tools/run_tests.py --op-list-file ops.txt     # ops from a file (one ID per line, # comments)
python tools/run_tests.py --stages stable            # all ops at a maturity stage (default: stable)
python tools/run_tests.py --ops add --gpus 3         # run on a specific GPU id
python tools/run_tests.py --stages all --gpus all    # every op, spread across all detected GPUs
```

- **Which ops:** `--ops` (comma-separated IDs) > `--op-list-file` > `--stages`
  (`alpha,beta,stable,all`, default `stable`); `--start <id>` skips ops before an ID.
- **Where:** `--gpus` accepts `0` (default), a comma-separated list like `0,1,3`, or `all`
  (every detected device).
- **Reporting:** `--output-dir` (default `results`), `--dump-output`, `--color`.
- An op missing an accuracy or benchmark test is reported `NotFound` for that phase.

CI per-PR vs. full-suite selection is orchestrated separately by `tools/test-op.sh`.

## Architecture (the parts that span multiple files)

### Registration & dispatch flow
1. `src/flag_gems/__init__.py` builds `_FULL_CONFIG`: a big tuple mapping ATen overload names
   (e.g. `"mm"`, `"add.Tensor"`, `"softmax_out"`) to their Python implementation functions.
2. `flag_gems.enable()` / `only_enable()` / `with use_gems():` install those mappings into a
   `torch.library.Library("aten", "IMPL")` via `GeneralOpRegistrar`, so PyTorch dispatches the
   real `torch.*` calls into FlagGems. `enable(unused=...)` excludes ops; `only_enable(include=...)`
   registers a whitelist; both can take a list, a YAML path, or `"default"` (auto-loads the
   vendor/arch `enable_configs.yaml`). `use_gems` is the scoped context-manager form.
3. Operators can also be called directly, bypassing dispatch: `from flag_gems import ops; ops.mm(...)`
   or `from flag_gems.fused.<name> import <fn>`.

### Multi-backend runtime (`src/flag_gems/runtime/`)
- `runtime/backend/_<vendor>/` holds each vendor's overrides: `__init__.py` (a `VendorDescriptor`
  with `vendor_name` / `device_name` / `device_query_cmd`), `tune_configs.yaml`
  (`triton.autotune` configs), `heuristics_config_utils.py` (`triton.heuristics`), and an `ops/`
  folder for **vendor-specialized kernel implementations** that override the generic ones.
- `device_finder.py` detects the active vendor at import time; `runtime.device` exposes the
  resolved device name/vendor. `SpecOpRegistrar` (applied in `__init__.py`) swaps in the
  current backend's specialized ops over the generic implementations.
- To add a new backend, follow `src/flag_gems/runtime/backend/README.md` (copy `_nvidia` and
  edit the `VendorDescriptor`).

### Operator source layout
- `ops/` — single (non-fused) operators, one file per op. `fused/` — fused operators (rms_norm,
  moe, attention, etc.). `modules/` — higher-level `nn`-style building blocks. `experimental_ops/`
  — newer/less-stable ops. Each package's `__init__.py` re-exports its public symbols, which
  `flag_gems/__init__.py` star-imports to assemble `_FULL_CONFIG`.

### Pointwise codegen (`utils/pointwise_dynamic.py`)
The signature abstraction. The `@pointwise_dynamic` decorator wraps a `@triton.jit` scalar-style
function and **generates** the actual Triton kernel + wrapper, handling broadcasting, arbitrary
ranks/strides, non-contiguous/overlapping storage, type promotion, and multiple outputs. A
minimal op is just the payload plus metadata:

```python
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)
```

Key decorator args: `is_tensor=[...]` (which args are tensors), `dtypes=[...]` (non-tensor
types), `promotion_methods=[(idx, RULE)]` (output dtype rules: `DEFAULT`, `NO_OPMATH`,
`INT_TO_FLOAT`, `ALWAYS_BOOL`, `COMPLEX_TO_FLOAT`, `BOOL_TO_LONG`), and `num_outputs`. Output
tensors are passed as keyword args named `out0`, `out1`, … which is how in-place / `out=` variants
are implemented. Full details in `docs/content/en/overview/pointwise-dynamic.md`.

### Autotune wrappers (`utils/libentry.py`)
`@libentry()` and `libtuner` wrap Triton's `Autotuner`/`KernelInterface` to add FlagGems-specific
kernel caching and per-function dispatch. Non-pointwise kernels typically use these.

## Adding or changing an operator

1. Implement the kernel in the right package (`ops/`, `fused/`, …) and export it from that
   package's `__init__.py`. Prefer `@pointwise_dynamic` for elementwise ops.
2. Register the ATen overload → function mapping in `src/flag_gems/__init__.py` (`_FULL_CONFIG`).
   In-place (`_` suffix) and `out=` variants are separate ATen overloads and get separate entries.
3. Add an entry to `conf/operators.yaml` (the operator inventory, since v4.2). Every registered
   aten/fused op needs a unique `id`, plus `description` / `for` / `labels` / `kind` / `stages`
   (maturity: `alpha` → `beta` → `stable`; AI-generated `KernelGen` ops start at `alpha`,
   hand-written at `beta`).
4. Add accuracy tests in `tests/test_<op>.py` and (for new/optimized ops) a benchmark in
   `benchmark/test_<op>.py`, each decorated `@pytest.mark.<op_id>`. Verify both pass together with
   `python tools/run_tests.py --ops <op_id>` — an op is only complete when accuracy **and**
   benchmark pass.
5. Vendor-specific tuning/specialization goes under `runtime/backend/_<vendor>/`, not in the
   generic op.
6. C++-wrapped operators additionally need a `ctests/` entry — see
   `docs/content/en/contribution/cpp-wrapper.md`.

## Conventions

- Ops emit a `logger.debug("GEMS <NAME>")` line at entry (see any file in `ops/`); keep this
  pattern for new ops.
- `flag_gems.device` / `flag_gems.vendor_name` are the portable way to reference the active
  device in tests and examples — avoid hard-coding `"cuda"`.
- Vendor package names in `backends.yaml` / `pyproject.toml` pin exact torch/triton/flagtree
  versions per backend; don't loosen these casually.
- Documentation source is under `docs/content/{en,zh-cn}/` (Hugo); the English tree is the
  reference for architecture and workflow details referenced above.
