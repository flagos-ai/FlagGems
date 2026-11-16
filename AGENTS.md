# Project: FlagGems

FlagGems is a high-performance, generic [Triton](https://github.com/openai/triton) operator library that accelerates LLM training and inference across diverse AI hardware by registering into PyTorch's ATen backend.

## 1. Core Principles

- **Triton-First & Multi-Backend**: Implement core kernels in Triton for portability across NVIDIA, Ascend, Hygon, Moore Threads, MetaX, Iluvatar, Kunlunxin, Cambricon, etc.
- **Seamless Integration**: Register operators into PyTorch's ATen dispatcher for zero-code-change acceleration.
- **Host Function Isolation**: Host functions MUST NOT invoke PyTorch compute ops (only allocation/init like `empty_like`, `zeros`, `randn`, or other FlagGems ops are allowed) and must launch device kernels.
- **Dual Validation**: Every operator change requires numerical accuracy verification (vs. CPU/native ATen) and performance benchmarking.

## 2. Directory Structure

- `src/flag_gems/`: Main package — `ops/` (standard kernels), `fused/` (fused kernels), `experimental_ops/` (incubating ops), `modules/` (`nn.Module` blocks like RMSNorm/RoPE), `runtime/` (multi-backend dispatchers & `backend/_<vendor>/`), `utils/` (`@pointwise_dynamic`, `LibTuner`, caching), `testing/` (test helpers).
- `cpp/`: C++ extensions (`include/`, `lib/` for `LibTritonJIT`, `ctest/`).
- `conf/`: `operators.yaml` (operator metadata source of truth), `backends.yaml` (toolchains), `ci_test_aliases.yaml` (test markers).
- `tests/` & `benchmark/`: Pytest accuracy suite and `triton.testing.do_bench` performance suite.
- `examples/` & `tools/ci_checks/`: Model examples, pre-tuning scripts, and CI/pre-commit rule checks.
- `setup.sh` & `pyproject.toml`: Per-backend environment setup and build config.

## 3. Operator Development Checklist

1. **Implement Kernel**:
   - Place in `src/flag_gems/{ops,fused,experimental_ops}/<op_name>.py` (`snake_case`, lowercase, no leading `_` or `.`).
   - **Generic Baseline ("公版算子")**: Must run on the official upstream Triton compiler on NVIDIA Ampere (e.g., A100). Place vendor-specific customizations under `src/flag_gems/runtime/backend/_<vendor>/`.
   - **vLLM Notice**: vLLM-specific fused operators have migrated to [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm); submit vLLM kernels there instead of `src/flag_gems/fused/`.
2. **Register & Export**:
   - Export in `src/flag_gems/__init__.py` (or `fused/__init__.py`) and map ATen ops in `_FULL_CONFIG`.
   - Sort `__all__` alphabetically via `python tools/ci_checks/sort_exports.py`.
3. **Register Metadata (`conf/operators.yaml`)**:
   ```yaml
     - id: <op_id>                  # Matches function name
       description: <Description>
       for: [<aten_op_name>]        # e.g., aten::add.Tensor
       labels: [aten]               # e.g., aten, Reduction, pointwise
       kind: [Math]                 # e.g., Math, NeuralNetwork, LinearAlg
       stages: [{beta: '5.4'}]      # alpha, beta, or stable with version
       name: [<abstract_op_name>]   # OpInfo style name
       source: [aten]
   ```
4. **Accuracy Tests (`tests/test_<op_id>.py`)**:
   - Decorate tests with `@pytest.mark.<op_id>`.
   - Run: `pytest tests/test_<op_id>.py` (add `--ref cpu` for CPU baseline, `--quick` for smoke tests).
5. **Benchmarks (`benchmark/test_<op_id>_perf.py`)**:
   - Implement using `Benchmark` and run: `pytest benchmark/test_<op_id>_perf.py -s`.
6. **Validate Static CI Checks**:
   ```bash
   python tools/ci_checks/check_operators_yaml.py --operators '["<op_id>"]'
   python tools/ci_checks/check_init_exports.py
   python tools/ci_checks/check_operator_markers.py --operators '["<op_id>"]'
   python tools/ci_checks/check_aten_operators.py --operators '["<op_id>"]'
   ```

## 4. Coding Standards & Environment

- **Formatting & Linting**: Python uses `black`, `isort` (`black` profile), and `flake8` (max line length 120); C++ uses `clang-format`. Always run `pre-commit run --all-files`.
- **Environment & Build**:
  ```bash
  # Install backend deps (e.g., nvidia-cuda128, ascend-cann900, metax-maca3810)
  [COMPILER=flagtree|triton] [ENABLE_CPP=1] ./setup.sh <backend>
  export USE_C_EXTENSION=1  # Enable C++ runtime dispatcher if built with ENABLE_CPP=1
  ```
- **Device Isolation**: Always isolate devices on shared nodes before testing:
  - NVIDIA: `CUDA_VISIBLE_DEVICES=<id>`
  - Ascend: `ASCEND_RT_VISIBLE_DEVICES=<id>`
  - Enflame: `TOPS_VISIBLE_DEVICES=<id>`
