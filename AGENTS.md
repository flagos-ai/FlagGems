# Project: FlagGems

FlagGems is a high-performance, generic operator library implemented in the [Triton](https://github.com/openai/triton) language. It is designed to accelerate Large Language Model (LLM) training and inference across diverse AI hardware platforms by seamlessly registering with PyTorch's ATen backend.

---

## 1. Core Architecture & Principles

- **Triton-First**: All core kernels should be implemented in Triton for backend neutrality, maintainability, and near-native performance across multiple AI chips.
- **Seamless PyTorch Integration**: Operators register into PyTorch's ATen dispatcher or vLLM custom ops, allowing zero-code-change drop-in acceleration for model developers.
- **Host Function Isolation**: Operator host functions must NOT invoke PyTorch compute-heavy operators (only data allocation/initialization like `empty_like`, `zeros`, `randn` or other FlagGems operators are allowed). Operators must invoke device-side kernels.
- **Dual Validation (Accuracy & Performance)**: Every operator modification must pass numerical accuracy verification against reference implementations (e.g. CPU/native ATen) and benchmark performance tests.
- **Multi-Backend Neutrality**: Ensure backend portability across NVIDIA, Ascend, Hygon, Moore Threads, MetaX, Iluvatar, Kunlunxin, Cambricon, and other supported accelerators.

---

## 2. Directory Structure

```none
FlagGems
├── src/flag_gems/            # Main Python library source
│   ├── ops/                  # Pointwise, reduction, and fundamental operator kernels
│   ├── fused/                # Fused operator kernels (e.g. FlashAttention, MoE align)
│   ├── modules/              # PyTorch nn.Module neural network blocks (RMSNorm, RoPE, etc.)
│   ├── runtime/              # Multi-backend device runtime and dispatchers
│   ├── utils/                # Dynamic code generation (@pointwise_dynamic), LibTuner, caching
│   ├── experimental_ops/     # Incubating / experimental operators
│   └── testing/              # Internal testing utilities and accuracy helpers
├── cpp/                      # C++ dispatching extensions
│   ├── include/              # Public C++ headers
│   ├── lib/                  # C++ wrapped operator implementations (LibTritonJIT)
│   └── ctest/                # Native C++ unit tests
├── conf/                     # Configuration files & operator inventory
│   ├── operators.yaml        # Single source of truth for operator metadata
│   ├── backends.yaml         # Multi-hardware dependency and toolchain definitions
│   └── ci_test_aliases.yaml  # Test marker aliases for CI checks
├── tests/                    # Accuracy test suite (pytest)
├── benchmark/                # Performance benchmarking suite (triton.testing.do_bench)
├── examples/                 # End-to-end model examples and pre-tuning scripts
├── tools/                    # Tooling, packaging, and CI validation checks
│   └── ci_checks/            # Lightweight rule checks run in CI and pre-commit
├── setup.sh                  # Recommended environment setup script per backend
└── pyproject.toml            # Build configuration and project dependencies
```

---

## 3. Workflow: Adding or Modifying an Operator

When adding a new operator or modifying existing operators, complete all steps in this checklist:

### Step 1: Implement the Kernel
- Place standard operators in `src/flag_gems/ops/<op_name>.py`.
- Place fused operators in `src/flag_gems/fused/<op_name>.py`.
- Place incubating operators in `src/flag_gems/experimental_ops/<op_name>.py`.
- **Generic Baseline Implementations ("公版算子")**: Implementations in `ops/`, `fused/`, and `experimental_ops/` serve as the generic baseline. They **MUST run based on the official upstream Triton compiler on the NVIDIA Ampere architecture (e.g. A100)** to ensure maximum compatibility across hardware platforms. Vendor-specific optimizations must be placed under `src/flag_gems/runtime/backend/_<vendor>/`.
- **vLLM Operators Notice**: vLLM-specific fused operators have migrated to the [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm) repository. Developers and agents should submit all vLLM optimization and kernel code directly to FlagGems-vllm rather than `src/flag_gems/fused/`.
- Follow operator naming conventions: `snake_case`, lowercase, no leading underscore, no dots (`.`).

### Step 2: Register in `_FULL_CONFIG` and Export API
- Export the operator function in `src/flag_gems/__init__.py` (or `fused/__init__.py`).
- Add ATen operator mappings in `_FULL_CONFIG` in `src/flag_gems/__init__.py`.
- Keep `__all__` sorted alphabetically. You can run the auto-formatter:
  ```bash
  python tools/ci_checks/sort_exports.py
  ```

### Step 3: Register Metadata in `conf/operators.yaml`
Add a dedicated entry for the operator in `conf/operators.yaml`:
```yaml
  - id: <op_id>                  # Unique ID matching the function name
    description: <Description>   # Brief explanation of operator semantics
    for:
      - <aten_op_name>           # e.g., aten::add.Tensor
    labels:
      - aten                     # e.g., aten, Reduction, pointwise
    kind:
      - Math                     # e.g., Math, NeuralNetwork, LinearAlg
    stages:
      - beta: '5.4'              # Mature stage: alpha, beta, or stable with version
    name:
      - <abstract_op_name>       # OpInfo style abstract name
    source:
      - aten                     # Source framework
```

### Step 4: Add Unit Accuracy Tests
Create `tests/test_<op_id>.py` decorated with the matching marker:
```python
import pytest
import torch
import flag_gems

@pytest.mark.<op_id>
def test_accuracy_<op_id>():
    ...
```
Run accuracy tests:
```bash
# Run on current target accelerator
pytest tests/test_<op_id>.py

# Run with CPU as reference baseline
pytest tests/test_<op_id>.py --ref cpu

# Quick smoke test with reduced shapes
pytest tests/test_<op_id>.py --quick
```

### Step 5: Add Performance Benchmarks
Create `benchmark/test_<op_id>_perf.py` leveraging the `Benchmark` suite:
```bash
pytest benchmark/test_<op_id>_perf.py -s
```

### Step 6: Validate Static CI Rules Locally
Run CI rule checks before committing to prevent PR failures:
```bash
python tools/ci_checks/check_operators_yaml.py --operators '["<op_id>"]'
python tools/ci_checks/check_init_exports.py
python tools/ci_checks/check_operator_markers.py --operators '["<op_id>"]'
python tools/ci_checks/check_aten_operators.py --operators '["<op_id>"]'
```

---

## 4. Coding Standards & Pre-commit

- **Python Formatting**: `black` and `isort` (with `black` profile).
- **Python Linting**: `flake8` with max line length of 120.
- **C++ Formatting**: `clang-format`.
- **Pre-commit**: Always run pre-commit hooks to ensure code cleanliness:
  ```bash
  pre-commit run --all-files
  ```

---

## 5. Multi-Backend & Environment Tips

- **Backend Dependency Installation**:
  Use `setup.sh` to automatically install the exact compatible Python, PyTorch, and vendor toolchains:
  ```bash
  ./setup.sh <backend>   # e.g., nvidia-cuda128, ascend-cann900, metax-maca3810
  ```
- **Device Isolation**:
  Always isolate devices in multi-tenant or shared nodes before executing tests:
  - NVIDIA: `CUDA_VISIBLE_DEVICES=<id>`
  - Ascend: `ASCEND_RT_VISIBLE_DEVICES=<id>`
  - Enflame: `TOPS_VISIBLE_DEVICES=<id>`
- **Compiler Selection**:
  - `COMPILER=flagtree ./setup.sh <backend>` to use FlagTree compiler.
  - `COMPILER=triton ./setup.sh <backend>` to use vendor upstream Triton.
- **C++ Extension Support**:
  Build with `ENABLE_CPP=1 ./setup.sh <backend>`, and enable at runtime via `export USE_C_EXTENSION=1`.
