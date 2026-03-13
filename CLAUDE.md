# FlagGems Project Guide

## Project Overview

FlagGems is a high-performance, generic operator library implemented in Triton language. It's part of FlagOS, a unified, open-source AI system software stack. The project aims to accelerate LLM (Large Language Model) training and inference across diverse hardware platforms by providing backend-neutral kernels.

**Key Characteristics:**
- PyTorch-compatible operators via ATen backend registration
- Triton-based implementation for readability and performance
- Eager-mode ready, independent of `torch.compile`
- Multi-backend support for 10+ hardware platforms
- Automatic pointwise operator code generation
- Fast per-function runtime kernel dispatching

**Version:** 4.2.1.rc.0

**License:** Apache License (Version 2.0)

## Project Structure

```
FlagGems/
├── src/flag_gems/          # Main Python source code
│   ├── ops/                # Single operators (175+ operators)
│   ├── fused/              # Fused operators (30+ fused ops)
│   ├── experimental_ops/   # Experimental operators (147+ ops)
│   ├── modules/            # PyTorch modules
│   ├── patches/            # Patching scripts
│   ├── utils/              # Utilities and code generation
│   ├── runtime/            # Runtime system
│   ├── testing/            # Testing utilities
│   └── csrc/               # C source code
├── tests/                  # Accuracy test files
├── experimental_tests/     # Experimental operator tests
├── benchmark/              # Performance test files
├── examples/               # Example models
├── cmake/                  # CMake files for C-extension
├── include/                # C++ headers
├── lib/                    # C++ source code for operator lib
├── ctests/                 # C++ testing files
├── triton_src/             # Triton JIT functions source
├── docs/                   # Documentation
└── tools/                  # Development tools
```

## Architecture & Design

### Core Components

1. **Operator Library** (`src/flag_gems/ops/`)
   - 175+ individual operators implemented in Triton
   - PyTorch ATen backend registration for seamless integration
   - Support for float16, float32, and bfloat16 data types

2. **Fused Operators** (`src/flag_gems/fused/`)
   - 30+ fused operations for improved performance
   - Common fusion patterns for LLM workloads

3. **Experimental Operators** (`src/flag_gems/experimental_ops/`)
   - 147+ operators in development
   - Testing ground for new features before production release

4. **Runtime System** (`src/flag_gems/runtime/`)
   - LibEntry: Function-level kernel dispatching
   - Backend abstraction for multi-platform support
   - Kernel cache management

5. **Automatic Code Generation** (`src/flag_gems/utils/`)
   - Pointwise operator generation
   - Type promotion handling
   - Dynamic operator creation

### Backend Support

FlagGems supports 10+ hardware backends:

| Vendor     | Status          | float16 | float32 | bfloat16 |
|------------|-----------------|---------|---------|----------|
| NVIDIA     | ✅ Full         | ✅      | ✅      | ✅       |
| AMD        | ✅ Full         | ✅      | ✅      | ✅       |
| Cambricon  | ✅ Full         | ✅      | ✅      | ✅       |
| Hygon      | ✅ Full         | ✅      | ✅      | ✅       |
| Iluvatar   | ✅ Full         | ✅      | ✅      | ✅       |
| Kunlunxin  | ✅ Full         | ✅      | ✅      | ✅       |
| MetaX      | ✅ Full         | ✅      | ✅      | ✅       |
| Mthreads   | ✅ Full         | ✅      | ✅      | ✅       |
| AIPU       | ✅ Partial      | ✅      | ✅      | ✅       |
| Ascend     | ✅ Partial      | ✅      | ✅      | ✅       |

## Development Workflow

### Installation

**Prerequisites:**
```bash
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

**Standard Installation:**
```bash
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation .
```

**Editable Installation (for development):**
```bash
cd FlagGems
pip install --no-build-isolation -e .
```

**With FlagTree Backend (optional):**
```bash
pip install -r flag_tree_requirements/requirements_nvidia.txt
pip install --no-build-isolation .
```

### Code Formatting

FlagGems uses pre-commit hooks for code formatting:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Testing Strategy

#### 1. Unit Tests (`tests/`)
- Test operator correctness against PyTorch reference
- Organized by operator category
- Use pytest markers for selective testing

**Running tests:**
```bash
# Run all tests
pytest tests/

# Run specific operator tests
pytest -m <OP_NAME> tests/

# Run specific test file
pytest tests/test_binary_pointwise_ops.py
```

#### 2. Experimental Tests (`experimental_tests/`)
- Tests for experimental operators
- Separated into accuracy and performance tests

#### 3. Performance Benchmarks (`benchmark/`)
- Performance comparison against PyTorch ATen
- Organized by operator category:
  - `test_binary_pointwise_perf.py` - Binary operations
  - `test_unary_pointwise_perf.py` - Unary operations
  - `test_reduction_perf.py` - Reduction operations
  - `test_blas_perf.py` - BLAS operations
  - `test_attention_perf.py` - Attention operations
  - `test_norm_perf.py` - Normalization operations
  - And more...

**Running benchmarks:**
```bash
pytest benchmark/test_<category>_perf.py
```

### CI/CD Pipeline

The project has four main CI pipelines:

1. **Code Format Check**
   - Runs pre-commit hooks
   - Ensures code style consistency

2. **Op Unit Test**
   - Validates operator correctness
   - Must pass for all PRs

3. **Model Test**
   - Tests on real models (Bert, Llama-2-7b, Llava-1.5-7b)
   - Ensures end-to-end functionality

4. **Python Coverage**
   - Requires 90%+ coverage for new code
   - Excludes Triton JIT functions from coverage

**Local coverage check:**
```bash
cd $FlagGemsROOT
PR_ID=your_pr_id
bash tools/op-unit-test.sh
bash tools/model-test.sh
tools/code_coverage/coverage.sh PR_ID
```

## Contributing Guide

### Adding a New Operator

When adding a new operator to FlagGems, follow these steps:

1. **Implement the operator** in `src/flag_gems/ops/`
   - Write the Triton kernel implementation
   - Register with PyTorch ATen backend
   - Follow existing operator patterns

2. **Add unit tests** in `tests/`
   - Create test cases in the appropriate test file
   - Use `@pytest.mark.{OP_NAME}` decorator
   - Test against PyTorch reference implementation
   - Cover edge cases and different data types

3. **Add performance benchmarks** in `benchmark/`
   - Select appropriate benchmark file based on operator category
   - Add test cases following existing patterns
   - Consider custom input generators if needed

4. **Update documentation**
   - Add operator to operator list if applicable
   - Update relevant documentation files

### Adding Test Cases

**For unit tests:**
```python
@pytest.mark.my_operator
def test_my_operator():
    # Test implementation
    pass
```

**For performance benchmarks:**
- Choose the right file: `test_reduction_perf.py`, `test_binary_pointwise_perf.py`, etc.
- Check if existing benchmark classes fit your needs
- Add custom `input_generator` if needed
- Create new benchmark class if metric collection differs

### Pull Request Requirements

1. **Code Quality**
   - Pass pre-commit checks
   - Follow project coding standards
   - No security vulnerabilities (XSS, SQL injection, etc.)

2. **Testing**
   - All unit tests must pass
   - Add tests for new functionality
   - 90%+ code coverage for new code

3. **Documentation**
   - Describe what changed and why
   - Provide test cases if applicable
   - Update relevant documentation

4. **Review Process**
   - Requires approval from one team member
   - Must pass all CI checks
   - Address review feedback

## Usage Guide

### Basic Usage

**Enable FlagGems globally:**
```python
import torch
import flag_gems

flag_gems.enable()

# All PyTorch operations will now use FlagGems
A = torch.randn((1024, 1024), dtype=torch.float16, device='cuda')
B = torch.randn((1024, 1024), dtype=torch.float16, device='cuda')
C = torch.mm(A, B)  # Uses FlagGems implementation
```

**Enable FlagGems temporarily:**
```python
import torch
import flag_gems

A = torch.randn((1024, 1024), dtype=torch.float16, device='cuda')
B = torch.randn((1024, 1024), dtype=torch.float16, device='cuda')

with flag_gems.use_gems():
    C = torch.mm(A, B)  # Uses FlagGems implementation
```

### Using Experimental Operators

```python
import flag_gems

flag_gems.enable()
result = flag_gems.experimental_ops.rmsnorm(*args)

# Or with context manager
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```

### Device Information

```python
import flag_gems

print(flag_gems.device)        # Device name (e.g., 'cuda')
print(flag_gems.vendor_name)   # Vendor name (e.g., 'NVIDIA')
```

## Key Technical Features

### 1. LibEntry - Fast Kernel Dispatching

LibEntry is FlagGems' function-level kernel dispatching system that:
- Independently manages kernel cache
- Bypasses runtime overhead of Autotuner, Heuristics, and JitFunction
- Simplifies cache key format
- Reduces unnecessary key computation
- Preserves full tuning functionality when wrapping Autotuner/Heuristics

### 2. Automatic Pointwise Operator Generation

FlagGems provides automatic code generation for pointwise operators:
- Supports arbitrary input types and layouts
- Handles type promotion automatically
- Enables rapid operator development
- See `docs/pointwise_dynamic.md` for details

### 3. C++ Runtime (Optional)

FlagGems can be built with C++ extensions for improved performance:
- Reduces Python runtime overhead
- Better end-to-end performance
- Optional feature, works as pure Python package by default

## Common Development Tasks

### Running Tests Locally

**Run all unit tests:**
```bash
pytest tests/
```

**Run specific operator tests:**
```bash
pytest -m addcdiv tests/
pytest -m softmax tests/
```

**Run specific test file:**
```bash
pytest tests/test_binary_pointwise_ops.py
pytest tests/test_attention_ops.py
```

**Run experimental tests:**
```bash
pytest experimental_tests/
```

### Running Benchmarks

**Run all benchmarks:**
```bash
pytest benchmark/
```

**Run specific benchmark category:**
```bash
pytest benchmark/test_blas_perf.py
pytest benchmark/test_attention_perf.py
pytest benchmark/test_reduction_perf.py
```

### Code Coverage

**Generate coverage report:**
```bash
bash tools/op-unit-test.sh
bash tools/model-test.sh
tools/code_coverage/coverage.sh <PR_ID>
```

## Important Files and Directories

### Source Code
- `src/flag_gems/__init__.py` - Main entry point, operator registration
- `src/flag_gems/ops/` - Individual operator implementations
- `src/flag_gems/fused/` - Fused operator implementations
- `src/flag_gems/experimental_ops/` - Experimental operators
- `src/flag_gems/runtime/` - Runtime system and backend abstraction
- `src/flag_gems/utils/` - Code generation utilities
- `src/flag_gems/config.py` - Configuration management

### Tests
- `tests/` - Unit tests for operators
- `experimental_tests/` - Tests for experimental operators
- `benchmark/` - Performance benchmarks
- `ctests/` - C++ tests

### Documentation
- `docs/getting-started.md` - Quick start guide
- `docs/contribution.md` - Contribution guidelines
- `docs/features.md` - Feature documentation
- `docs/how_to_use_flaggems.md` - Usage details
- `docs/operators.md` - Operator list

### Build and Configuration
- `pyproject.toml` - Python package configuration
- `CMakeLists.txt` - CMake build configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

## Coding Standards and Best Practices

### Operator Implementation

1. **Follow existing patterns** - Review similar operators before implementing
2. **Use LibEntry** - Decorate Triton kernels with LibEntry for better performance
3. **Support multiple data types** - Implement float16, float32, and bfloat16 support
4. **Handle edge cases** - Test with empty tensors, single elements, large tensors
5. **Optimize for common shapes** - Consider typical LLM workload patterns

### Testing Guidelines

1. **Comprehensive coverage** - Test various input shapes, data types, and edge cases
2. **Use pytest markers** - Tag tests with `@pytest.mark.{OP_NAME}` for selective running
3. **Compare against PyTorch** - Validate correctness against PyTorch reference
4. **Performance benchmarks** - Add benchmarks for new operators
5. **Document test cases** - Explain what each test validates

### Code Style

1. **Follow PEP 8** - Use pre-commit hooks to enforce style
2. **Type hints** - Add type annotations where appropriate
3. **Docstrings** - Document public APIs and complex functions
4. **Comments** - Explain non-obvious implementation details
5. **Naming conventions** - Use clear, descriptive names

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure FlagGems is installed: `pip install --no-build-isolation -e .`
- Check Python version: Requires Python >= 3.8.0
- Verify PyTorch installation: Requires torch >= 2.2.0

**Test failures:**
- Run pre-commit checks: `pre-commit run --all-files`
- Check for missing dependencies: `pip install -e .[test]`
- Verify CUDA/GPU availability for hardware-specific tests

**Performance issues:**
- Consider building with C++ extensions for better performance
- Check backend configuration for your hardware
- Review LibEntry usage for kernel dispatching

## Resources

### Documentation
- [Getting Started Guide](./docs/getting-started.md)
- [Features Documentation](./docs/features.md)
- [Usage Guide](./docs/usage.md)
- [How to Use FlagGems](./docs/how_to_use_flaggems.md)
- [Contribution Guide](./docs/contribution.md)
- [Operator List](./docs/operators.md)

### Community
- GitHub Issues: https://github.com/flagos-ai/FlagGems/issues
- Email: contact@flagos.io
- WeChat Group: See QR code in README.md

### Related Projects
- FlagOS: https://flagos.io/
- FlagTree Compiler: https://github.com/flagos-ai/flagtree/
- OpenAI Triton: https://github.com/openai/triton

2. **Testing**
   - All unit tests must pass
   - Add tests for new functionality
   - 90%+ code coverage for new code

3. **Documentation**
   - Describe what changed and why
   - Provide test cases if applicable
   - Update relevant documentation

4. **Review Process**
   - Requires approval from one team member
   - Must pass all CI checks
   - Address review feedback

