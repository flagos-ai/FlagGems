# Project: FlagGems

FlagGems is a high-performance, generic operator library implemented in the [Triton](https://github.com/openai/triton) language, designed to accelerate LLM training and inference across diverse hardware platforms by registering with the PyTorch ATen backend.

## Architecture & Conventions

- **Triton-First**: All core kernels should be implemented in Triton for backend neutrality and high performance.
- **PyTorch Integration**: Operators are registered with PyTorch's ATen backend to provide a seamless transition for model developers.
- **Performance & Accuracy**: Every operator change must be validated for both numerical accuracy and performance gains/consistency.

## Directory Structure

- `src/flag_gems/`: Main Python source code.
    - `ops/`: Individual operator implementations.
    - `fused/`: Fused operator implementations.
    - `testing/`: Internal testing utilities.
- `tests/`: Accuracy test suite (using `pytest`).
- `benchmark/`: Performance benchmarking scripts.
- `lib/`, `include/`: C++ source and headers for dispatcher and extensions.

## Coding Standards

- **Python Formatting**: Use `black` and `isort` (with `black` profile).
- **Python Linting**: `flake8` with a max line length of 120.
- **C++ Formatting**: `clang-format`.
- **Pre-commit**: Use `pre-commit` to ensure all standards are met before pushing.

## Common Workflows

### Accuracy Testing
Run accuracy tests using `pytest`:
```bash
pytest tests/test_your_op.py
```

### Performance Benchmarking
Run performance benchmarks:
```bash
python benchmark/test_your_op.py
```

### Adding a New Operator
1. Implement the Triton kernel in `src/flag_gems/ops/` (or `fused/` for fused ops).
2. Register the operator in the appropriate entry point.
3. Add accuracy tests in `tests/`.
4. Add performance benchmarks in `benchmark/`.
