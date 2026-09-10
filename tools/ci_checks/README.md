# CI Rule Checks

Lightweight, incremental rule checks for FlagGems PRs. These checks validate operator definitions, test coverage, and coding conventions without requiring GPU resources.

## Architecture

```
rule-check.yaml (GitHub Actions workflow)
└── derive-changed-operators (git diff → operator IDs)
    ├── check-operators-yaml       [blocking]  schema + duplicate ID
    ├── check-init-exports         [blocking]  __all__ + registry
    ├── check-kernelgen-tests      [blocking]  use_gems() prohibition
    ├── check-operator-markers     [blocking]  test file + marker
    ├── check-aten-operators       [blocking]  aten name format
    ├── check-api-logs             [warning]   logger.debug convention
    └── check-benchmark-coverage   [warning]   benchmark file existence
```

## How It Works

1. `derive_changed_operators.py` diffs `base..head` and maps changed files to operator IDs using `conf/operators.yaml`
2. Each check script receives the list of affected operators via `--operators '[...]'`
3. Blocking checks exit 1 on failure; warning checks always exit 0 but emit `::warning` annotations

## Running Locally

```bash
# Check a specific operator
python tools/ci_checks/check_operators_yaml.py --operators '["my_new_op"]'
python tools/ci_checks/check_operator_markers.py --operators '["my_new_op"]'
python tools/ci_checks/check_aten_operators.py --operators '["my_new_op"]'
python tools/ci_checks/check_kernelgen_tests.py --operators '["my_new_op"]'
python tools/ci_checks/check_api_logs.py --operators '["my_new_op"]'
python tools/ci_checks/check_performance_reference.py --operators '["my_new_op"]'

# Check all (for __init__.py)
python tools/ci_checks/check_init_exports.py

# Check all operators.yaml entries
python tools/ci_checks/check_operators_yaml.py --all
```

## Pre-commit Integration

Two checks run as local pre-commit hooks:

- **check-init-exports**: triggers when `src/flag_gems/__init__.py` is modified
- **check-operators-yaml**: triggers when `conf/operators.yaml` is modified

These run automatically on `git commit`. To install:

```bash
pre-commit install
```

## Check Details

### check_operators_yaml.py (blocking)

- Required fields: `id`, `name`, `labels`
- No duplicate `id` entries (global check, catches merge conflicts)
- Sort order check (disabled, available for future)

### check_init_exports.py (blocking)

- `__all__` must be sorted alphabetically
- `_FULL_CONFIG` must not have duplicate keys (known exceptions in allowlist)

### check_kernelgen_tests.py (blocking)

- Test files for KernelGen operators must NOT call `use_gems()`
- KernelGen tests should use `flag_gems.enable()` or direct kernel calls

### check_operator_markers.py (blocking)

- Each operator must have a test file: `tests/test_<op_id>.py`
- Test file must contain `@pytest.mark.<op_id>` decorator
- Supports aliases via `conf/ci_test_aliases.yaml` for non-standard names

### check_aten_operators.py (blocking)

- Operators with `aten` label must have a non-empty `for` field
- Each aten name must match the pattern: `aten::<name>.<overload>` or `aten::<name>`

### check_api_logs.py (warning)

- Source file should have `import logging` + `logger = logging.getLogger(__name__)`
- Public functions should call `logger.debug("GEMS <OP_NAME>")`
- Triton kernels (`@triton.jit`, `@pointwise_dynamic`) are automatically skipped

### check_performance_reference.py (warning)

- Each operator should have `benchmark/test_<op_id>.py`
- Benchmark file should contain a `.run()` call or Benchmark class

## Skipping / Exemptions

### CI-level skip

If a check is not relevant to your PR (e.g., documentation-only change), the workflow automatically skips when no operators are detected as changed.

### Test aliases

For operators that share test files (e.g., `addmm_out` tested in `test_addmm.py`), add a mapping to `conf/ci_test_aliases.yaml`:

```yaml
addmm_out: addmm
```

### Known duplicate keys

Some `_FULL_CONFIG` duplicates are intentional (overloads). These are listed in the allowlist inside `check_init_exports.py`.

## Dependencies

Only `pyyaml` is needed (already a project dependency). All checks use Python stdlib AST parsing — no import of FlagGems or PyTorch required.
