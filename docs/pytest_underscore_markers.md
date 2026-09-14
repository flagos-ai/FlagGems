# Pytest Underscore Marker Plugin

## Problem

Pytest doesn't allow accessing markers with underscore-prefixed names via attribute access. For example:

```python
@pytest.mark._stack  # This would fail without special handling
def test__stack():
    pass
```

This is a limitation in pytest's marker system, where underscore-prefixed attributes are treated specially.

## Solution

We've implemented an automatic marker registration system in `tests/conftest.py` that:

1. **Auto-registers underscore markers**: Reads all operator IDs from `conf/operators.yaml` and automatically registers any underscore-prefixed operators as pytest markers
2. **Validates test markers**: Checks that test functions for underscore operators have the correct marker

## Usage

### For Test Authors

Simply use the operator name directly in your marker, no special setup needed:

```python
import pytest

@pytest.mark._stack  # Works automatically!
def test__stack():
    # Your test here
    pass
```

**No need to:**
- Manually call `setattr(pytest.mark, "_stack", ...)`
- Use alternative naming like `@pytest.mark.underscore_stack`
- Import `_pytest.mark.structures`

### For CI

The CI check script `tools/ci_checks/check_operator_markers.py` works without modification. It expects `@pytest.mark._stack` for operator `_stack`, and the plugin ensures this works correctly.

### Running Tests

All standard pytest commands work as expected:

```bash
# Run all tests for _stack operator
pytest -m _stack

# Run specific test file
pytest tests/test__stack.py

# Collect test information
pytest --collect-only tests/test__stack.py
```

## Implementation Details

### Auto-Registration (`_register_underscore_markers`)

Called during `pytest_configure` hook:
- Loads `conf/operators.yaml`
- Finds all operator IDs starting with `_`
- Registers each as a marker using `setattr(pytest.mark, op_id, ...)`
- Adds marker documentation to pytest's marker registry

### Validation (`_validate_underscore_operator_markers`)

Called during `pytest_collection_modifyitems` hook:
- For test files named `test__xxx.py`, checks if test functions have `@pytest.mark._xxx`
- Skips tests with missing markers and provides helpful error messages
- Helps developers catch marker issues early

## Current Underscore-Prefixed Operators

As of this implementation, the following operators use underscore prefixes:
- `_adaptive_avg_pool3d_backward`
- `_flash_attention_forward`
- `_reshape_alias`
- `_weight_norm`

New underscore-prefixed operators are automatically supported when added to `operators.yaml`.

## Migration Notes

### Before (Old Approach)

```python
from _pytest.mark.structures import Mark, MarkDecorator

# Manual registration required
setattr(
    pytest.mark,
    "_reshape_alias",
    MarkDecorator(Mark("_reshape_alias", (), {}, _ispytest=True), _ispytest=True),
)

@pytest.mark._reshape_alias
def test__reshape_alias():
    pass
```

### After (New Approach)

```python
# No imports or setup needed!

@pytest.mark._reshape_alias
def test__reshape_alias():
    pass
```

The manual `setattr` code can be removed from existing test files. The plugin handles it automatically.
