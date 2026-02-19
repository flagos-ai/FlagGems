# UPSAMPLE_NEAREST2D_BACKWARD Test Results

## Summary
- **Operator**: `upsample_nearest2d_backward`
- **Branch**: `codex/upsample_nearest2d`
- **Status**: Tests **not executed** (missing `pytest`).

## Implementation Overview
- **Forward/Backward shared scale**: `_get_reciprocal_scale` (see `src/flag_gems/ops/upsample_nearest2d.py:15`).
- **Backward**: scatter-add accumulation with float32 upcast for `float16/bfloat16`.
- **Registration**:
  - `src/flag_gems/__init__.py:348`
  - `src/flag_gems/ops/__init__.py:224`

## Environment
- **Python**: `python3` available
- **PyTest**: not installed (`No module named pytest`)
- **CUDA / Triton**: not verified (GPU runtime required).

## Test Commands
Attempted locally:

```bash
python3 -m pytest tests/test_special_ops.py::test_upsample_nearest2d_backward -v
```

Result:
```
No module named pytest
```

Run on a CUDA-enabled environment (with pytest installed):

```bash
python -m pytest tests/test_special_ops.py::test_upsample_nearest2d_backward -v
```

Optional forward verification:

```bash
python -m pytest tests/test_special_ops.py::test_upsample_nearest2d -v
```

## Expected Coverage
The backward test in `tests/test_special_ops.py:859` covers:
- **Shapes**: `(1, 1, 4, 4)`, `(2, 3, 32, 32)`, `(1, 4, 128, 128)`
- **Scales**: `(2.0, 2.0)`, `(1.5, 0.75)`, `(0.5, 0.5)`
- **Dtypes**: `float16`, `float32`, `bfloat16`
- **Scale modes**:
  - `use_scales=False` (derive from sizes)
  - `use_scales=True` (explicit `scales_h/scales_w`)

## Results (to be filled after GPU run)
| Metric | Value |
|--------|-------|
| Total tests | N/A |
| Passed | N/A |
| Failed | N/A |
| Notes | N/A |

## Notes
- Update this report with real pass/fail counts after running tests on GPU.
