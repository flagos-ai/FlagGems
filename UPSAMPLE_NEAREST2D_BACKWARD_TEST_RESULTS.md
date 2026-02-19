# UPSAMPLE_NEAREST2D_BACKWARD Test Results

## Summary
- **Operator**: `upsample_nearest2d_backward`
- **Branch**: `codex/upsample_nearest2d`
- **Status**: Tests executed on GPU (user-run); local environment lacks pytest.

## Implementation Overview
- **Forward/Backward shared scale**: `_get_reciprocal_scale` (see `src/flag_gems/ops/upsample_nearest2d.py:15`).
- **Backward**: scatter-add accumulation with float32 upcast for `float16/bfloat16`.
- **Registration**:
  - `src/flag_gems/__init__.py:348`
  - `src/flag_gems/ops/__init__.py:224`

## Environment
- **User-run GPU environment**: CUDA-enabled (details not provided)
- **Local environment**: `python3` available, `pytest` missing

## Test Commands
Run on a CUDA-enabled environment:

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

## Results (user-run)
### Overall
- **Total tests**: 54
- **Passed**: 54 (100%)
- **Failed**: 0

## Root Cause
FlagGems intercepts multiple ops (`to_copy`, `scatter_add_`) causing:
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

## Fix Applied
Switched dtype conversion to call PyTorch’s `_to_copy` via redispatch with a
`CompositeExplicitAutograd` keyset (bypasses FlagGems `to_copy`):
```
torch.ops.aten._to_copy.default.redispatch(_FALLBACK_KEYSET, ...)
```
Also switched scatter accumulation to PyTorch’s `scatter_add_` via redispatch:
```
torch.ops.aten.scatter_add_.default.redispatch(
    _FALLBACK_KEYSET, grad_input_flat, 2, index, grad_output_flat
)
```

## Current State
- All tests pass after bypassing FlagGems for both `_to_copy` and `scatter_add_`.
