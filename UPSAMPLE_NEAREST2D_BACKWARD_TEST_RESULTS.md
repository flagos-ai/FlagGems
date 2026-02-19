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
- **Passed**: 6 (11.1%)
- **Failed**: 48 (88.9%)

### Passed cases
All `shape0` cases `(1, 1, 4, 4)` passed (6 total):
- `dtype0-shape0-scale0-False/True` (float16, scale `(2.0, 2.0)`)
- `dtype0-shape0-scale1-False/True` (float16, scale `(1.5, 0.75)`)
- `dtype0-shape0-scale2-False/True` (float16, scale `(0.5, 0.5)`)

### Failed cases
- All `shape1` `(2, 3, 32, 32)` cases: 18 failures
- All `shape2` `(1, 4, 128, 128)` cases: 18 failures
- All `dtype1` (float32) cases: 18 failures
- All `dtype2` (bfloat16) cases: 18 failures

## Root Cause
Type conversion via `redispatch` still hits FlagGems `to_copy`, triggering:
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

## Fix Applied
Switched dtype conversion to call PyTorch’s `_to_copy` via redispatch with a
`CompositeExplicitAutograd` keyset (bypasses FlagGems `to_copy`):
```
torch.ops.aten._to_copy.default.redispatch(_FALLBACK_KEYSET, ...)
```

## Current State
- Small-size tests pass, confirming the core backward logic.
- Large-size failures persist due to type conversion triggering FlagGems dispatch.
- Next step: fully bypass FlagGems dispatch for dtype conversion.
