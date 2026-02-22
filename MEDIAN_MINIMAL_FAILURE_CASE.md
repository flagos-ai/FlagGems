# Median Minimal Failure Case (bfloat16)

## Summary
Index mismatches are tied to **tie-break behavior** when the median value occurs
multiple times. The current implementation uses **float64 upcast + kthvalue**
for fp16/bf16 and still sees rare mismatches in large bfloat16 cases.

## Test Results Summary

### Test Statistics (latest)
- **Total tests**: 36
- **Passed**: 33 best / 31 typical
- **Failed**: 3–5 (seed-dependent)
- **Value accuracy**: 100% ✅
- **Index accuracy**: 86–92%

**Note**: Failure count may vary with random seed (3-5 failures)

### Typical Failed Test Cases
1. `dtype2-True-shape4-1` - bfloat16, keepdim=True, shape=(1024, 1024), dim=1
2. `dtype2-False-shape3-1` - bfloat16, keepdim=False, shape=(64, 64), dim=1 (seed-dependent)
3. `dtype2-False-shape4-1` - bfloat16, keepdim=False, shape=(1024, 1024), dim=1
4. Other bfloat16 large-size test cases (seed-dependent)

## Minimal Repro

### Example 1: Simple Repeated Values (tie-break sensitive)

```python
import torch
import flag_gems

data = torch.tensor(
    [-0.05126953125, 0.1, -0.05126953125, 0.2, -0.05126953125],
    dtype=torch.bfloat16,
    device="cuda",
)

# PyTorch reference (CPU)
ref_vals, ref_idx = torch.median(data.cpu(), dim=0, keepdim=False)

# FlagGems
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(data, dim=0, keepdim=False)
```

## Key Details
- **shape**: `(5,)`
- **dim**: `0`
- **keepdim**: `False`
- **dtype**: `bfloat16`
- **data**: `[-0.05126953125, 0.1, -0.05126953125, 0.2, -0.05126953125]`
- **median value**: `-0.05126953125`
- **value positions**: `[0, 2, 4]`
- **Tie-break note**: CPU and CUDA can select different indices among duplicates;
  this is the source of remaining mismatches.

### Example 2: Extracted from Actual Failure Case

From `dtype2-True-shape4-1` failure case:

```python
# Extracted from row 25 of (1024, 1024) matrix
# Median value -0.05126953125 appears at positions: [0, 58, 135, 460, 633, 1004]
# PyTorch selects: 1004 (last occurrence)
# Previous FlagGems selected: 633 (first matching position, but actually middle)
```

**Key Information**:
- **shape**: `(1024,)`
- **dim**: `0`
- **keepdim**: `False`
- **dtype**: `torch.bfloat16`
- **median value**: `-0.05126953125`
- **value positions**: `[0, 58, 135, 460, 633, 1004]`
- **PyTorch index**: `1004`
- **Previous FlagGems index**: `633`
- **Failure reason**: PyTorch selects last occurrence, previous implementation selected first matching position

### Example 3: Other Test Cases

```python
# Test case: [1.0, 2.0, 2.0, 2.0, 3.0]
# Median value: 2.0, positions: [1, 2, 3]
# PyTorch index: 2 (middle position)
# Previous FlagGems index: 1 (first)
# Mismatch

# Test case: [2.0, 2.0, 2.0, 3.0, 4.0]
# Median value: 2.0, positions: [0, 1, 2]
# PyTorch index: 2 (last)
# Previous FlagGems index: 0 (first)
# Mismatch
```

## Problem Analysis

### PyTorch CUDA median tie-break behavior

From test results, PyTorch CUDA's `median` behavior when selecting indices:
1. **When multiple identical median values exist, select the index of the last occurrence**
2. This is inconsistent with the previous implementation (which selected the first occurrence)

### Previous Implementation Issue

The previous implementation used `torch.min(masked_index, dim=dim)` to select indices, which selects the **first** matching position. However, PyTorch CUDA selects the **last** matching position.

### Fix Direction

Change `torch.min(masked_index, dim=dim)` to select the last matching position, for example:
- Use `torch.max` instead of `torch.min`
- Or use `torch.argmax` on reversed masked_index
- Or use other methods to ensure selecting the last matching position

---

**Last updated**: 2026-02-16
**Status**: Remaining failures are tie-break differences for duplicate values in bfloat16
