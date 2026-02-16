# Median Minimal Failure Case (bfloat16)

## Summary
PyTorch CUDA `median` chooses the **last** occurrence index when the median value
appears multiple times. The previous FlagGems implementation chose the **first**
occurrence, which caused index mismatches while values were correct.

## Minimal Repro

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
# ref_idx == 4 (last occurrence)

# FlagGems
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(data, dim=0, keepdim=False)
# previous res_idx == 0 (first occurrence)
```

## Key Details
- **shape**: `(5,)`
- **dim**: `0`
- **dtype**: `bfloat16`
- **median value**: `-0.05126953125`
- **value positions**: `[0, 2, 4]`
- **PyTorch CUDA index**: `4` (last occurrence)

## Fix Direction
Select the **last occurrence** of the median value for fp16/bf16 by taking
`torch.max(masked_index, dim=dim)` instead of `torch.min`.
