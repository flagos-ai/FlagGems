# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied

Replace `redispatch` with a direct computation using **stable sort**:

- Use `torch.sort(..., stable=True)` along the target dim
- Select the lower median index `k = (size + 1) // 2 - 1`
- Gather both values and indices at that position

This avoids dispatcher recursion and matches PyTorch's index selection when
duplicate median values exist.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
    k = (self.size(dim) + 1) // 2 - 1
    sorted_vals, sorted_idx = torch.sort(self, dim=dim, stable=True)
    gather_index = torch.full(index_shape, k, device=..., dtype=torch.long)
    values = torch.take_along_dim(sorted_vals, gather_index, dim=dim)
    indices = torch.take_along_dim(sorted_idx, gather_index, dim=dim)
    return values, indices
```

## Test Status

- **Accuracy tests**: pending re-run after stable-sort fix
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-14
**Status**: recursion fixed; index matching should align with PyTorch; awaiting re-test results
