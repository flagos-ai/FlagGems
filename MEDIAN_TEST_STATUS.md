# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied (current)

Use **first-occurrence index** for float16/bfloat16 and stable argsort for others:

- **float16 / bfloat16**:
  - compute median value via `torch.kthvalue` (lower median)
  - pick the **first occurrence** of that value along `dim`
    using a masked `min` over indices
- **other dtypes**: `torch.argsort(self, stable=True)` and gather at `k`

This matches PyTorch CUDA median’s tie-breaking (first occurrence) while keeping
the float32 path stable.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
    if self.dtype in (torch.float16, torch.bfloat16):
        k = (self.size(dim) + 1) // 2
        values = torch.kthvalue(self, k, dim=dim, keepdim=True).values
        indices = first_occurrence_indices(self, values, dim)
        values = torch.take_along_dim(self, indices, dim=dim)
    else:
        k = (self.size(dim) + 1) // 2 - 1
        sorted_idx = torch.argsort(self, dim=dim, stable=True)
        sorted_vals = torch.take_along_dim(self, sorted_idx, dim=dim)
        gather_index = torch.full(index_shape, k, device=..., dtype=sorted_idx.dtype)
        values = torch.take_along_dim(sorted_vals, gather_index, dim=dim)
        indices = torch.take_along_dim(sorted_idx, gather_index, dim=dim)
    if not keepdim:
        values = values.squeeze(dim)
        indices = indices.squeeze(dim)
    return values, indices
```

## Test Status

- **Accuracy tests**: pending re-run after first-occurrence fix
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-14
**Status**: recursion fixed; fp16/bf16 first-occurrence + stable argsort implemented; awaiting re-test results
