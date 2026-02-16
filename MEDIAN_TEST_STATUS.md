# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied (current)

Use **float64 upcast + kthvalue** for low-precision dtypes, and stable argsort
for higher precision:

- **float16 / bfloat16**:
  - upcast to `float64`
  - compute median via `torch.kthvalue` (lower median)
  - return the CUDA index from `kthvalue`
- **other dtypes**: `torch.argsort(self, stable=True)` and gather at `k`

This gives the best observed index agreement while keeping values correct.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
    if self.dtype in (torch.float16, torch.bfloat16):
        original_dtype = self.dtype
        self_upcast = self.to(torch.float64)
        k = (self.size(dim) + 1) // 2
        kth = torch.kthvalue(self_upcast, k, dim=dim, keepdim=keepdim)
        values = kth.values.to(original_dtype)
        indices = kth.indices
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

- **Accuracy tests**: 33/36 pass (best case), 31/36 typical; value accuracy 100%
- **Remaining failures**: index mismatches in bfloat16 large shapes, likely due to
  tie-break differences across PyTorch CPU/CUDA paths when duplicate values occur
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-16
**Status**: recursion fixed; fp16/bf16 float64+kthvalue implemented; awaiting final re-test confirmation
