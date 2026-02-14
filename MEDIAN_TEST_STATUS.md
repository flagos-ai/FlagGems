# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied (current)

Use **stable argsort** to match PyTorch’s median index selection deterministically:

- `sorted_idx = torch.argsort(self, dim=dim, stable=True)`
- gather `k = (size + 1) // 2 - 1` along `dim` for both values and indices

This avoids dispatcher recursion and removes reliance on redispatch or `kthvalue`
tie-breaking for low-precision dtypes.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
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

- **Accuracy tests**: pending re-run after argsort-stable fix
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-14
**Status**: recursion fixed; argsort-stable index selection implemented; awaiting re-test results
