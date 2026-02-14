# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied (current)

Hybrid strategy to match PyTorch indices while avoiding recursion:

- **float16 / bfloat16**: use `torch.kthvalue` (lower median) to avoid index
  drift from low-precision sorting
- **other dtypes**: use **PyTorch stable sort via redispatch**
  (`torch.ops.aten.sort.stable.redispatch`) and gather at `k`

This keeps correct values and improves index matching for low-precision dtypes
where stable sort still showed mismatches in large shapes.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
    k = (self.size(dim) + 1) // 2
    if self.dtype in (torch.float16, torch.bfloat16):
        return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)
    k = k - 1
    sorted_vals, sorted_idx = torch.ops.aten.sort.stable.redispatch(
        keyset, self, stable=True, dim=dim, descending=False
    )
    gather_index = torch.full(index_shape, k, device=..., dtype=torch.long)
    values = torch.take_along_dim(sorted_vals, gather_index, dim=dim)
    indices = torch.take_along_dim(sorted_idx, gather_index, dim=dim)
    if not keepdim:
        values = values.squeeze(dim)
        indices = indices.squeeze(dim)
    return values, indices
```

## Test Status

- **Accuracy tests**: pending re-run after hybrid fix
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-14
**Status**: recursion fixed; hybrid index selection implemented; awaiting re-test results
