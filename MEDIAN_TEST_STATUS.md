# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied

Replace `redispatch` with a direct computation using `torch.kthvalue`:

- `k = (size + 1) // 2` (lower median for even sizes, 1-indexed for `kthvalue`)
- `torch.kthvalue` returns both values and indices, matching `median.dim`

This avoids dispatcher recursion entirely.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
    k = (self.size(dim) + 1) // 2
    return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)
```

## Test Status

- **Accuracy tests**: pending (re-run after fix)
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-14
**Status**: recursion fixed; awaiting re-test results
