# Median Test Status

## Root Cause (previous failure)

`median.dim` was implemented via `redispatch`, but `redispatch` still called back
into the registered FlagGems implementation, causing infinite recursion.

## Fix Applied (current)

Use **deterministic ordering for float16/bfloat16** and stable argsort for others:

- **float16 / bfloat16**: build an order-preserving integer key from IEEE bits
  (value order + index as tie-break), then `torch.argsort` on the key
- **other dtypes**: `torch.argsort(self, stable=True)`
- gather `k = (size + 1) // 2 - 1` along `dim` for both values and indices

This avoids dispatcher recursion and removes reliance on unstable tie-breaking
in low-precision sorts.

## Current Implementation (after fix)

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    dim = dim % self.dim()
    k = (self.size(dim) + 1) // 2 - 1
    if self.dtype in (torch.float16, torch.bfloat16):
        ordered = _ordered_key_fp16(self)  # value order + index tie-break
        key = ordered * size + index
        sorted_idx = torch.argsort(key, dim=dim)
    else:
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

- **Accuracy tests**: pending re-run after fp16/bf16 ordered-key fix
- **Performance tests**: pending (no benchmark case yet)

---

**Last updated**: 2026-02-14
**Status**: recursion fixed; fp16/bf16 ordered-key + stable argsort implemented; awaiting re-test results
