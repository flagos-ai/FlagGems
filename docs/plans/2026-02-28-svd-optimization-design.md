# SVD Kernel Performance Optimization Design

**Date**: 2026-02-28
**Goal**: Improve SVD kernel performance from < 0.5x to >= 0.9x speedup vs PyTorch

## Problem

The current One-sided Jacobi SVD kernel uses a broadcast column extraction pattern that is O(M*N) per column access instead of O(M). For N=64 with 64 sweeps and 2016 pairs/sweep, this wastes ~2 billion FLOPs on extraction alone — roughly 64x the useful computation.

## Solution: Column-Major Global Memory Buffer

### Core Change

Replace register-resident matrices with column-major scratch buffers in global memory. For matrices up to 64x64 (16KB in float32), the entire working set fits in L1 cache.

**Before** (O(M*N) per column):
```python
a_p = tl.sum(A * (col_idx[None, :] == p).to(tl.float32), axis=1)
```

**After** (O(M) per column):
```python
a_p = tl.load(A_work_ptr + pid * batch_stride + p * M + row_idx, mask=row_idx < M)
```

### Secondary Optimizations

1. **Sort outside kernel**: Move descending-order sorting from in-kernel bubble sort to post-kernel `torch.argsort` + `torch.gather`. Eliminates O(N^2) broadcast sort loop.

2. **Early convergence detection**: After each sweep, check if total off-diagonal energy < epsilon. Most matrices converge in 5-15 sweeps instead of 64.

### Kernel Flow

1. Load input A from global memory → write to A_work (column-major layout)
2. Initialize V_work = Identity (column-major layout)
3. For each sweep (up to NUM_SWEEPS, with early exit):
   a. For each pair (p, q) where p < q < N:
      - Load columns p, q from A_work via `tl.load`: O(M)
      - Compute Gram entries (alpha, beta, gamma): O(M)
      - Compute Jacobi rotation (c, s): O(1)
      - Apply rotation, store back via `tl.store`: O(M)
      - Same for V_work columns: O(N)
4. Extract S = column norms of A_work
5. Normalize U = A_work / S (if compute_uv)
6. Store S, U, V to output buffers (unsorted)

### Wrapper Flow

1. Dimension handling, transpose normalization (same as before)
2. Allocate scratch buffers: A_work (batch, N, M) contiguous, V_work (batch, N, N)
3. Launch kernel
4. Sort: `argsort(S, descending=True)` → `gather` on S, U, V
5. Reshape, transpose undo (same as before)

### What Stays the Same

- Public API: `svd(input, some=True, compute_uv=True) → SVDResult(U, S, V)`
- All edge case handling (empty batch, zero matrix, m < n transpose)
- Test suite and benchmark infrastructure
- MAX_SVD_DIM = 64 constraint
- Jacobi rotation mathematics

### Expected Impact

| Optimization | Expected Impact |
|-------------|----------------|
| Column-major buffer | 10-50x on inner loop |
| Sort outside kernel | ~10% kernel time saved |
| Early convergence | 2-4x fewer sweeps |
| **Combined** | **20-100x total kernel speedup** |

### Files Modified

- `src/flag_gems/ops/svd.py` — kernel and wrapper rewrite
- No changes to registration, tests, or benchmarks (interface unchanged)
