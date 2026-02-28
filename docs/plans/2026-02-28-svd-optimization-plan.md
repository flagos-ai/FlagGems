# SVD Kernel Performance Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the One-sided Jacobi SVD kernel from < 0.5x to >= 0.9x speedup vs PyTorch by replacing broadcast column extraction with direct global-memory column access, moving sorting outside the kernel, and adding early convergence detection.

**Architecture:** The kernel reads/writes columns of the working matrices (A_work, V_work) via `tl.load`/`tl.store` on column-major global-memory scratch buffers instead of using O(N) broadcast extraction on register-resident matrices. Post-kernel sorting uses `torch.argsort` + `torch.gather`. The public API is unchanged.

**Tech Stack:** Python, Triton, PyTorch, FlagGems utilities (`libentry`, `tle`, `torch_device_fn`)

**Design doc:** `docs/plans/2026-02-28-svd-optimization-design.md`

---

### Task 1: Bring SVD baseline code into this worktree

**Files:**
- Create: `src/flag_gems/ops/svd.py` (from `feat/add-svd` branch)
- Modify: `src/flag_gems/ops/__init__.py` (add svd import)
- Modify: `src/flag_gems/__init__.py` (add svd to `_FULL_CONFIG`)
- Copy test additions from `feat/add-svd:tests/test_special_ops.py`
- Copy benchmark additions from `feat/add-svd:benchmark/test_special_perf.py`

**Step 1: Cherry-pick SVD files from feat/add-svd**

```bash
# Copy svd.py from the feat/add-svd branch
git show feat/add-svd:src/flag_gems/ops/svd.py > src/flag_gems/ops/svd.py

# Apply the registration changes (ops/__init__.py and __init__.py)
# These are small diffs — add the import and _FULL_CONFIG entry
```

In `src/flag_gems/ops/__init__.py`, add between `sum` and `tan` imports:
```python
from flag_gems.ops.svd import svd
```

And in the `__all__` list, add `"svd"` between `"sub_"` and `"sum"`.

In `src/flag_gems/__init__.py`, add to `_FULL_CONFIG` between `"sub_.Tensor"` and `"sum"`:
```python
("svd", svd),
```

Copy SVD test section (lines 1895-2048) from `feat/add-svd:tests/test_special_ops.py` and append to `tests/test_special_ops.py`.

Copy SVD benchmark section (lines 826-863) from `feat/add-svd:benchmark/test_special_perf.py` and append to `benchmark/test_special_perf.py`.

**Step 2: Run baseline tests to verify correctness**

```bash
pytest tests/test_special_ops.py -m svd -v --timeout=300
```

Expected: All SVD tests PASS (same as on feat/add-svd).

**Step 3: Commit baseline**

```bash
git add src/flag_gems/ops/svd.py src/flag_gems/ops/__init__.py src/flag_gems/__init__.py tests/test_special_ops.py benchmark/test_special_perf.py
git commit -m "feat(svd): bring SVD baseline from feat/add-svd branch"
```

---

### Task 2: Rewrite kernel — column-major global memory buffer for A

This is the core optimization. Replace register-resident A matrix with a column-major scratch buffer.

**Files:**
- Modify: `src/flag_gems/ops/svd.py`

**Step 1: Modify the kernel signature**

In `jacobi_svd_kernel`, replace the current signature (lines 36-59) with:

```python
@libentry()
@triton.jit
def jacobi_svd_kernel(
    A_ptr,           # Input A (batch, M, N) row-major — read only
    A_work_ptr,      # Scratch buffer for A — column-major (batch, N, M) contiguous
    V_work_ptr,      # Scratch buffer for V — column-major (batch, N, N) contiguous
    U_ptr,           # Output U
    S_ptr,           # Output S
    V_ptr,           # Output V
    # Input A strides
    batch_stride_A,
    m_stride_A,
    n_stride_A,
    # A_work layout: A_work[batch, col, row] is contiguous
    # stride for batch = N * M, stride for col = M, stride for row = 1
    aw_batch_stride,
    aw_col_stride,   # = M (column-major: each column is M contiguous elements)
    # V_work layout: V_work[batch, col, row] is contiguous
    vw_batch_stride,
    vw_col_stride,   # = N
    # Output strides
    batch_stride_U,
    m_stride_U,
    k_stride_U,
    batch_stride_S,
    batch_stride_V,
    n_stride_V,
    k_stride_V,
    # Dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    out_k_U: tl.constexpr,
    out_k_V: tl.constexpr,
    compute_uv: tl.constexpr,
    NUM_SWEEPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
```

**Step 2: Rewrite the kernel body — initialization**

Replace lines 69-92 (load A into registers, init V identity) with:

```python
    pid = tle.program_id(0)
    row_idx = tl.arange(0, BLOCK_M)
    row_mask = row_idx < M

    # Base pointers for this batch element's scratch buffers
    aw_base = A_work_ptr + pid * aw_batch_stride
    vw_base = V_work_ptr + pid * vw_batch_stride

    # Copy input A into A_work (column-major)
    # A_work[col, row] = A[row, col]
    col_idx = tl.arange(0, BLOCK_N)
    for j in range(N):
        # Load column j from input A (row-major)
        a_col = tl.load(
            A_ptr + pid * batch_stride_A + row_idx * m_stride_A + j * n_stride_A,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        # Store to A_work column-major: A_work[j, :] = a_col
        tl.store(aw_base + j * aw_col_stride + row_idx, a_col, mask=row_mask)

    # Initialize V_work = Identity (column-major)
    if compute_uv:
        v_row_idx = tl.arange(0, BLOCK_N)
        v_mask = v_row_idx < N
        for j in range(N):
            v_col = tl.where(
                v_row_idx == j,
                tl.full((BLOCK_N,), 1.0, dtype=tl.float32),
                tl.full((BLOCK_N,), 0.0, dtype=tl.float32),
            )
            tl.store(vw_base + j * vw_col_stride + v_row_idx, v_col, mask=v_mask)
```

**Step 3: Rewrite the Jacobi sweep loop**

Replace lines 94-153 with:

```python
    # Jacobi sweeps with column-major load/store
    for _sweep in range(NUM_SWEEPS):
        for p in range(N):
            for q in range(p + 1, N):
                # Load columns p and q from A_work — O(M) direct load
                a_p = tl.load(aw_base + p * aw_col_stride + row_idx, mask=row_mask, other=0.0)
                a_q = tl.load(aw_base + q * aw_col_stride + row_idx, mask=row_mask, other=0.0)

                # Compute Gram matrix entries
                alpha = tl.sum(a_p * a_p)
                beta = tl.sum(a_q * a_q)
                gamma = tl.sum(a_p * a_q)

                # Check convergence
                converged = tl.abs(gamma) < 1e-7 * tl.sqrt(alpha * beta + 1e-30)

                # Compute Jacobi rotation angle
                safe_gamma = tl.where(converged, tl.full((), 1.0, dtype=tl.float32), gamma)
                zeta = (beta - alpha) / (2.0 * safe_gamma)
                sign_zeta = tl.where(zeta >= 0, tl.full((), 1.0, dtype=tl.float32), tl.full((), -1.0, dtype=tl.float32))
                t = sign_zeta / (tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c

                # Skip rotation if converged
                c = tl.where(converged, tl.full((), 1.0, dtype=tl.float32), c)
                s = tl.where(converged, tl.full((), 0.0, dtype=tl.float32), s)

                # Apply rotation to A columns — O(M) store
                new_a_p = c * a_p - s * a_q
                new_a_q = s * a_p + c * a_q
                tl.store(aw_base + p * aw_col_stride + row_idx, new_a_p, mask=row_mask)
                tl.store(aw_base + q * aw_col_stride + row_idx, new_a_q, mask=row_mask)

                # Apply rotation to V columns
                if compute_uv:
                    v_row_idx2 = tl.arange(0, BLOCK_N)
                    v_mask2 = v_row_idx2 < N
                    v_p = tl.load(vw_base + p * vw_col_stride + v_row_idx2, mask=v_mask2, other=0.0)
                    v_q = tl.load(vw_base + q * vw_col_stride + v_row_idx2, mask=v_mask2, other=0.0)
                    new_v_p = c * v_p - s * v_q
                    new_v_q = s * v_p + c * v_q
                    tl.store(vw_base + p * vw_col_stride + v_row_idx2, new_v_p, mask=v_mask2)
                    tl.store(vw_base + q * vw_col_stride + v_row_idx2, new_v_q, mask=v_mask2)
```

**Step 4: Rewrite singular value extraction and output storage**

Replace lines 155-232 (extraction + sorting + store) with:

```python
    # Extract singular values as column norms of A_work
    s_idx = tl.arange(0, BLOCK_N)
    s_mask = s_idx < K
    S_vals = tl.full((BLOCK_N,), 0.0, dtype=tl.float32)
    for j in range(N):
        a_col_j = tl.load(aw_base + j * aw_col_stride + row_idx, mask=row_mask, other=0.0)
        norm_sq = tl.sum(a_col_j * a_col_j)
        # Store norm at position j
        S_vals = tl.where(s_idx == j, tl.sqrt(norm_sq), S_vals)

    # Store S (unsorted — sorting done in wrapper)
    s_ptrs = S_ptr + pid * batch_stride_S + s_idx
    tl.store(s_ptrs, S_vals, mask=s_mask)

    # Compute and store U = A_work_col / S_col (normalize columns)
    if compute_uv:
        for j in range(N):
            a_col_j = tl.load(aw_base + j * aw_col_stride + row_idx, mask=row_mask, other=0.0)
            s_j = tl.sum(S_vals * (s_idx == j).to(tl.float32))
            safe_s_j = tl.where(s_j > 1e-10, s_j, tl.full((), 1.0, dtype=tl.float32))
            u_col_j = a_col_j / safe_s_j

            # Store U column j
            u_mask_j = row_mask & (j < out_k_U)
            u_ptrs = U_ptr + pid * batch_stride_U + row_idx * m_stride_U + j * k_stride_U
            tl.store(u_ptrs, u_col_j, mask=u_mask_j)

        # Store V from V_work
        v_row_idx3 = tl.arange(0, BLOCK_N)
        for j in range(N):
            v_col_j = tl.load(vw_base + j * vw_col_stride + v_row_idx3, mask=v_row_idx3 < N, other=0.0)
            v_mask_j = (v_row_idx3 < N) & (j < out_k_V)
            v_ptrs = V_ptr + pid * batch_stride_V + v_row_idx3 * n_stride_V + j * k_stride_V
            tl.store(v_ptrs, v_col_j, mask=v_mask_j)
```

**Step 5: Run tests**

```bash
pytest tests/test_special_ops.py -m svd -v --timeout=300
```

Expected: All tests PASS with the new kernel (correctness unchanged).

**Step 6: Commit**

```bash
git add src/flag_gems/ops/svd.py
git commit -m "perf(svd): replace broadcast column extraction with global memory buffer"
```

---

### Task 3: Move sorting outside the kernel

**Files:**
- Modify: `src/flag_gems/ops/svd.py` (wrapper function)

**Step 1: Add post-kernel sorting in the wrapper**

In the `svd()` wrapper function, after the kernel call (after line 351 in original), add sorting logic before the transpose undo:

```python
    # Sort singular values in descending order (outside kernel for efficiency)
    sorted_indices = torch.argsort(S_out, dim=-1, descending=True)
    S_out = torch.gather(S_out, -1, sorted_indices)

    if compute_uv:
        # Reorder U columns to match sorted S
        idx_U = sorted_indices.unsqueeze(-2).expand_as(U_out)
        U_out = torch.gather(U_out, -1, idx_U)

        # Reorder V columns to match sorted S
        idx_V = sorted_indices.unsqueeze(-2).expand_as(V_out)
        V_out = torch.gather(V_out, -1, idx_V)
```

**Step 2: Update the wrapper to allocate scratch buffers and pass new strides**

Replace the kernel launch section of the wrapper with:

```python
    # Allocate column-major scratch buffers
    # A_work shape: (batch_size, n, m) — column j of A is A_work[batch, j, :]
    A_work = torch.empty(batch_size, n, m, device=input.device, dtype=torch.float32)
    V_work = torch.empty(batch_size, n, n, device=input.device, dtype=torch.float32)

    BLOCK_M = _next_power_of_2(m)
    BLOCK_N = _next_power_of_2(n)
    num_sweeps = max(10, n)
    grid = (batch_size,)

    with torch_device_fn.device(input.device):
        jacobi_svd_kernel[grid](
            A,
            A_work,
            V_work,
            U_out,
            S_out,
            V_out,
            # Input A strides
            A.stride(0),
            A.stride(1),
            A.stride(2),
            # A_work strides (column-major)
            A_work.stride(0),
            A_work.stride(1),  # col_stride = m
            # V_work strides (column-major)
            V_work.stride(0),
            V_work.stride(1),  # col_stride = n
            # Output strides
            U_out.stride(0),
            U_out.stride(1),
            U_out.stride(2),
            S_out.stride(0),
            V_out.stride(0),
            V_out.stride(1),
            V_out.stride(2),
            # Dimensions
            M=m,
            N=n,
            K=k,
            out_k_U=out_k_U,
            out_k_V=out_k_V,
            compute_uv=compute_uv,
            NUM_SWEEPS=num_sweeps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    # Sort singular values descending
    sorted_indices = torch.argsort(S_out, dim=-1, descending=True)
    S_out = torch.gather(S_out, -1, sorted_indices)
    if compute_uv:
        idx_U = sorted_indices.unsqueeze(-2).expand_as(U_out)
        U_out = torch.gather(U_out, -1, idx_U)
        idx_V = sorted_indices.unsqueeze(-2).expand_as(V_out)
        V_out = torch.gather(V_out, -1, idx_V)
```

**Step 3: Run tests**

```bash
pytest tests/test_special_ops.py -m svd -v --timeout=300
```

Expected: All tests PASS. Singular values still in descending order.

**Step 4: Commit**

```bash
git add src/flag_gems/ops/svd.py
git commit -m "perf(svd): move sorting outside kernel to wrapper using torch.argsort"
```

---

### Task 4: Add early convergence detection

**Files:**
- Modify: `src/flag_gems/ops/svd.py` (kernel only)

**Step 1: Add convergence tracking to the sweep loop**

Wrap the sweep loop with a convergence check. After each full sweep, check if any rotation was applied. If none were, the matrix has converged:

In the kernel, change the sweep loop to track whether any non-trivial rotation was applied during the sweep. Since Triton doesn't support `break`, we use a `converged_all` flag to skip work in subsequent sweeps:

```python
    # Jacobi sweeps with early convergence
    sweep_converged = tl.full((), 0, dtype=tl.int32)  # 0 = not converged

    for _sweep in range(NUM_SWEEPS):
        # Skip if already globally converged
        max_gamma = tl.full((), 0.0, dtype=tl.float32)

        for p in range(N):
            for q in range(p + 1, N):
                # Load columns ...
                a_p = tl.load(aw_base + p * aw_col_stride + row_idx, mask=row_mask, other=0.0)
                a_q = tl.load(aw_base + q * aw_col_stride + row_idx, mask=row_mask, other=0.0)

                alpha = tl.sum(a_p * a_p)
                beta = tl.sum(a_q * a_q)
                gamma = tl.sum(a_p * a_q)

                abs_gamma = tl.abs(gamma)
                threshold = 1e-7 * tl.sqrt(alpha * beta + 1e-30)
                converged = abs_gamma < threshold

                # Track max off-diagonal for convergence check
                max_gamma = tl.where(abs_gamma > max_gamma, abs_gamma, max_gamma)

                # Only apply rotation if not converged and sweep not done
                should_rotate = ~converged & (sweep_converged == 0)

                safe_gamma = tl.where(converged, tl.full((), 1.0, dtype=tl.float32), gamma)
                zeta = (beta - alpha) / (2.0 * safe_gamma)
                sign_zeta = tl.where(zeta >= 0, tl.full((), 1.0, dtype=tl.float32), tl.full((), -1.0, dtype=tl.float32))
                t = sign_zeta / (tl.abs(zeta) + tl.sqrt(1.0 + zeta * zeta))
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c
                c = tl.where(should_rotate, c, tl.full((), 1.0, dtype=tl.float32))
                s = tl.where(should_rotate, s, tl.full((), 0.0, dtype=tl.float32))

                new_a_p = c * a_p - s * a_q
                new_a_q = s * a_p + c * a_q
                tl.store(aw_base + p * aw_col_stride + row_idx, new_a_p, mask=row_mask)
                tl.store(aw_base + q * aw_col_stride + row_idx, new_a_q, mask=row_mask)

                if compute_uv:
                    v_row_idx2 = tl.arange(0, BLOCK_N)
                    v_mask2 = v_row_idx2 < N
                    v_p = tl.load(vw_base + p * vw_col_stride + v_row_idx2, mask=v_mask2, other=0.0)
                    v_q = tl.load(vw_base + q * vw_col_stride + v_row_idx2, mask=v_mask2, other=0.0)
                    new_v_p = c * v_p - s * v_q
                    new_v_q = s * v_p + c * v_q
                    tl.store(vw_base + p * vw_col_stride + v_row_idx2, new_v_p, mask=v_mask2)
                    tl.store(vw_base + q * vw_col_stride + v_row_idx2, new_v_q, mask=v_mask2)

        # After each sweep, check global convergence
        sweep_converged = tl.where(max_gamma < 1e-7, tl.full((), 1, dtype=tl.int32), sweep_converged)
```

Note: Even though Triton can't `break`, the `should_rotate` flag causes `c=1, s=0` for converged sweeps, making the rotation a no-op (identity). The load/store overhead remains, but the heavy rotation math is skipped.

**Step 2: Reduce default NUM_SWEEPS**

In the wrapper, reduce the sweep cap:

```python
num_sweeps = max(15, n)  # was max(10, n) — 15 is enough with early convergence
```

Actually, since we now detect convergence, we can keep it as-is or even increase it safely. The early exit means extra sweeps cost almost nothing. Keep `max(10, n)`.

**Step 3: Run tests**

```bash
pytest tests/test_special_ops.py -m svd -v --timeout=300
```

Expected: All tests PASS (convergence detection doesn't change output).

**Step 4: Commit**

```bash
git add src/flag_gems/ops/svd.py
git commit -m "perf(svd): add early convergence detection to skip redundant sweeps"
```

---

### Task 5: Run benchmarks and verify performance improvement

**Files:**
- No file changes — this is a verification step

**Step 1: Run SVD benchmark**

```bash
cd benchmark && pytest test_special_perf.py -s -m svd --timeout=600
```

Expected: Performance data showing speedup ratios. Target: >= 0.9x vs PyTorch for all shapes.

**Step 2: Analyze results**

Check the output for each shape:
- (8, 8), (16, 16), (32, 32), (64, 32), (32, 64)
- Batched: (10, 3, 3), (100, 8, 8), (50, 16, 16)

If any shape shows speedup < 0.9x, investigate:
- Is the bottleneck kernel launch overhead? (for very small matrices)
- Is the bottleneck the global memory access pattern? (for larger matrices)
- Is the sort overhead significant? (compare with/without sort)

**Step 3: Commit any tuning adjustments**

If needed, adjust `num_sweeps` or other parameters based on benchmark results.

---

### Task 6: Final cleanup and pre-commit

**Files:**
- Modify: `src/flag_gems/ops/svd.py` (formatting only)

**Step 1: Run pre-commit checks**

```bash
pre-commit run --all-files
```

Expected: All checks pass. If auto-fixed, re-stage and commit.

**Step 2: Run full test suite one more time**

```bash
pytest tests/test_special_ops.py -m svd -v --timeout=300
```

Expected: All PASS.

**Step 3: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style(svd): fix formatting from pre-commit"
```
