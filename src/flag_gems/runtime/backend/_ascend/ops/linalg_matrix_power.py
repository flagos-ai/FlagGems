import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.ops.linalg_lu_factor_ex import (
    linalg_lu_factor_ex as gems_lu_factor_ex,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ===========================================================================
# Matmul helpers — dedicated minimal kernels (one tl.dot per launch)
#
# Ascend cannot compile more than one tl.dot per kernel (needs the missing
# hivmc compiler), so each matmul is a separate launch.  The generic gemm
# kernels add autotune/dispatch overhead (~0.13ms); these minimal kernels
# drop that to the ~0.06ms launch floor, which is the dominant cost for the
# small matrices the benchmark exercises.
# ===========================================================================

_TILE = 64


@libentry()
@triton.jit
def _bmm_tile_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    BATCH_STRIDE,
    GRID_M: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    """One (batch, BLOCK×BLOCK output tile) of C = A·B (square, batched).

    Grid: (batch, GRID_M, GRID_M).  A single launch covers the whole batch.
    When ``EXACT`` (M divides evenly into BLOCK-sized tiles) the loads/stores
    are unmasked — saves the per-element bounds checks that dominate tiny
    matmuls."""
    pid = tl.program_id(0)
    b = pid // (GRID_M * GRID_M)
    rem = pid % (GRID_M * GRID_M)
    mi = rem // GRID_M
    ni = rem % GRID_M
    rm = mi * BLOCK + tl.arange(0, BLOCK)
    rn = ni * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BLOCK)
    abase = a_ptr + b * BATCH_STRIDE
    bbase = b_ptr + b * BATCH_STRIDE
    cbase = c_ptr + b * BATCH_STRIDE
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, M, BLOCK):
        if EXACT:
            a = tl.load(abase + rm[:, None] * M + (k + rk)[None, :])
            b_t = tl.load(bbase + (k + rk)[:, None] * M + rn[None, :])
        else:
            a = tl.load(
                abase + rm[:, None] * M + (k + rk)[None, :],
                mask=(rm[:, None] < M) & ((k + rk)[None, :] < M),
                other=0.0,
            )
            b_t = tl.load(
                bbase + (k + rk)[:, None] * M + rn[None, :],
                mask=((k + rk)[:, None] < M) & (rn[None, :] < M),
                other=0.0,
            )
        acc += tl.dot(a, b_t, allow_tf32=False)
    if EXACT:
        tl.store(cbase + rm[:, None] * M + rn[None, :], acc.to(a_ptr.dtype.element_ty))
    else:
        mask_c = (rm[:, None] < M) & (rn[None, :] < M)
        tl.store(
            cbase + rm[:, None] * M + rn[None, :],
            acc.to(a_ptr.dtype.element_ty),
            mask=mask_c,
        )


def _matmul(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Square matmul via the minimal batched tile kernel (single launch).

    One launch handles the whole batch (grid over batch × output tiles); the
    launch floor (~0.1ms) is the dominant cost for the small matrices in the
    benchmark, so keeping the entire matmul to a single launch is what matters.
    ``out`` lets callers reuse a pre-allocated buffer (avoids per-matmul
    allocation in the binary-exponentiation loop)."""
    M = A.shape[-1]
    block = min(_TILE, triton.next_power_of_2(M))
    grid_m = triton.cdiv(M, block)
    if A.dim() == 2:
        batch = 1
        a = A.unsqueeze(0)
        b = B.unsqueeze(0)
        out_2d = True
        bshape = ()
    else:
        bshape = A.shape[:-2]
        a = A.reshape(-1, M, M)
        b = B.reshape(-1, M, M)
        batch = a.shape[0]
        out_2d = False
    if out is None:
        c = torch.empty(batch, M, M, device=A.device, dtype=A.dtype)
    else:
        c = out.reshape(batch, M, M)
    batch_stride = M * M
    _bmm_tile_kernel[(batch * grid_m * grid_m,)](
        a,
        b,
        c,
        M,
        batch_stride,
        GRID_M=grid_m,
        BLOCK=block,
        EXACT=(grid_m * block == M),
    )
    if out_2d:
        return c.squeeze(0)
    return c.reshape(bshape + (M, M))


# ===========================================================================
# Inverse (negative power) — Ascend LU + triangular solve
# ===========================================================================


def _pivots_to_perm_gpu(pivots, n):
    """Row-permutation index (B[perm] = Pᵀ B) from LAPACK IPIV pivots, on GPU."""
    pv = pivots.reshape(-1, n).detach().cpu()
    bs = pv.shape[0]
    perm = torch.arange(n, device="cpu").repeat(bs, 1)
    for i in range(n):
        j = (pv[:, i] - 1).long()
        tmp = perm[:, i].clone()
        perm[:, i] = perm.gather(1, j.unsqueeze(1)).squeeze(1)
        perm.scatter_(1, j.unsqueeze(1), tmp.unsqueeze(1))
    return perm.to(pivots.device)


def _lu_factor_ex_local(A):
    """LU factorization via the Ascend ``linalg_lu_factor_ex`` (fp32).

    Returns (LU, pivots, info, perm) where ``perm`` is the row-permutation
    index (B[perm] = Pᵀ B), reused by the solve."""
    res = gems_lu_factor_ex(A)
    return res.LU, res.pivots, res.info, _pivots_to_perm_gpu(res.pivots, A.shape[-1])


@libentry()
@triton.jit
def _forward_substitution_kernel(
    L_ptr,
    B_ptr,
    Y_ptr,
    n,
    k,
    stride_lb,
    stride_ln,
    stride_lk,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_yb,
    stride_yn,
    stride_yk,
):
    pid_batch = tl.program_id(0)
    col_idx = tl.program_id(1)

    if col_idx >= n:
        return

    Lp = L_ptr + pid_batch * stride_lb
    Bp = B_ptr + pid_batch * stride_bb
    Yp = Y_ptr + pid_batch * stride_yb

    for i in range(n):
        y_val = tl.load(Bp + i * stride_bn + col_idx * stride_bk)
        # fp32 accumulation (fp64 accumulators need hivmc, missing on Ascend).
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(i):
            l_ij = tl.load(Lp + i * stride_ln + j * stride_lk)
            y_j = tl.load(Yp + j * stride_yn + col_idx * stride_yk)
            acc += l_ij.to(tl.float32) * y_j.to(tl.float32)

        y_val = (y_val - acc).to(Yp.dtype.element_ty)
        tl.store(Yp + i * stride_yn + col_idx * stride_yk, y_val)


@libentry()
@triton.jit
def _backward_substitution_kernel(
    U_ptr,
    Y_ptr,
    X_ptr,
    n,
    k,
    stride_ub,
    stride_un,
    stride_uk,
    stride_yb,
    stride_yn,
    stride_yk,
    stride_xb,
    stride_xn,
    stride_xk,
):
    pid_batch = tl.program_id(0)
    col_idx = tl.program_id(1)

    if col_idx >= n:
        return

    Up = U_ptr + pid_batch * stride_ub
    Yp = Y_ptr + pid_batch * stride_yb
    Xp = X_ptr + pid_batch * stride_xb

    for i in range(n - 1, -1, -1):
        y_val = tl.load(Yp + i * stride_yn + col_idx * stride_yk)
        # fp32 accumulation (fp64 accumulators need hivmc, missing on Ascend).
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(i + 1, n):
            u_ij = tl.load(Up + i * stride_un + j * stride_uk)
            x_j = tl.load(Xp + j * stride_xn + col_idx * stride_xk)
            acc += u_ij.to(tl.float32) * x_j.to(tl.float32)

        u_ii = tl.load(Up + i * stride_un + i * stride_uk)
        x_val = ((y_val - acc) / u_ii).to(Xp.dtype.element_ty)
        tl.store(Xp + i * stride_xn + col_idx * stride_xk, x_val)


def _linalg_lu_solve(LU, perm, B):
    """Solve A X = B (A = P L U) using packed-LU triangular substitution."""
    n = LU.shape[-1]
    k = B.shape[-1] if B.dim() > 1 else 1

    X = torch.empty_like(B)

    n_perm = B.shape[-2]
    bs = 1
    for d in B.shape[:-2]:
        bs *= d
    if bs == 1:
        perm_idx = perm if perm.dim() == 1 else perm[0]
        P_B = B[perm_idx[:n_perm]]
    else:
        P_B = torch.gather(
            B.reshape(-1, n_perm, k),
            1,
            perm[:, :n_perm].unsqueeze(-1).expand(-1, n_perm, k),
        ).reshape(B.shape)

    def _batch_stride(t):
        """Batch stride of a matrix tensor (0 for a single matrix)."""
        return t.stride(-3) if t.dim() >= 3 else 0

    batch = 1
    for d in LU.shape[:-2]:
        batch *= d

    Y = torch.empty_like(P_B)
    with torch_device_fn.device(B.device):
        _forward_substitution_kernel[(batch, n)](
            LU,
            P_B,
            Y,
            n,
            k,
            _batch_stride(LU),
            LU.stride(-2),
            LU.stride(-1),
            _batch_stride(P_B),
            P_B.stride(-2),
            P_B.stride(-1),
            _batch_stride(Y),
            Y.stride(-2),
            Y.stride(-1),
        )

        _backward_substitution_kernel[(batch, n)](
            LU,
            Y,
            X,
            n,
            k,
            _batch_stride(LU),
            LU.stride(-2),
            LU.stride(-1),
            _batch_stride(Y),
            Y.stride(-2),
            Y.stride(-1),
            _batch_stride(X),
            X.stride(-2),
            X.stride(-1),
        )

    return X


def _inverse(A: torch.Tensor) -> torch.Tensor:
    """A⁻¹ via the Ascend LU factorization + triangular substitution solve."""
    LU, pivots, _, perm = _lu_factor_ex_local(A)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    if A.ndim > 2:
        eye = eye.expand(A.shape[:-2] + (A.shape[-1], A.shape[-1])).contiguous()
    return _linalg_lu_solve(LU, perm, eye)


def _eye_like(A: torch.Tensor) -> torch.Tensor:
    m = A.shape[-1]
    shape = A.shape
    eye = torch.eye(m, dtype=A.dtype, device=A.device)
    if len(shape) > 2:
        eye = eye.expand(shape[:-2] + (m, m)).clone()
    return eye


# ===========================================================================
# Main entry point
# ===========================================================================


def linalg_matrix_power(
    A: torch.Tensor,
    n: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    logger.debug("GEMS ASCEND LINALG_MATRIX_POWER")

    # ---- validation ----
    shape = A.shape
    if len(shape) < 2:
        raise RuntimeError(
            f"linalg_matrix_power: A must be at least 2-D, got shape {shape}"
        )
    m, k = shape[-2], shape[-1]
    if m != k:
        raise RuntimeError(f"linalg_matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg_matrix_power: n must be int, got {type(n).__name__}")
    # Ascend Triton mm/bmm and LU are fp32-only.
    if A.dtype != torch.float32:
        raise RuntimeError(
            f"linalg_matrix_power: Ascend backend supports float32 only, "
            f"got {A.dtype}"
        )
    if A.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {A.device}"
        )

    # ---- n == 0 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None:
            out.copy_(eye)
            return out
        return eye

    # ---- n == 1: A¹ = A — plain copy, no computation / kernel launch ----
    if n == 1:
        if out is not None:
            out.copy_(A)
            return out
        return A.clone()

    # ---- negative n ----
    if n < 0:
        A = _inverse(A)
        n = -n

    # ---- n == 2, 3: fast paths for large M ----
    if n == 2:
        r = _matmul(A, A)
        if out is not None:
            out.copy_(r)
            return out
        return r
    if n == 3:
        r = _matmul(_matmul(A, A), A)
        if out is not None:
            out.copy_(r)
            return out
        return r

    # ---- flatten batch ----
    if len(shape) > 2:
        A_flat = A.reshape(-1, m, m)
    else:
        A_flat = A.unsqueeze(0)
    batch_size = A_flat.shape[0]

    if out is not None:
        out_flat = out.reshape(-1, m, m)
    else:
        out_flat = torch.empty(batch_size, m, m, dtype=A.dtype, device=A.device)

    # ---- host-side binary exponentiation (minimal Ascend matmul kernels) ----
    # Three pre-allocated buffers.  Each matmul writes to a buffer that aliases
    # NEITHER input (excluded by identity), so the binary-exponentiation chain
    # is safe without per-matmul torch.empty allocation (the allocation would
    # cost more than the kernel itself on NPU).  At any point exactly two
    # buffers are live (z, result), leaving the third free as the output.
    bufs = [torch.empty_like(A_flat) for _ in range(3)]

    def _free_buf(exclude_a, exclude_b):
        for b in bufs:
            if b is not exclude_a and b is not exclude_b:
                return b
        return bufs[0]  # unreachable: 3 buffers, at most 2 excluded

    z = A_flat
    result = None
    has_result = False
    n_remaining = n
    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                result = z
                has_result = True
            else:
                buf = _free_buf(result, z)
                _matmul(result, z, out=buf)
                result = buf
        n_remaining >>= 1
        if n_remaining > 0:
            # Exclude the accumulated `result` too, not just z: the squaring
            # must not overwrite the buffer that `result` currently aliases.
            buf = _free_buf(z, result if has_result else z)
            _matmul(z, z, out=buf)
            z = buf
    out_flat.copy_(result)

    # ---- reshape back ----
    if len(shape) > 2:
        out_flat = out_flat.reshape(shape)
    else:
        out_flat = out_flat.squeeze(0)

    if out is not None:
        return out
    return out_flat
