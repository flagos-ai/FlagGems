# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import itertools

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

_SINGLE_BLOCK_MAX_N = 128
_SPLIT_HEAD = 128

_generic = importlib.import_module("flag_gems.ops.cholesky_solve")


@triton.jit
def _cholesky_solve_row_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs,
    batch_stride_L,
    batch_stride_B,
    batch_stride_X,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    stride_X_row,
    stride_X_col,
    ROW: tl.constexpr,
    FORWARD: tl.constexpr,
    upper: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    rhs_pid = tl.program_id(1)
    cols = rhs_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    X_base = batch_pid * batch_stride_X

    if FORWARD:
        value = tl.load(
            B_ptr + B_base + ROW * stride_B_row + cols * stride_B_col,
            mask=cols_mask,
            other=0.0,
        )
        for col in range(ROW):
            if upper:
                factor = tl.load(
                    L_ptr + L_base + col * stride_L_row + ROW * stride_L_col
                )
            else:
                factor = tl.load(
                    L_ptr + L_base + ROW * stride_L_row + col * stride_L_col
                )
            previous = tl.load(
                X_ptr + X_base + col * stride_X_row + cols * stride_X_col,
                mask=cols_mask,
                other=0.0,
            )
            value -= factor * previous
    else:
        value = tl.load(
            X_ptr + X_base + ROW * stride_X_row + cols * stride_X_col,
            mask=cols_mask,
            other=0.0,
        )
        for col in range(ROW + 1, N):
            if upper:
                factor = tl.load(
                    L_ptr + L_base + ROW * stride_L_row + col * stride_L_col
                )
            else:
                factor = tl.load(
                    L_ptr + L_base + col * stride_L_row + ROW * stride_L_col
                )
            previous = tl.load(
                X_ptr + X_base + col * stride_X_row + cols * stride_X_col,
                mask=cols_mask,
                other=0.0,
            )
            value -= factor * previous

    diagonal = tl.load(L_ptr + L_base + ROW * stride_L_row + ROW * stride_L_col)
    tl.store(
        X_ptr + X_base + ROW * stride_X_row + cols * stride_X_col,
        value / diagonal,
        mask=cols_mask,
    )


@triton.jit
def _cholesky_solve_complex_row_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs,
    batch_stride_L,
    batch_stride_B,
    batch_stride_X,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    stride_X_row,
    stride_X_col,
    ROW: tl.constexpr,
    FORWARD: tl.constexpr,
    upper: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    batch_pid = tl.program_id(0)
    rhs_pid = tl.program_id(1)
    cols = rhs_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    X_base = batch_pid * batch_stride_X

    if FORWARD:
        b_offset = B_base + ROW * stride_B_row + cols * stride_B_col
        value_real = tl.load(B_ptr + b_offset, mask=cols_mask, other=0.0)
        value_imag = tl.load(B_ptr + b_offset + 1, mask=cols_mask, other=0.0)
        start = 0
        end = ROW
    else:
        x_offset = X_base + ROW * stride_X_row + cols * stride_X_col
        value_real = tl.load(X_ptr + x_offset, mask=cols_mask, other=0.0)
        value_imag = tl.load(X_ptr + x_offset + 1, mask=cols_mask, other=0.0)
        start = ROW + 1
        end = N

    for col in range(start, end):
        if FORWARD:
            if upper:
                factor_offset = L_base + col * stride_L_row + ROW * stride_L_col
            else:
                factor_offset = L_base + ROW * stride_L_row + col * stride_L_col
        else:
            if upper:
                factor_offset = L_base + ROW * stride_L_row + col * stride_L_col
            else:
                factor_offset = L_base + col * stride_L_row + ROW * stride_L_col
        factor_real = tl.load(L_ptr + factor_offset)
        factor_imag = tl.load(L_ptr + factor_offset + 1)
        if (FORWARD and upper) or ((not FORWARD) and (not upper)):
            factor_imag = -factor_imag
        previous_offset = X_base + col * stride_X_row + cols * stride_X_col
        previous_real = tl.load(X_ptr + previous_offset, mask=cols_mask, other=0.0)
        previous_imag = tl.load(X_ptr + previous_offset + 1, mask=cols_mask, other=0.0)
        value_real -= factor_real * previous_real - factor_imag * previous_imag
        value_imag -= factor_real * previous_imag + factor_imag * previous_real

    diagonal_offset = L_base + ROW * stride_L_row + ROW * stride_L_col
    diagonal_real = tl.load(L_ptr + diagonal_offset)
    diagonal_imag = tl.load(L_ptr + diagonal_offset + 1)
    denominator = diagonal_real * diagonal_real + diagonal_imag * diagonal_imag
    out_real = (value_real * diagonal_real + value_imag * diagonal_imag) / denominator
    out_imag = (value_imag * diagonal_real - value_real * diagonal_imag) / denominator
    out_offset = X_base + ROW * stride_X_row + cols * stride_X_col
    tl.store(X_ptr + out_offset, out_real, mask=cols_mask)
    tl.store(X_ptr + out_offset + 1, out_imag, mask=cols_mask)


@triton.jit
def _cholesky_solve_serial_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    N: tl.constexpr,
    nrhs,
    batch_stride_L,
    batch_stride_B,
    batch_stride_X,
    stride_L_row,
    stride_L_col,
    stride_B_row,
    stride_B_col,
    stride_X_row,
    stride_X_col,
    upper: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Fully serial Cholesky solve for one batch and one RHS tile.

    One program owns one (batch, RHS-tile) pair and walks the rows in
    order, keeping each partial row vector in registers; prior rows are
    re-loaded from global memory. There is no tl.dot (a dot drags the whole
    kernel into the SDNN pipeline and fails to lower here) and no tl.gather
    (tt.gather is explicitly illegal on this backend), so every construct is
    scalar/vector load-store arithmetic the TritonXPU backend can lower.
    """
    batch_pid = tl.program_id(0)
    rhs_pid = tl.program_id(1)
    cols = rhs_pid * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    cols_mask = cols < nrhs

    L_base = batch_pid * batch_stride_L
    B_base = batch_pid * batch_stride_B
    X_base = batch_pid * batch_stride_X

    # Forward solve: L * Y = B. Row r only touches rows < r, which every
    # program already stored, so the serial walk needs no barrier.
    for row in range(N):
        value = tl.load(
            B_ptr + B_base + row * stride_B_row + cols * stride_B_col,
            mask=cols_mask,
            other=0.0,
        )
        for col in range(row):
            if upper:
                factor = tl.load(
                    L_ptr + L_base + col * stride_L_row + row * stride_L_col
                )
            else:
                factor = tl.load(
                    L_ptr + L_base + row * stride_L_row + col * stride_L_col
                )
            previous = tl.load(
                X_ptr + X_base + col * stride_X_row + cols * stride_X_col,
                mask=cols_mask,
                other=0.0,
            )
            value -= factor * previous
        diagonal = tl.load(L_ptr + L_base + row * stride_L_row + row * stride_L_col)
        tl.store(
            X_ptr + X_base + row * stride_X_row + cols * stride_X_col,
            value / diagonal,
            mask=cols_mask,
        )

    # Backward solve: L^H * X = Y (upper: U). Row r uses rows > r.
    for row in range(N - 1, -1, -1):
        value = tl.load(
            X_ptr + X_base + row * stride_X_row + cols * stride_X_col,
            mask=cols_mask,
            other=0.0,
        )
        for col in range(row + 1, N):
            if upper:
                factor = tl.load(
                    L_ptr + L_base + row * stride_L_row + col * stride_L_col
                )
            else:
                factor = tl.load(
                    L_ptr + L_base + col * stride_L_row + row * stride_L_col
                )
            previous = tl.load(
                X_ptr + X_base + col * stride_X_row + cols * stride_X_col,
                mask=cols_mask,
                other=0.0,
            )
            value -= factor * previous
        diagonal = tl.load(L_ptr + L_base + row * stride_L_row + row * stride_L_col)
        tl.store(
            X_ptr + X_base + row * stride_X_row + cols * stride_X_col,
            value / diagonal,
            mask=cols_mask,
        )


def _can_use_row_kernel(B: torch.Tensor, L: torch.Tensor) -> bool:
    if B.dtype != torch.float32 or L.dtype != torch.float32 or B.ndim < 2 or L.ndim < 2:
        return False
    if B.shape[:-2] != L.shape[:-2] or L.shape[-2] != L.shape[-1]:
        return False
    n, nrhs = B.shape[-2:]
    return n == L.shape[-1] and ((n <= 32 and nrhs <= 16) or (n < 64 and nrhs == 1))


def _can_use_complex_row_kernel(B: torch.Tensor, L: torch.Tensor) -> bool:
    if B.dtype != torch.complex64 or L.dtype != torch.complex64:
        return False
    if B.ndim < 2 or L.ndim < 2 or B.is_conj() or L.is_conj():
        return False
    if B.shape[:-2] != L.shape[:-2] or L.shape[-2] != L.shape[-1]:
        return False
    n, nrhs = B.shape[-2:]
    return n == L.shape[-1] and ((n <= 32 and nrhs <= 16) or (n < 64 and nrhs == 1))


def _cholesky_solve_complex_rows(B, L, upper, out):
    B_real = torch.view_as_real(B).reshape(-1, B.shape[-2], B.shape[-1], 2)
    L_real = torch.view_as_real(L).reshape(-1, L.shape[-2], L.shape[-1], 2)
    X_real = torch.view_as_real(out).reshape(-1, out.shape[-2], out.shape[-1], 2)
    batch_size = B_real.shape[0]
    n, nrhs = B.shape[-2:]
    block_rhs = triton.next_power_of_2(nrhs)
    grid = (batch_size, triton.cdiv(nrhs, block_rhs))

    for row in range(n):
        _cholesky_solve_complex_row_kernel[grid](
            L_real,
            B_real,
            X_real,
            n,
            nrhs,
            L_real.stride(0) if L_real.ndim > 3 else 0,
            B_real.stride(0) if B_real.ndim > 3 else 0,
            X_real.stride(0) if X_real.ndim > 3 else 0,
            L_real.stride(-3),
            L_real.stride(-2),
            B_real.stride(-3),
            B_real.stride(-2),
            X_real.stride(-3),
            X_real.stride(-2),
            ROW=row,
            FORWARD=True,
            upper=upper,
            BLOCK_RHS=block_rhs,
            num_warps=1,
            num_stages=1,
        )
    for row in range(n - 1, -1, -1):
        _cholesky_solve_complex_row_kernel[grid](
            L_real,
            B_real,
            X_real,
            n,
            nrhs,
            L_real.stride(0) if L_real.ndim > 3 else 0,
            B_real.stride(0) if B_real.ndim > 3 else 0,
            X_real.stride(0) if X_real.ndim > 3 else 0,
            L_real.stride(-3),
            L_real.stride(-2),
            B_real.stride(-3),
            B_real.stride(-2),
            X_real.stride(-3),
            X_real.stride(-2),
            ROW=row,
            FORWARD=False,
            upper=upper,
            BLOCK_RHS=block_rhs,
            num_warps=1,
            num_stages=1,
        )
    return out


@libentry()
@triton.jit
def cholesky_solve_column_kernel(
    L_ptr,
    B_ptr,
    X_ptr,
    bL,
    bB,
    bX,
    sL,
    sB,
    sX,
    N,
    nrhs,
    BN: tl.constexpr,
    upper: tl.constexpr,
):
    """Whole-system register-resident sweep for N <= 128, one program per
    (batch, rhs column).  Solves L y = b then L^T x = y (or the upper-storage
    counterparts) with a diagonal-pre-scaled serial sweep."""
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, BN)
    m = rows < N
    Lp = L_ptr + batch * bL
    Bp = B_ptr + batch * bB + col
    Xp = X_ptr + batch * bX + col
    b = tl.load(Bp + rows * sB, mask=m, other=0.0)
    diag = tl.load(Lp + rows * sL + rows, mask=m, other=1.0)
    inv = 1.0 / diag
    inv = inv * (2.0 - diag * inv)
    w = b * inv
    for i in range(N):
        if upper:
            colv = tl.load(Lp + i * sL + rows, mask=m, other=0.0)
        else:
            colv = tl.load(Lp + rows * sL + i, mask=m, other=0.0)
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        fac = tl.maximum(tl.minimum(df, 1.0), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        w = w - fac * (colv * inv) * w_i
    w = w * inv
    for i in range(N - 1, -1, -1):
        if upper:
            colv = tl.load(Lp + rows * sL + i, mask=m, other=0.0)
        else:
            colv = tl.load(Lp + i * sL + rows, mask=m, other=0.0)
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        fac2 = tl.maximum(tl.minimum(-df, 1.0), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        w = w - fac2 * (colv * inv) * w_i
    tl.store(Xp + rows * sX, w, mask=m)


@libentry()
@triton.jit
def cholesky_solve_fwd_kernel(
    L_ptr,
    IN_ptr,
    OUT_ptr,
    off_L,
    off_IN,
    off_OUT,
    bL,
    bIN,
    bOUT,
    sL,
    sIN,
    sOUT,
    Nb,
    nrhs,
    BN: tl.constexpr,
    upper: tl.constexpr,
):
    """Forward sub-solve of one diagonal block: OUT <- solve(L_bb, IN).

    lower storage: solves L_bb y = b for the block.
    upper storage: solves U_bb^T y = b for the block.
    """
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, BN)
    m = rows < Nb
    Lp = L_ptr + off_L + batch * bL
    Ip = IN_ptr + off_IN + batch * bIN + col
    Op = OUT_ptr + off_OUT + batch * bOUT + col
    diag = tl.load(Lp + rows * sL + rows, mask=m, other=1.0)
    inv = 1.0 / diag
    inv = inv * (2.0 - diag * inv)
    w = tl.load(Ip + rows * sIN, mask=m, other=0.0) * inv
    for i in range(Nb):
        if upper:
            colv = tl.load(Lp + i * sL + rows, mask=m, other=0.0)
        else:
            colv = tl.load(Lp + rows * sL + i, mask=m, other=0.0)
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        fac = tl.maximum(tl.minimum(df, 1.0), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        w = w - fac * (colv * inv) * w_i
    tl.store(Op + rows * sOUT, w, mask=m)


@libentry()
@triton.jit
def cholesky_solve_bwd_kernel(
    L_ptr,
    IN_ptr,
    OUT_ptr,
    off_L,
    off_IN,
    off_OUT,
    bL,
    bIN,
    bOUT,
    sL,
    sIN,
    sOUT,
    Nb,
    nrhs,
    BN: tl.constexpr,
    upper: tl.constexpr,
):
    """Backward sub-solve of one diagonal block: OUT <- solve(L_bb^T, IN).

    lower storage: solves L_bb^T x = y for the block.
    upper storage: solves U_bb x = y for the block.
    """
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, BN)
    m = rows < Nb
    Lp = L_ptr + off_L + batch * bL
    Ip = IN_ptr + off_IN + batch * bIN + col
    Op = OUT_ptr + off_OUT + batch * bOUT + col
    diag = tl.load(Lp + rows * sL + rows, mask=m, other=1.0)
    inv = 1.0 / diag
    inv = inv * (2.0 - diag * inv)
    w = tl.load(Ip + rows * sIN, mask=m, other=0.0) * inv
    for i in range(Nb - 1, -1, -1):
        if upper:
            colv = tl.load(Lp + rows * sL + i, mask=m, other=0.0)
        else:
            colv = tl.load(Lp + i * sL + rows, mask=m, other=0.0)
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        fac2 = tl.maximum(tl.minimum(-df, 1.0), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        w = w - fac2 * (colv * inv) * w_i
    tl.store(Op + rows * sOUT, w, mask=m)


@libentry()
@triton.jit
def _cholesky_matvec_left_kernel(
    A_ptr,
    Y_ptr,
    ZIN_ptr,
    ZOUT_ptr,
    offA,
    offY,
    offZI,
    offZO,
    bA,
    bY,
    bZI,
    bZO,
    sA,
    sY,
    sZI,
    sZO,
    M,
    K,
    nrhs,
    MBN: tl.constexpr,
    KBN: tl.constexpr,
):
    """ZOUT[:, c] <- ZIN[:, c] - A @ Y[:, c] for every rhs column c.

    Fused because a sum-derived value stores correctly even with a strided
    store; only dot-derived values need the scratch + apply split."""
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, MBN)
    cc = tl.arange(0, KBN)
    m = rows < M
    Ap = A_ptr + offA + batch * bA
    Yp = Y_ptr + offY + batch * bY + col
    ZIp = ZIN_ptr + offZI + batch * bZI + col
    ZOp = ZOUT_ptr + offZO + batch * bZO + col
    t = tl.load(ZIp + rows * sZI, mask=m, other=0.0)
    At = tl.load(Ap + rows[:, None] * sA + cc[None, :])
    yv = tl.load(Yp + cc * sY)
    t = t - tl.sum(At * yv[None, :], axis=1)
    tl.store(ZOp + rows * sZO, t, mask=m)


@libentry()
@triton.jit
def _cholesky_dot_right_kernel(
    A_ptr,
    Y_ptr,
    UPD_ptr,
    offA,
    offY,
    offU,
    bA,
    bY,
    sA,
    sY,
    sU,
    M,
    K,
    nrhs,
    KBN: tl.constexpr,
    MBN: tl.constexpr,
):
    """UPD[:, c] <- A^T @ Y[:, c] for every rhs column c (dot-only, see left)."""
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, KBN)
    mm = tl.arange(0, MBN)
    Ap = A_ptr + offA + batch * bA
    Yp = Y_ptr + offY + batch * bY + col
    Up = UPD_ptr + offU + col * sU
    At = tl.load(Ap + mm[:, None] * sA + rows[None, :])
    yv = tl.load(Yp + mm * sY)
    upd = tl.dot(yv[None, :], At, input_precision="ieee")
    tl.store(Up + rows, tl.reshape(upd, [KBN]))


@libentry()
@triton.jit
def _cholesky_apply_sub_kernel(
    ZIN_ptr,
    ZOUT_ptr,
    UPD_ptr,
    offZI,
    offZO,
    offU,
    bZI,
    bZO,
    sZI,
    sZO,
    sU,
    K,
    nrhs,
    KBN: tl.constexpr,
):
    """ZOUT[:, c] <- ZIN[:, c] - UPD[:, c] for every rhs column c."""
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, KBN)
    ZIp = ZIN_ptr + offZI + batch * bZI + col
    ZOp = ZOUT_ptr + offZO + batch * bZO + col
    Up = UPD_ptr + offU + col * sU
    t = tl.load(ZIp + rows * sZI)
    u = tl.load(Up + rows)
    tl.store(ZOp + rows * sZO, t - u)


def _solve_single_block(X, B, L, batch_size, N, nrhs, upper):
    """N <= 128: one combined kernel launch."""
    BN = triton.next_power_of_2(N)
    grid = (batch_size * nrhs,)
    Lk = L.reshape(-1, N, N)
    Bk = B.reshape(-1, N, nrhs)
    Xk = X.reshape(-1, N, nrhs)
    cholesky_solve_column_kernel[grid](
        Lk,
        Bk,
        Xk,
        Lk.stride(0),
        Bk.stride(0),
        Xk.stride(0),
        Lk.stride(1),
        Bk.stride(1),
        Xk.stride(1),
        N,
        nrhs,
        BN=BN,
        upper=upper,
    )


def _solve_two_block(X, B, L, batch_size, N, nrhs, upper):
    """N == 256: two-block decomposition with sub-solves plus matvec updates.

    Runs entirely in device kernels; X carries the working data and the
    result.  The right-transpose matvec is split into a dot kernel writing a
    contiguous scratch plus an apply kernel (``Z -= scratch``) because a
    strided store of a dot-derived value miscompiles in this backend."""
    h1 = _SPLIT_HEAD
    h2 = N - h1
    if h2 != h1:
        raise RuntimeError(
            "cholesky_solve: N > 128 is only supported for N == 256 on this backend"
        )
    BN1 = triton.next_power_of_2(h1)
    BN2 = triton.next_power_of_2(h2)
    grid = (batch_size * nrhs,)
    Xk = X.reshape(-1, N, nrhs)
    Bk = B.reshape(-1, N, nrhs)
    Lk = L.reshape(-1, N, N)
    bX = Xk.stride(0)
    sX = Xk.stride(1)
    bL = Lk.stride(0)
    sL = Lk.stride(1)
    bB = Bk.stride(0)
    sB = Bk.stride(1)
    scratch = torch.empty((nrhs, h1), dtype=X.dtype, device=X.device)
    sU = scratch.stride(0)
    off_x1 = 0
    off_x2 = h1 * sX
    off_l11 = 0
    off_l22 = h1 * sL + h1

    def _fwd(off_l, use_x, off_o, ni, up):
        cholesky_solve_fwd_kernel[grid](
            Lk,
            Xk if use_x else Bk,
            Xk,
            off_l,
            off_o if use_x else 0,
            off_o,
            bL,
            bX if use_x else bB,
            bX,
            sL,
            sX if use_x else sB,
            sX,
            ni,
            nrhs,
            BN=(BN1 if ni == h1 else BN2),
            upper=up,
        )

    def _bwd(off_l, off_io, ni, up):
        cholesky_solve_bwd_kernel[grid](
            Lk,
            Xk,
            Xk,
            off_l,
            off_io,
            off_io,
            bL,
            bX,
            bX,
            sL,
            sX,
            sX,
            ni,
            nrhs,
            BN=(BN1 if ni == h1 else BN2),
            upper=up,
        )

    def _mv_left(off_a, off_yv, from_b, off_zi, off_zo, m, k):
        zi_ptr = Bk if from_b else Xk
        zi_off = h1 * sB if from_b else off_zi
        bzi = bB if from_b else bX
        szi = sB if from_b else sX
        _cholesky_matvec_left_kernel[grid](
            Lk,
            Xk,
            zi_ptr,
            Xk,
            off_a,
            off_yv,
            zi_off,
            off_zo,
            bL,
            bX,
            bzi,
            bX,
            sL,
            sX,
            szi,
            sX,
            m,
            k,
            nrhs,
            MBN=(BN1 if m == h1 else BN2),
            KBN=(BN1 if k == h1 else BN2),
        )

    def _dot_right_apply(off_a, off_yv, from_b, off_zt, m, k):
        _cholesky_dot_right_kernel[grid](
            Lk,
            Xk,
            scratch,
            off_a,
            off_yv,
            0,
            bL,
            bX,
            sL,
            sX,
            sU,
            m,
            k,
            nrhs,
            KBN=(BN1 if k == h1 else BN2),
            MBN=(BN1 if m == h1 else BN2),
        )
        zi_ptr = Bk if from_b else Xk
        zi_off = h1 * sB if from_b else off_zt
        bzi = bB if from_b else bX
        szi = sB if from_b else sX
        _cholesky_apply_sub_kernel[grid](
            zi_ptr,
            Xk,
            scratch,
            zi_off,
            off_zt,
            0,
            bzi,
            bX,
            szi,
            sX,
            sU,
            k,
            nrhs,
            KBN=(BN1 if k == h1 else BN2),
        )

    if not upper:
        off_cross = h1 * sL
        _fwd(off_l11, False, off_x1, h1, 0)
        _mv_left(off_cross, off_x1, True, 0, off_x2, h2, h1)
        _fwd(off_l22, True, off_x2, h2, 0)
        _bwd(off_l22, off_x2, h2, 0)
        _dot_right_apply(off_cross, off_x2, False, off_x1, h2, h1)
        _bwd(off_l11, off_x1, h1, 0)
    else:
        off_cross = h1
        _fwd(off_l11, False, off_x1, h1, 1)
        _dot_right_apply(off_cross, off_x1, True, off_x2, h1, h2)
        _fwd(off_l22, True, off_x2, h2, 1)
        _bwd(off_l22, off_x2, h2, 1)
        _mv_left(off_cross, off_x2, False, off_x1, off_x1, h1, h2)
        _bwd(off_l11, off_x1, h1, 1)


def cholesky_solve(B, L, upper=False, *, _out=None):
    if B.numel() == 0 or L.numel() == 0:
        if _out is not None:
            return _generic._copy_cholesky_solve_out(B, _out)
        return B
    assert B.dtype == L.dtype, "B and L must have the same dtype"
    if B.device != L.device:
        raise ValueError("B and L must be on the same device")
    if len(L.shape) < 2:
        raise ValueError("L must be at least 2D")
    if len(B.shape) < 2:
        raise ValueError("B must be at least 2D")
    if L.shape[-2] != L.shape[-1]:
        raise ValueError("L must be a square matrix")
    if B.shape[-2] != L.shape[-1]:
        raise ValueError(
            "B's second-to-last dimension must equal L's last dimension, "
            f"got {B.shape[-2]} != {L.shape[-1]}"
        )
    try:
        batch_shape = torch.broadcast_shapes(B.shape[:-2], L.shape[:-2])
    except RuntimeError:
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)

    result_shape = batch_shape + B.shape[-2:]
    if B.shape[:-2] != batch_shape or L.shape[:-2] != batch_shape:
        B_expanded = B.expand(result_shape)
        L_expanded = L.expand(batch_shape + L.shape[-2:])
        X = torch.empty(result_shape, dtype=B.dtype, device=B.device)
        for index in itertools.product(*(range(dim) for dim in batch_shape)):
            cholesky_solve(
                B_expanded[index], L_expanded[index], upper=upper, _out=X[index]
            )
        if _out is None:
            return X
        return _generic._copy_cholesky_solve_out(X, _out)

    X = torch.empty_like(B) if _out is None else _out
    if _out is not None and (
        X.shape != B.shape
        or X.dtype != B.dtype
        or X.device != B.device
        or torch._C._is_alias_of(X, B)
        or torch._C._is_alias_of(X, L)
    ):
        result = cholesky_solve(B, L, upper=upper)
        return _generic._copy_cholesky_solve_out(result, X)

    if B.dtype == torch.complex64:
        if _can_use_complex_row_kernel(B, L):
            if X.shape == B.shape and X.dtype == B.dtype and X.device == B.device:
                return _cholesky_solve_complex_rows(B, L, upper, X)
            return _generic.cholesky_solve(B, L, upper=upper, _out=_out)
        # No register-gather kernel is available for large complex systems on
        # this backend (tt.gather is illegal); use the serial row kernels for
        # any complex64 size so the op stays functional.
        if X.shape == B.shape and X.dtype == B.dtype and X.device == B.device:
            return _cholesky_solve_complex_rows(B, L, upper, X)
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)

    if X.shape != B.shape or X.dtype != B.dtype or X.device != B.device:
        return _generic.cholesky_solve(B, L, upper=upper, _out=_out)

    n, nrhs = B.shape[-2:]
    output = X

    # Zero-copy layout normalization mirroring the generic dispatch: a
    # transposed view flips the factor orientation for a lower solve.
    if L.is_contiguous():
        effective_upper = upper
        L_kernel = L
    elif L.mT.is_contiguous():
        L_kernel = L.mT
        effective_upper = not upper
    else:
        L_kernel = L.contiguous()
        effective_upper = upper
    if not B.is_contiguous():
        B_kernel = B.contiguous()
    else:
        B_kernel = B

    L_kernel = L_kernel.reshape(-1, n, n)
    B_kernel = B_kernel.reshape(-1, n, nrhs)
    X_kernel = X.reshape(-1, n, nrhs)
    batch_size = B_kernel.shape[0]

    # Peer-derived Triton-only FP32 solver. Preserve the local wrapper and
    # all unsupported/complex paths. The two-block scratch is not batch-indexed,
    # so only unbatched N=256 is eligible; other sizes retain the local path.
    if B.dtype == torch.float32 and X.is_contiguous():
        if n <= _SINGLE_BLOCK_MAX_N:
            _solve_single_block(
                X_kernel, B_kernel, L_kernel, batch_size, n, nrhs, effective_upper
            )
            return output
        if n == 256 and batch_size == 1:
            _solve_two_block(
                X_kernel, B_kernel, L_kernel, batch_size, n, nrhs, effective_upper
            )
            return output

    if _can_use_row_kernel(B, L):
        block_rhs = triton.next_power_of_2(nrhs)
        grid = (batch_size, triton.cdiv(nrhs, block_rhs))
        for row in range(n):
            _cholesky_solve_row_kernel[grid](
                L_kernel,
                B_kernel,
                X_kernel,
                n,
                nrhs,
                L_kernel.stride(0) if L_kernel.ndim > 2 else 0,
                B_kernel.stride(0) if B_kernel.ndim > 2 else 0,
                X_kernel.stride(0) if X_kernel.ndim > 2 else 0,
                L_kernel.stride(-2),
                L_kernel.stride(-1),
                B_kernel.stride(-2),
                B_kernel.stride(-1),
                X_kernel.stride(-2),
                X_kernel.stride(-1),
                ROW=row,
                FORWARD=True,
                upper=effective_upper,
                BLOCK_RHS=block_rhs,
                num_warps=1,
                num_stages=1,
            )
        for row in range(n - 1, -1, -1):
            _cholesky_solve_row_kernel[grid](
                L_kernel,
                B_kernel,
                X_kernel,
                n,
                nrhs,
                L_kernel.stride(0) if L_kernel.ndim > 2 else 0,
                B_kernel.stride(0) if B_kernel.ndim > 2 else 0,
                X_kernel.stride(0) if X_kernel.ndim > 2 else 0,
                L_kernel.stride(-2),
                L_kernel.stride(-1),
                B_kernel.stride(-2),
                B_kernel.stride(-1),
                X_kernel.stride(-2),
                X_kernel.stride(-1),
                ROW=row,
                FORWARD=False,
                upper=effective_upper,
                BLOCK_RHS=block_rhs,
                num_warps=1,
                num_stages=1,
            )
        return output

    block_rhs = max(triton.next_power_of_2(nrhs), 16)
    if block_rhs > 64:
        block_rhs = 64
    grid = (batch_size, triton.cdiv(nrhs, block_rhs))
    _cholesky_solve_serial_kernel[grid](
        L_kernel,
        B_kernel,
        X_kernel,
        n,
        nrhs,
        L_kernel.stride(0) if L_kernel.ndim > 2 else 0,
        B_kernel.stride(0) if B_kernel.ndim > 2 else 0,
        X_kernel.stride(0) if X_kernel.ndim > 2 else 0,
        L_kernel.stride(-2),
        L_kernel.stride(-1),
        B_kernel.stride(-2),
        B_kernel.stride(-1),
        X_kernel.stride(-2),
        X_kernel.stride(-1),
        upper=effective_upper,
        BLOCK_RHS=block_rhs,
        num_warps=1,
        num_stages=1,
    )
    return output


def cholesky_solve_out(B, L, upper=False, *, out):
    _generic._check_cholesky_solve_out(B, out)
    return cholesky_solve(B, L, upper=upper, _out=out)
