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
"""Kunlunxin cholesky_solve backend.

The generic implementation relies on ``tl.gather`` row extraction and 2D
masked loads.  On the XPU triton 3.6 stack those constructs do not work:
``tt.gather`` fails legalization in ``ConvertTritonToTritonXPU``, 2D masked
loads silently ignore their mask, and EQ-predicate selects / ``i1->f32``
casts miscompile at 256 lanes.  This backend keeps the same numerical scheme
(diagonal-pre-scaled serial sweep, one program per (batch, rhs column)) but
uses only constructs verified to compile and execute correctly on XPU:

* 1D masked loads/stores (any length),
* scalar loads for pivots,
* purely arithmetic row-selection factors (``clip(rows - i)``, ``max(1-|df|,0)``),
* 1D reductions for the pivot extraction.

Systems with ``N > 128`` are solved by a two-block decomposition built from
forward / backward sub-solves plus matvec updates, because vector ops wider
than 128 lanes miscompile in this compiler.  The whole computation runs in
device kernels (no eager torch ops on the data path).
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .contiguous import contiguous

logger = logging.getLogger(__name__)

_SINGLE_BLOCK_MAX_N = 128
_SPLIT_HEAD = 128


def _broadcast_shapes(*shapes):
    """Host-side broadcast of shape tuples (equivalent of torch.broadcast_shapes)."""
    rank = max(len(shape) for shape in shapes)
    padded = [(1,) * (rank - len(shape)) + tuple(shape) for shape in shapes]
    out = []
    for dims in zip(*padded):
        dim = 1
        for value in dims:
            if value == 1:
                continue
            if dim == 1:
                dim = value
            elif value != dim:
                raise RuntimeError(
                    "shape mismatch: objects cannot be broadcast to a single shape"
                )
        out.append(dim)
    return tuple(out)


def _check_cholesky_solve_out(B: torch.Tensor, out: torch.Tensor) -> None:
    """Match the device and safe-cast checks of aten::cholesky_solve.out."""
    if out.device != B.device:
        raise RuntimeError(
            "cholesky_solve: Expected result and input tensors to be on the "
            f"same device, but got result on {out.device} and input on {B.device}"
        )
    if not torch.can_cast(B.dtype, out.dtype):
        raise RuntimeError(
            "cholesky_solve: Expected result to be safely castable from "
            f"{B.dtype} dtype, but got result with dtype {out.dtype}"
        )


def _can_write_cholesky_solve_out_direct(
    B: torch.Tensor, L: torch.Tensor, out: torch.Tensor
) -> bool:
    """Whether the solve kernels can safely use ``out`` as their X buffer."""
    if B.layout != torch.strided or L.layout != torch.strided:
        return False
    if B.ndim < 2 or L.ndim < 2:
        return False
    if B.numel() == 0 or L.numel() == 0:
        return False
    if B.shape[:-2] != L.shape[:-2]:
        # Broadcasted solves may produce a result shape different from B.
        return False
    if out.shape != B.shape or out.dtype != B.dtype or out.device != B.device:
        return False
    if not out.is_contiguous() or out.is_conj() or out.is_neg():
        return False
    if torch._C._is_alias_of(out, B) or torch._C._is_alias_of(out, L):
        return False
    return True


def _copy_cholesky_solve_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Resize and copy a temporary solve result into an out tensor."""
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
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
    """Solves a system of linear equations with a positive-definite
    matrix using the Cholesky factorization.

    Args:
        B: right-hand side tensor of shape (*, N, nrhs)
        L: Cholesky factor of shape (*, N, N), lower-triangular unless upper=True

    Returns:
        X: solution tensor of shape (*, N, nrhs)
    """
    logger.debug("GEMS_KUNLUNXIN CHOLESKY_SOLVE")
    if B.is_complex() or L.is_complex():
        raise RuntimeError(
            "cholesky_solve: complex inputs are not supported on this backend"
        )
    assert L.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), "cholesky_solve only supports float32, float64, complex64 and complex128"
    assert B.dtype == L.dtype, "B and L must have the same dtype"
    if B.device != L.device:
        raise ValueError("B and L must be on the same device")
    if B.numel() == 0 or L.numel() == 0:
        return B
    L_shape = L.shape
    B_shape = B.shape
    if len(L_shape) < 2:
        raise ValueError("L must be at least 2D")
    if len(B_shape) < 2:
        raise ValueError("B must be at least 2D")
    N = L_shape[-1]
    if L_shape[-2] != N:
        raise ValueError("L must be a square matrix")
    if B_shape[-2] != N:
        raise ValueError(
            f"B's second-to-last dimension must equal L's last dimension, "
            f"got {B_shape[-2]} != {N}"
        )
    nrhs = B_shape[-1]
    B_batch = B_shape[:-2]
    L_batch = L_shape[:-2]
    if B_batch == L_batch:
        batch_shape = B_batch
    else:
        try:
            batch_shape = _broadcast_shapes(B_batch, L_batch)
        except RuntimeError as exc:
            raise ValueError(
                f"B and L batch dimensions are not broadcastable: "
                f"{B_batch} vs {L_batch}"
            ) from exc
        L = L.expand(batch_shape + L_shape[-2:])
        B = B.expand(batch_shape + B_shape[-2:])
    # Zero-copy layout normalization (same convention as the generic kernel):
    # solving with lower L is solving with upper U = L^T, so a transposed view
    # flips orientation for free.
    if L.is_contiguous():
        effective_upper = upper
    elif L.mT.is_contiguous():
        L = L.mT
        effective_upper = not upper
    else:
        L = contiguous(L)
        effective_upper = upper
    if not B.is_contiguous():
        B = contiguous(B)
    X = torch.empty_like(B) if _out is None else _out
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    if batch_size == 0 or N == 0:
        return X
    with torch.no_grad():
        if N <= _SINGLE_BLOCK_MAX_N:
            _solve_single_block(X, B, L, batch_size, N, nrhs, effective_upper)
        else:
            _solve_two_block(X, B, L, batch_size, N, nrhs, effective_upper)
    return X


def cholesky_solve_out(B, L, upper=False, *, out):
    """Out variant with direct writes for the common compatible case."""
    logger.debug("GEMS_KUNLUNXIN CHOLESKY_SOLVE_OUT")
    _check_cholesky_solve_out(B, out)
    if _can_write_cholesky_solve_out_direct(B, L, out):
        return cholesky_solve(B, L, upper=upper, _out=out)
    result = cholesky_solve(B, L, upper=upper)
    return _copy_cholesky_solve_out(result, out)
