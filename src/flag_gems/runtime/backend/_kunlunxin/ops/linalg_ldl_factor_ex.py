"""Batched Bunch-Kaufman LDL factorization using native Triton kernels.

Initialization copies strided inputs into a padded workspace. The factorization
runs all 1x1/2x2 pivot steps in one launch. Two workspace buffers prevent
read/write overlap during symmetric swaps and trailing-matrix updates. The
real-valued lower triangle is used for both values of ``hermitian``.

Float64 inputs retain the previous implementation's float32 working precision.
The padded output is returned as a view, avoiding an extra device copy.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
MAX_MATRIX_SIZE = 64


@libentry()
@triton.jit
def _ldl_initialize(
    A,
    W,
    N: tl.constexpr,
    B: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BATCH_SHAPE: tl.constexpr,
    BATCH_STRIDES: tl.constexpr,
):
    batch = tl.program_id(0)
    remaining = batch
    aoffset = 0
    for axis in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        aoffset += (remaining % BATCH_SHAPE[axis]) * BATCH_STRIDES[axis]
        remaining = remaining // BATCH_SHAPE[axis]
    A += aoffset
    W += batch * 2 * B * B
    if S1 == 1:
        e = tl.arange(0, B * B)
        row = e // B
        col = e % B
        a = tl.load(A + row * S0 + col, (row < N) & (col < N), other=0.0).to(tl.float32)
        tl.store(W + e, a)
        tl.debug_barrier()
    else:
        col = tl.arange(0, B)
        for row in range(B):
            a = tl.load(A + row * S0 + col * S1, (row < N) & (col < N), other=0.0).to(
                tl.float32
            )
            tl.store(W + row * B + col, a)
            tl.debug_barrier()


@libentry()
@triton.jit
def _ldl_fused_kernel(W, LD, P, INFO, N: tl.constexpr, B: tl.constexpr):
    batch = tl.program_id(0)
    W += batch * 2 * B * B
    U = W + B * B
    LD += batch * B * B
    P += batch * N
    INFO += batch
    e = tl.arange(0, B * B)
    row = e // B
    col = e % B
    rr = tl.arange(0, B)
    src = W
    dst = U
    nextk = 0
    info = 0
    for k in range(N):
        if k >= nextk:
            ck = tl.load(src + rr * B + k, volatile=True)
            diag = tl.sum(tl.where(rr == k, ck, 0.0), 0)
            v = tl.where((rr > k) & (rr < N), tl.abs(ck), 0.0)
            cm = tl.max(v, 0)
            skip = ((diag == 0) & (cm == 0)) | (diag != diag)
            kp = k
            step = 1
            if not skip:
                if tl.abs(diag) < 0.6403882032022076 * cm:
                    im = tl.min(tl.where((rr > k) & (rr < N) & (v == cm), rr, B - 1), 0)
                    rowim = tl.load(
                        src + tl.maximum(im, rr) * B + tl.minimum(im, rr), volatile=True
                    )
                    rm = tl.max(
                        tl.where((rr >= k) & (rr < N) & (rr != im), tl.abs(rowim), 0.0),
                        0,
                    )
                    di = tl.sum(tl.where(rr == im, rowim, 0.0), 0)
                    if tl.abs(diag) < 0.6403882032022076 * cm * (cm / rm):
                        kp = im
                        if tl.abs(di) < 0.6403882032022076 * rm:
                            step = 2
                kk = k + step - 1
                if kp != kk:
                    rp = tl.where(row == kk, kp, tl.where(row == kp, kk, row))
                    cp = tl.where(col == kk, kp, tl.where(col == kp, kk, col))
                    ri = tl.where((row >= k) & (col >= k), tl.maximum(rp, cp), row)
                    ci = tl.where((row >= k) & (col >= k), tl.minimum(rp, cp), col)
                    t = tl.load(src + ri * B + ci, volatile=True)
                    tl.store(dst + e, t)
                    tl.debug_barrier()
                    old = src
                    src = dst
                    dst = old
                t = tl.load(src + e, volatile=True)
                x = tl.load(src + row * B + k, volatile=True)
                xc = tl.load(src + col * B + k, volatile=True)
                d0 = tl.load(src + k * B + k, volatile=True)
                if step == 1:
                    recip = 1.0 / d0
                    t = tl.where((row > k) & (col > k), t - recip * x * xc, t)
                    t = tl.where((row > k) & (col == k), x * recip, t)
                else:
                    k1 = k + 1
                    x1 = tl.load(src + row * B + k1, volatile=True)
                    x1c = tl.load(src + col * B + k1, volatile=True)
                    d1 = tl.load(src + k1 * B + k1, volatile=True)
                    off = tl.load(src + k1 * B + k, volatile=True)
                    d11 = d1 / off
                    d22 = d0 / off
                    mult = (1.0 / (d11 * d22 - 1.0)) / off
                    w0 = mult * (d11 * x - x1)
                    w1 = mult * (d22 * x1 - x)
                    w0c = mult * (d11 * xc - x1c)
                    w1c = mult * (d22 * x1c - xc)
                    t = tl.where((row > k1) & (col > k1), t - x * w0c - x1 * w1c, t)
                    t = tl.where((row > k1) & (col == k), w0, t)
                    t = tl.where((row > k1) & (col == k1), w1, t)
                tl.store(dst + e, t)
                tl.debug_barrier()
                old = src
                src = dst
                dst = old
            else:
                info = tl.where(info == 0, k + 1, info)
            pv = tl.where(step == 1, kp + 1, -(kp + 1))
            tl.store(P + k, pv)
            if step == 2:
                tl.store(P + k + 1, pv)
            nextk = k + step
    t = tl.load(src + e, volatile=True)
    tl.store(LD + e, tl.where((row >= col) & (row < N) & (col < N), t, 0.0))
    tl.store(INFO, info)


@libentry()
@triton.jit
def _ldl_zero_info(INFO, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(INFO + i, 0, i < COUNT)


def _check_linalg_ldl_factor_ex(A, hermitian, check_errors):
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor_ex: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor_ex: matrix must be square")
    if not isinstance(hermitian, bool):
        raise TypeError(f"hermitian must be a bool, got {type(hermitian)}")
    if not isinstance(check_errors, bool):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "Kunlunxin linalg_ldl_factor_ex supports float32 and float64 only"
        )
    if A.shape[-1] > MAX_MATRIX_SIZE:
        raise ValueError(
            f"linalg_ldl_factor_ex: matrix size {A.shape[-1]} exceeds maximum {MAX_MATRIX_SIZE}"
        )


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    _check_linalg_ldl_factor_ex(A, hermitian, check_errors)
    n = A.shape[-1]
    batch_shape = tuple(A.shape[:-2])
    batch_count = math.prod(batch_shape)
    pivots = torch.empty(A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    if n == 0 or batch_count == 0:
        ld = torch.empty_like(A)
        if batch_count:
            with torch_device_fn.device(A.device):
                _ldl_zero_info[(triton.cdiv(batch_count, 128),)](
                    info, batch_count, 128, num_warps=1
                )
        return ld, pivots, info
    b = max(32, triton.next_power_of_2(n))
    workspace = torch.empty(
        (batch_count, 2 * b * b), dtype=torch.float32, device=A.device
    )
    ld_storage = torch.empty((*batch_shape, b, b), dtype=A.dtype, device=A.device)
    with torch_device_fn.device(A.device):
        _ldl_initialize[(batch_count,)](
            A,
            workspace,
            n,
            b,
            A.stride(-2),
            A.stride(-1),
            batch_shape,
            tuple(A.stride()[:-2]),
            num_warps=1,
        )
        _ldl_fused_kernel[(batch_count,)](
            workspace,
            ld_storage,
            pivots,
            info,
            n,
            b,
            num_warps=1,
            enable_fp_fusion=False,
        )
    if check_errors:
        for code in info.reshape(-1).tolist():
            if code:
                raise RuntimeError(
                    f"torch.linalg.ldl_factor_ex: factorization failed because "
                    f"diagonal element {code} is zero or non-finite"
                )
    return ld_storage[..., :n, :n], pivots, info
