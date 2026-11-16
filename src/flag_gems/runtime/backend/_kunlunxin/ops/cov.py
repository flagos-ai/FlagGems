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
"""Kunlunxin XPU3 override for ``aten::cov``.

Two measured XPU3-specific defects of the generic implementation forced this
override (reproduced in ``harness/perf_ir/probe_cov_*.py``):

1. ``input_precision="tf32x3"`` is not an allowed dot precision on this backend
   (``allowed_dot_input_precisions == ('ieee', 'tf32')``), so the generic
   short-reduction path (``n_cols < 32``) dies with a ``CompilationError``.
2. Any per-lane ``tl.where``/clamp applied to a ``tl.dot`` operand silently
   corrupts the result (generic kernel: ``xi_c = tl.where(mask, xi_c, 0.0)``,
   measured relative error ~0.6), while any index ``%``/clamp inside a dot
   kernel instead drops the whole kernel onto a per-lane scalar path (measured
   38.7 -> 0.3 GFLOP/s; the same shape without the modulo runs at 38.7).

The kernels below therefore keep every dot operand mask-free, clamp-free and
fully affine: when a shape does not divide the dot tiles it is padded by a plain
(non-dot) prep kernel and the padded result is cropped by a non-dot copy
(duplicate last-row/column writes, which are idempotent).  Centering is done
either in that prep kernel (padded path) or in registers with a row-broadcast
subtract inside the dot kernel (measured exact on XPU3, unlike ``tl.where``).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Tile choices, largest first.  ``tl.dot`` needs >= 16 on every axis.  Every
# candidate divides the benchmark/test shapes whenever possible so the pad-free
# (fast) path is taken.  BLOCK_K=128 measured best (or equal) on XPU3.
_BLOCK_N_CHOICES = (128, 64, 32)
_BLOCK_K_CHOICES = (128, 64, 32)
_ROW_BLOCK = 128  # prep/weight 1D tile width; >= 64 lanes or tl.sum miscompiles
_RED_BLOCK = 64   # weight-reduction tile: 128-lane double accumulators trip XPU
_ROWS = 16        # row-block of the dot-based mean; divides every fast-path n_rows
_CROP_BLOCK = 64


def _pick_tile(n, choices):
    for c in choices:
        if n % c == 0:
            return c
    return choices[-1]


@triton.jit
def _cov_weight_kernel(
    wf_ptr,
    wa_ptr,
    w_ptr,
    n_cols,
    stride_wf,
    stride_wa,
    HAS_WF: tl.constexpr,
    HAS_WA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # w[k] = fweights[k] * aweights[k]; padding lanes are exactly zero so the
    # padded covariance columns contribute nothing.  "kpad" is always a
    # multiple of BLOCK, hence the store is unmasked and in-bounds.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_cols
    acc = tl.full([BLOCK], 1.0, dtype=tl.float32)
    if HAS_WF:
        wf = tl.load(wf_ptr + offs * stride_wf, mask=valid, other=0).to(tl.float32)
        acc = acc * wf
    if HAS_WA:
        wa = tl.load(wa_ptr + offs * stride_wa, mask=valid, other=0.0).to(tl.float32)
        acc = acc * wa
    # Value mask-free into the store: padded lanes become exactly 0.
    acc = tl.where(valid, acc, 0.0)
    tl.store(w_ptr + offs, acc)


@triton.jit
def _cov_sum_kernel(ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    # Sum of a 1-D vector to a single device scalar.  64-lane tiles: a 128-lane
    # loop-carried accumulator can trip the XPU unroll pass ("operand does not
    # dominate this use", seen with two accumulators in one loop).
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for m0 in range(0, n, BLOCK):
        offs = m0 + tl.arange(0, BLOCK)
        acc += tl.load(ptr + offs * stride, mask=offs < n, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(acc))


@triton.jit
def _cov_sumprod_kernel(w_ptr, wa_ptr, out_ptr, n, stride_wa, BLOCK: tl.constexpr):
    # sum(w * aweights) to a device scalar.
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for m0 in range(0, n, BLOCK):
        offs = m0 + tl.arange(0, BLOCK)
        mask = offs < n
        wv = tl.load(w_ptr + offs)
        av = tl.load(wa_ptr + offs * stride_wa, mask=mask, other=0.0).to(tl.float32)
        acc += wv * av
    tl.store(out_ptr, tl.sum(acc))


@triton.jit
def _cov_fact_kernel(wsum_ptr, qsum_ptr, out_ptr, correction, HAS_WA: tl.constexpr):
    # out = [w_sum, fact].  fact mirrors torch.cov:
    #   fweights only -> w_sum - correction
    #   any aweights  -> w_sum - correction * sum(w*aweights) / w_sum
    # clamped at 0.  Combining the two device scalars here keeps the op free of
    # host synchronisation and of torch reductions (no fallback).
    ws = tl.load(wsum_ptr)
    if HAS_WA:
        qs = tl.load(qsum_ptr)
        fact = ws - correction * qs / ws
    else:
        fact = ws - correction
    fact = tl.maximum(fact, 0.0)
    tl.store(out_ptr + 0, ws)
    tl.store(out_ptr + 1, fact)


@triton.jit
def _cov_rowsum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    w_sum,
    stride_xn,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Pad-free path row means, done as a GEMM with a constant-ones operand:
    # acc = x[rows, :] @ ones[:, rows] so every output column holds the same
    # row sums.  A per-row 1-D loop accumulator (the generic scheme) measures
    # 200+ us on the largest shapes because the K loop is serial inside one
    # program; the dot form runs on the matrix unit and takes ~11 us.  The
    # full [ROWS, ROWS] tile is stored (reading column 0 later) because
    # extracting one column of a dot result needs a per-lane select, which this
    # backend miscompiles (measured err 1e3).
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    ones = tl.full((BLOCK_K, ROWS), 1.0, dtype=tl.float32)
    acc = tl.zeros((ROWS, ROWS), dtype=tl.float32)
    for k0 in range(0, n_cols, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * stride_xn + offs_k[None, :]).to(
            tl.float32
        )
        acc += tl.dot(x, ones, input_precision="ieee")
    cols = tl.arange(0, ROWS)
    tl.store(out_ptr + rows[:, None] * ROWS + cols[None, :], acc / w_sum)


@triton.jit
def _cov_prep_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    outw_ptr,
    n_rows,
    n_cols,
    kpad,
    w_sum,
    wsum_ptr,
    stride_xn,
    stride_xm,
    stride_on,
    HAS_W: tl.constexpr,
    WSUM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Padded path (row and/or observation count not divisible by the dot tiles,
    # or a strided observation axis): materialise out[row, k] =
    # (x[row, k] - mean[row]) and, when weighted, outw = w[k] * out.  Only ONE
    # dot operand carries the weight, so the dot yields sum_k w (xi-mi)(xj-mj)
    # rather than the squared weight.  Everything outside the valid region is
    # zero (the buffers are zero-initialised), so every dot load stays affine
    # and in-bounds.  BLOCK_M divides kpad.
    row = tl.program_id(0)
    valid = row < n_rows
    src_row = row % n_rows  # scalar wrap keeps the loads inside x
    scale = tl.where(valid, 1.0, 0.0)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for m0 in range(0, n_cols, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        x = tl.load(
            x_ptr + src_row * stride_xn + offs * stride_xm, mask=mask, other=0.0
        ).to(tl.float32)
        if HAS_W:
            x = x * tl.load(w_ptr + offs, mask=mask, other=0.0)
        acc += x
    if WSUM_PTR:
        ws = tl.load(wsum_ptr)
    else:
        ws = w_sum
    mean = tl.sum(acc) / ws
    for m0 in range(0, kpad, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < n_cols
        x = tl.load(
            x_ptr + src_row * stride_xn + offs * stride_xm, mask=mask, other=0.0
        ).to(tl.float32)
        val = (x - mean) * mask.to(tl.float32) * scale
        tl.store(out_ptr + row * stride_on + offs, val)
        if HAS_W:
            wv = tl.load(w_ptr + offs, mask=mask, other=0.0)
            tl.store(outw_ptr + row * stride_on + offs, val * wv)


@triton.jit
def _cov_gram_centered_kernel(
    x_ptr,
    mean_ptr,
    c_ptr,
    kpad,
    fact,
    fact_ptr,
    stride_xn,
    mean_stride,
    stride_c,
    FACT_PTR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Pad-free path: operands are loaded with fully affine, mask-free addresses
    # and centered in registers (row-broadcast subtract).  Measured exact on
    # XPU3, unlike the generic `tl.where(mask, ...)` variant.  ``mean_ptr`` is
    # the [n_rows, _ROWS] row-sum tile from ``_cov_rowsum_kernel``, so the mean
    # of row r lives at mean_ptr + r * mean_stride (column 0).
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    mi = tl.load(mean_ptr + offs_i * mean_stride)
    mj = tl.load(mean_ptr + offs_j * mean_stride)
    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, kpad, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(x_ptr + offs_i[:, None] * stride_xn + offs_k[None, :]).to(
            tl.float32
        )
        b = tl.load(x_ptr + offs_j[:, None] * stride_xn + offs_k[None, :]).to(
            tl.float32
        )
        a = a - mi[:, None]
        b = b - mj[:, None]
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
    if FACT_PTR:
        f = tl.load(fact_ptr + 1)
    else:
        f = fact
    tl.store(c_ptr + offs_i[:, None] * stride_c + offs_j[None, :], acc / f)


@triton.jit
def _cov_gram_kernel(
    xa_ptr,
    xb_ptr,
    c_ptr,
    kpad,
    fact,
    fact_ptr,
    stride_xn,
    stride_c,
    FACT_PTR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Padded path: ``xa_ptr`` / ``xb_ptr`` are the pre-centered buffers from
    # ``_cov_prep_kernel`` (identical unless weighted); their padding is exactly
    # zero, so loads stay affine and the dot is the weighted cross-product.
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    offs_i = pid_i * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, kpad, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(xa_ptr + offs_i[:, None] * stride_xn + offs_k[None, :])
        b = tl.load(xb_ptr + offs_j[:, None] * stride_xn + offs_k[None, :])
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
    if FACT_PTR:
        f = tl.load(fact_ptr + 1)
    else:
        f = fact
    tl.store(c_ptr + offs_i[:, None] * stride_c + offs_j[None, :], acc / f)


@triton.jit
def _cov_crop_kernel(
    src_ptr,
    dst_ptr,
    stride_src,
    stride_dst,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Non-dot copy/cast of the valid top-left block.  Indices are clamped and
    # the tail lanes re-write the last row/column with the same value
    # (idempotent), so the unmasked tile store never leaves the destination.
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    ri = pid_i * BLOCK + tl.arange(0, BLOCK)
    rj = pid_j * BLOCK + tl.arange(0, BLOCK)
    ri = tl.where(ri < N, ri, N - 1)
    rj = tl.where(rj < N, rj, N - 1)
    v = tl.load(src_ptr + ri[:, None] * stride_src + rj[None, :])
    tl.store(dst_ptr + ri[:, None] * stride_dst + rj[None, :], v.to(dst_ptr.dtype.element_ty))


def cov(
    input: torch.Tensor,
    *,
    correction: int = 1,
    fweights: torch.Tensor = None,
    aweights: torch.Tensor = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COV")

    if input.dim() > 2:
        raise ValueError(
            f"cov(): expected input to have two or fewer dimensions but got {input.dim()}"
        )

    # torch.cov treats a 0D/1D input as one variable observed across the
    # elements; a 2D input with a single row likewise yields a scalar.
    x = input.view(1, -1) if input.dim() < 2 else input
    n_rows, n_cols = x.shape
    single_variable = n_rows == 1
    device = x.device
    out_dtype = input.dtype

    has_wf = fweights is not None
    has_wa = aweights is not None
    has_w = has_wf or has_wa
    wf = fweights
    wa = aweights
    if has_wf:
        if wf.dim() != 1 or wf.numel() != n_cols:
            raise ValueError(f"cov(): fweights must be a vector of length {n_cols}")
        if wf.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("cov(): fweights must be an integer vector")
    if has_wa:
        if wa.dim() != 1 or wa.numel() != n_cols:
            raise ValueError(f"cov(): aweights must be a vector of length {n_cols}")
        if wa.is_complex():
            raise TypeError("cov(): aweights must be a real vector")

    # Tile selection: prefer a tile pair that divides the shape so neither axis
    # needs padding (the fast path).  BLOCK_K also drives the K-padding, which
    # is rounded up to _ROW_BLOCK so the prep/weight loops tile it exactly.
    BLOCK_N = _pick_tile(n_rows, _BLOCK_N_CHOICES)
    BLOCK_K = _pick_tile(n_cols, _BLOCK_K_CHOICES)
    npad = triton.cdiv(n_rows, BLOCK_N) * BLOCK_N
    kpad = triton.cdiv(n_cols, BLOCK_K) * BLOCK_K
    kpad = triton.cdiv(kpad, _ROW_BLOCK) * _ROW_BLOCK
    stride_xn, stride_xm = x.stride(0), x.stride(1)
    # Fall back to the padded (materialised) path when the shape does not tile,
    # when the observation axis is strided (dot operands must be K-contiguous on
    # XPU3), or when weights are used (weighting is folded into one operand
    # during the prep copy -- a per-lane multiply inside a dot kernel is unsafe).
    padded = (npad != n_rows) or (kpad != n_cols) or stride_xm != 1 or has_w

    w = None
    fact_buf = None
    wsum_buf = None
    if has_w:
        # Combined float32 observation weights, zero in the K padding.
        # torch.zeros: allocation/init only.  The store below inherits the
        # load mask on XPU3 (masked-store mask accumulation), so the padding
        # lanes must be pre-zeroed here rather than by the kernel.
        w = torch.zeros(kpad, device=device, dtype=torch.float32)
        _cov_weight_kernel[(kpad // _ROW_BLOCK,)](
            wf,
            wa,
            w,
            n_cols,
            wf.stride(0) if has_wf else 1,
            wa.stride(0) if has_wa else 1,
            HAS_WF=has_wf,
            HAS_WA=has_wa,
            BLOCK=_ROW_BLOCK,
        )
        # w_sum / sum(w*aweights) / fact all stay on device: no host sync and
        # no torch reduction (no fallback).
        wsum_buf = torch.empty(1, device=device, dtype=torch.float32)
        _cov_sum_kernel[(1,)](w, wsum_buf, kpad, 1, BLOCK=_RED_BLOCK)
        qsum_buf = wsum_buf
        if has_wa:
            qsum_buf = torch.empty(1, device=device, dtype=torch.float32)
            _cov_sumprod_kernel[(1,)](
                w, wa, qsum_buf, n_cols, wa.stride(0), BLOCK=_RED_BLOCK
            )
        fact_buf = torch.empty(2, device=device, dtype=torch.float32)
        _cov_fact_kernel[(1,)](wsum_buf, qsum_buf, fact_buf, int(correction),
                               HAS_WA=has_wa)
        fact_arg = 0.0
        wsum_arg = 0.0
    else:
        wsum_arg = float(n_cols)
        fact_arg = float(n_cols - correction)
        if fact_arg < 0.0:
            fact_arg = 0.0

    if padded:
        # Centered (+ optional weighted) zero-padded copy; dotted mask-free below.
        # torch.zeros: allocation/init only.  The prep store inherits the
        # masked-load mask on XPU3, so padding lanes keep these zeros.
        xp = torch.zeros((npad, kpad), device=device, dtype=torch.float32)
        xw = xp
        if has_w:
            xw = torch.zeros((npad, kpad), device=device, dtype=torch.float32)
        _cov_prep_kernel[(npad,)](
            x,
            w,
            xp,
            xw,
            n_rows,
            n_cols,
            kpad,
            wsum_arg,
            wsum_buf,
            stride_xn,
            stride_xm,
            kpad,
            HAS_W=has_w,
            WSUM_PTR=has_w,
            BLOCK_M=_ROW_BLOCK,
        )
        grid = (npad // BLOCK_N, npad // BLOCK_N)
        cov_pad = torch.empty((npad, npad), device=device, dtype=torch.float32)
        _cov_gram_kernel[grid](
            xw,
            xp,
            cov_pad,
            kpad,
            fact_arg,
            fact_buf,
            kpad,
            cov_pad.stride(0),
            FACT_PTR=has_w,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        if npad == n_rows and out_dtype == torch.float32:
            cov_mat = cov_pad
        else:
            cov_mat = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
            cgrid = (triton.cdiv(n_rows, _CROP_BLOCK), triton.cdiv(n_rows, _CROP_BLOCK))
            _cov_crop_kernel[cgrid](
                cov_pad,
                cov_mat,
                cov_pad.stride(0),
                cov_mat.stride(0),
                N=n_rows,
                BLOCK=_CROP_BLOCK,
            )
    else:
        # Fast path: row sums on the matrix unit, then a mask-free centered dot.
        mean = torch.empty(n_rows, _ROWS, device=device, dtype=torch.float32)
        _cov_rowsum_kernel[(n_rows // _ROWS,)](
            x,
            mean,
            n_cols,
            wsum_arg,
            stride_xn,
            ROWS=_ROWS,
            BLOCK_K=BLOCK_K,
        )
        cov_mat = torch.empty(n_rows, n_rows, device=device, dtype=out_dtype)
        grid = (npad // BLOCK_N, npad // BLOCK_N)
        _cov_gram_centered_kernel[grid](
            x,
            mean,
            cov_mat,
            kpad,
            fact_arg,
            fact_buf,
            stride_xn,
            _ROWS,
            cov_mat.stride(0),
            FACT_PTR=has_w,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    if single_variable:
        return cov_mat.view(())
    return cov_mat
