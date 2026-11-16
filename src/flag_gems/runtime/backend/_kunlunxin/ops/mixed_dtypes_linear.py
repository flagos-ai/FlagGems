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

"""Kunlunxin XPU3 override for ``aten::_mixed_dtypes_linear``.

``out = act(input @ dequant(weight) * scale + bias)`` with ``input`` fp16/bf16,
``weight`` uint8 (int8 mode: one signed-int8 per output column; int4 mode: two
signed-int4 nibbles packed per byte) and per-output ``scale`` ``(N,)``.

Why a backend override
----------------------
The generic implementation does not compile on XPU3 (12/12 baseline cases fail
at compile time): the fp16 path promotes the weight tile to fp32 so ``tl.dot``
sees ``fp16 x fp32`` and asserts; the int4 path re-packs the two nibble
accumulators with ``tl.interleave`` and the SDNN lowering asserts ``rank == 2``;
and every load inside the dot loop is masked (the known dot+masked-load trap).

Design: pre-dequantise -> single-dot GEMM -> separate activation kernel
-----------------------------------------------------------------------
* A **non-dot** elementwise kernel materialises the dequantised weight as a
  dense ``[K, N]`` tensor of the input dtype.  int4 nibbles unpack to output
  columns ``2j`` / ``2j+1``; for bf16 the per-column ``scale`` is folded in here
  (the reference order of operations - folding it into the accumulator instead
  exceeds bf16 tolerance).  int8/fp16 reads the raw uint8 weight directly in the
  GEMM (fast path; no materialisation).
* The GEMM is a plain mask-free single-dot loop (``acc += tl.dot(...)``, 2-arg)
  over a dense / uint8 weight; the accumulator is written as **fp32** so the
  epilogue stays fp32 like the reference.
* ``silu``/``relu`` run in a **separate non-dot kernel** (``sigmoid`` / an
  ``x/(1+exp(-x))`` form inside the SDNN dot kernel dies with
  ``KeyError: 'tt.sigmoid'/'tt.silu'`` at ``make_ewtable``; the standalone
  elementwise path, as in ``swiglu.py``, is fine).  That kernel also casts and
  compacts to the real output.

XPU3 performance note (measured)
--------------------------------
The backend devectorises elementwise kernels to a crawl whenever an index is
not a plain affine ``pid*const + arange``: ``tl.minimum`` clamp (~60-100 ms),
flat ``off // N`` / ``off % N`` (~55-95 ms) and a runtime ``BASE`` offset
(~36 ms) all collapse a 16 M-element pass; the same pass with clean
exact-divisor blocks runs in ~0.5 ms.  So every elementwise kernel here uses
**exact-divisor, mask-free, affine tiles** (no clamp, no modulo, no dynamic
base) and every access is in bounds by construction.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_ACT_MAP = {"none": 0, None: 0, "relu": 1, "silu": 2}

_GROUP_M = 8
_MAX_ELEMS_PER_PROG = 8192  # keep below the XPU3 32768 illegal-access cliff


@triton.jit
def _grouped_pids(pid, M, N, BM: tl.constexpr, BN: tl.constexpr,
                  GROUP_M: tl.constexpr):
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + ((pid % width) % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


def _pow2_divisor(x, cap):
    """Largest power of two <= cap that divides x (at least 1)."""
    p = 1
    while p * 2 <= cap and x % (p * 2) == 0:
        p *= 2
    return p


def _largest_pow2_le(x, cap):
    """Largest power of two <= min(cap, x)."""
    p = 1
    lim = min(cap, x)
    while p * 2 <= lim:
        p *= 2
    return p


@libentry()
@triton.jit
def _mdl_dequant_kernel(
    W,
    S,
    WD,
    stride_wn,
    stride_wd,
    BASE: tl.constexpr,
    INT4: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Dense weight ``WD[k, n]`` of the input dtype from a uint8 weight.

    int8: ``WD = W - 128``.  int4: low/high nibble -> output column ``2j`` /
    ``2j+1``, ``WD = nibble - 8``.  ``PRE_SCALE`` folds in ``S[n]`` (bf16 path).

    Affine, mask-free tiles ``rc = BASE + pid_c*BC + arange(BC)``.  When ``BC``
    does not divide ``N`` a second launch with ``BASE = N - BC`` covers the tail
    (in bounds, re-writing an overlap with identical values).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rr = pid_r * BR + tl.arange(0, BR)
    rc = BASE + pid_c * BC + tl.arange(0, BC)
    if INT4:
        jc = rc // 2
        wb = tl.load(W + rr[:, None] * stride_wn + jc[None, :])
        w32 = wb.to(tl.int32)
        lo = (w32 & 0xF) - 8
        hi = ((w32 >> 4) & 0xF) - 8
        valf = (lo + (hi - lo) * (rc % 2)[None, :]).to(tl.float32)
    else:
        wb = tl.load(W + rr[:, None] * stride_wn + rc[None, :])
        valf = wb.to(tl.float32) - 128.0
    if PRE_SCALE:
        valf = valf * tl.load(S + rc).to(tl.float32)[None, :]
    tl.store(
        WD + rr[:, None] * stride_wd + rc[None, :],
        valf.to(WD.dtype.element_ty),
    )


@libentry()
@triton.jit
def _mdl_gemm_kernel(
    X,
    W,
    S,
    Bias,
    O,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    W_UINT8: tl.constexpr,
    EPILOGUE_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Single-dot GEMM over a dense (or raw uint8) weight; fp32 padded store.

    Rows ``% M`` / cols ``% N`` wrapped (in bounds); the store goes to the
    padded fp32 buffer.
    """
    pid = tl.program_id(0)
    pid_m, pid_n = _grouped_pids(pid, M, N, BM, BN, GROUP_M)

    rm = pid_m * BM + tl.arange(0, BM)
    ram = rm % M
    rn = pid_n * BN + tl.arange(0, BN)
    rbn = rn % N
    rk = tl.arange(0, BK)

    x_ptrs = X + ram[:, None] * stride_xm + rk[None, :]
    w_ptrs = W + rk[:, None] * stride_wn + rbn[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        xv = tl.load(x_ptrs)
        wb = tl.load(w_ptrs)
        if W_UINT8:
            wf = (wb.to(xv.dtype) - 128.0).to(xv.dtype)
        else:
            wf = wb
        acc += tl.dot(xv, wf, out_dtype=tl.float32, allow_tf32=False)
        x_ptrs += BK
        w_ptrs += BK * stride_wn

    if EPILOGUE_SCALE:
        acc = acc * tl.load(S + rbn).to(tl.float32)[None, :]
    if HAS_BIAS:
        acc = acc + tl.load(Bias + rbn).to(tl.float32)[None, :]
    o_ptrs = O + rm[:, None] * stride_om + rn[None, :]
    tl.store(o_ptrs, acc)


@libentry()
@triton.jit
def _mdl_act_kernel(
    Buf,
    O,
    stride_b,
    stride_o,
    BASE: tl.constexpr,
    ACT: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Separate (non-dot) epilogue: activation + cast over the padded buffer.

    Affine, mask-free tiles ``rc = BASE + pid_c*BC + arange(BC)`` (same tail
    trick as the dequant).  The padded buffer covers ``(Mpad, n_pad)`` with
    ``n_pad >= N`` so every access is in bounds.  This is the separate
    compilation unit that lets ``tl.sigmoid`` (silu) lower without the SDNN
    activation-table ``KeyError``.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rr = pid_r * BR + tl.arange(0, BR)
    rc = BASE + pid_c * BC + tl.arange(0, BC)
    v = tl.load(Buf + rr[:, None] * stride_b + rc[None, :])
    if ACT == 1:
        v = tl.maximum(v, 0.0)
    elif ACT == 2:
        v = v * tl.sigmoid(v)
    tl.store(O + rr[:, None] * stride_o + rc[None, :], v.to(O.dtype.element_ty))


@triton.jit
def _mdl_copy_kernel(
    Src,
    Dst,
    R,
    C,
    stride_sr,
    stride_sc,
    stride_dr,
    stride_dc,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Rare ragged-K padding copy. Clamped (in bounds) loads, masked store.

    Only used when ``K % 16 != 0`` (never on the tested shapes); the
    ``tl.minimum`` clamp is slow but safe here.
    """
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    rr = pr * BR + tl.arange(0, BR)
    rc = pc * BC + tl.arange(0, BC)
    m = (rr < R)[:, None] & (rc < C)[None, :]
    rrc = tl.minimum(rr, R - 1)
    rcc = tl.minimum(rc, C - 1)
    v = tl.load(Src + rrc[:, None] * stride_sr + rcc[None, :] * stride_sc)
    tl.store(Dst + rrc[:, None] * stride_dr + rcc[None, :] * stride_dc, v, mask=m)


def _copy2d(src, dst, rows, cols, src_strides, dst_stride0):
    br, bc = 32, 64
    _mdl_copy_kernel[(triton.cdiv(rows, br), triton.cdiv(cols, bc))](
        src,
        dst,
        rows,
        cols,
        src_strides[0],
        src_strides[1],
        dst_stride0,
        1,
        BR=br,
        BC=bc,
    )


def _launch_col_tiled(kernel, grid_r, N, BC, args, **kw):
    """Launch a column-tiled elementwise kernel over ``N`` columns.

    Full blocks use ``BASE=0`` (``pid_c`` affine).  If ``BC`` does not divide
    ``N`` a second launch with ``BASE = N - BC`` covers the tail: in bounds by
    construction, re-writing an overlap of width ``BC - rem`` with identical
    values.  Keeping ``BASE`` a constexpr keeps the index affine so the backend
    does not devectorise the kernel.
    """
    full = N // BC
    rem = N - full * BC
    if full > 0:
        kernel[(*grid_r, full)](*args, BASE=0, BC=BC, **kw)
    if rem > 0:
        kernel[(*grid_r, 1)](*args, BASE=N - BC, BC=BC, **kw)


def _pick_block_m(M):
    if M <= 16:
        return 16
    if M <= 32:
        return 32
    if M <= 64:
        return 64
    return 128


def _pick_bk(K):
    if K % 128 == 0:
        return 128
    if K % 64 == 0:
        return 64
    if K % 32 == 0:
        return 32
    return 16


def _pick_bn(N):
    if N >= 8192 and N % 256 == 0:
        return 256
    if N % 128 == 0:
        return 128
    if N % 64 == 0:
        return 64
    return 16


def mixed_dtypes_linear(input, weight, scale, bias=None, activation=None):
    logger.debug("GEMS _MIXED_DTYPES_LINEAR")

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input must be float16 or bfloat16, got {input.dtype}"
    assert weight.dtype == torch.uint8, f"weight must be uint8, got {weight.dtype}"
    assert (
        scale.dtype == input.dtype
    ), f"scale dtype {scale.dtype} must match input dtype {input.dtype}"
    act = activation if activation is not None else "none"
    assert act in _ACT_MAP, f"unsupported activation: {activation}"

    K = input.shape[-1]
    N = scale.shape[0]
    ncols = weight.shape[1]
    assert ncols == N or 2 * ncols == N, (
        f"weight columns {ncols} must equal scale size {N} (int8) "
        f"or half of it (packed int4)"
    )
    int4 = ncols != N

    ndim = input.dim()
    assert input.stride(-1) == 1, "input last dimension must be contiguous"
    for i in range(ndim - 2):
        assert input.stride(i) == input.shape[i + 1] * input.stride(
            i + 1
        ), "input rows must be uniformly strided"
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    stride_wn = weight.stride(0)
    stride_xm = input.stride(-2) if ndim >= 2 else 0
    M = input.numel() // K if K > 0 else 0

    out_shape = (*input.shape[:-1], N)
    if M == 0 or N == 0:
        return torch.empty(out_shape, dtype=input.dtype, device=input.device)

    device = input.device
    with torch_device_fn.device(device):
        ragged_k = K % 16 != 0
        if ragged_k:
            kpad = triton.cdiv(K, 16) * 16
            xpad = torch.zeros((M, kpad), dtype=input.dtype, device=device)
            _copy2d(input, xpad, M, K, (stride_xm, 1), kpad)
            input_ = xpad
            stride_xm = kpad
            wpad = torch.zeros((kpad, ncols), dtype=torch.uint8, device=device)
            _copy2d(weight, wpad, K, ncols, (stride_wn, 1), ncols)
            weight = wpad
            stride_wn = weight.stride(0)
            K = kpad
        else:
            input_ = input

        bm = _pick_block_m(M)
        bk = _pick_bk(K)
        bn = _pick_bn(N)

        # Dense dequantised weight for int4 (nibble unpack) and bf16 (in-loop
        # scale folded into the weight).  int8/fp16 uses the raw-uint8 fast path.
        need_dense = int4 or input.dtype == torch.bfloat16 or ragged_k
        if need_dense:
            wd = torch.empty((K, N), dtype=input.dtype, device=device)
            bc_dq = _largest_pow2_le(N, 4096)
            br_dq = _pow2_divisor(K, max(1, _MAX_ELEMS_PER_PROG // bc_dq))
            _launch_col_tiled(
                _mdl_dequant_kernel,
                (K // br_dq,),
                N,
                bc_dq,
                (weight, scale, wd, stride_wn, N),
                INT4=int4,
                PRE_SCALE=(input.dtype == torch.bfloat16),
                BR=br_dq,
            )
            w_arg = wd
            stride_wn_g = N
            w_uint8 = False
            epilogue_scale = input.dtype != torch.bfloat16
        else:
            w_arg = weight
            stride_wn_g = stride_wn
            w_uint8 = True
            epilogue_scale = True

        n_pad = triton.cdiv(N, bn) * bn
        m_pad = triton.cdiv(M, bm) * bm
        buf = torch.empty((m_pad, n_pad), dtype=torch.float32, device=device)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        bias_arg = bias if bias is not None else buf
        _mdl_gemm_kernel[grid](
            input_,
            w_arg,
            scale,
            bias_arg,
            buf,
            M,
            N,
            K,
            stride_xm,
            stride_wn_g,
            n_pad,
            W_UINT8=w_uint8,
            EPILOGUE_SCALE=epilogue_scale,
            HAS_BIAS=bias is not None,
            BM=bm,
            BN=bn,
            BK=bk,
            GROUP_M=_GROUP_M,
            num_warps=4,
            num_stages=3,
        )

        # Epilogue: activation + cast over the padded buffer, exact columns [0,N).
        out_buf = torch.empty((m_pad, N), dtype=input.dtype, device=device)
        bc_a = _largest_pow2_le(N, 4096)
        br_a = _pow2_divisor(m_pad, max(1, _MAX_ELEMS_PER_PROG // bc_a))
        _launch_col_tiled(
            _mdl_act_kernel,
            (m_pad // br_a,),
            N,
            bc_a,
            (buf, out_buf, n_pad, N),
            ACT=_ACT_MAP[act],
            BR=br_a,
        )

        res = out_buf[:M]
    return res.reshape(out_shape)
