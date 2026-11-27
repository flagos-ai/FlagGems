"""Triton implementation of ``torch.linalg.vander``.

Contract established from the target's actual torch 2.10 behaviour plus the
Gems pytest:

* ``out.shape == x.shape + (N,)`` (a 0-dim ``x`` is treated as shape ``(1,)``)
* ``out[..., k, j] = x[..., k] ** j``   (powers increasing along the last axis)
* ``N`` defaults to ``x.shape[-1]`` and must be > 1
* integer inputs produce an ``int64`` result
* the pytest exercises float16, float32, bfloat16 and complex64 (complex128 is
  skipped on this device), so half inputs are computed in float32 and complex
  inputs are handled as (re, im) pairs.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _x_batch_off(b, STRIDE_B, S0, S1, S2, S3, D1, D2, D3, NDIM: tl.constexpr):
    """Element offset of flat batch index ``b`` inside ``x``."""
    if NDIM == 0:
        return b * STRIDE_B
    elif NDIM == 1:
        return b * S0
    elif NDIM == 2:
        return (b // D1) * S0 + (b % D1) * S1
    elif NDIM == 3:
        q = b // (D1 * D2)
        r = b % (D1 * D2)
        return q * S0 + (r // D2) * S1 + (r % D2) * S2
    else:
        q = b // (D1 * D2 * D3)
        r = b % (D1 * D2 * D3)
        u = r // (D2 * D3)
        v = r % (D2 * D3)
        return q * S0 + u * S1 + (v // D3) * S2 + (v % D3) * S3


@triton.jit
def _vander_pow_tile(
    x_ptr,
    o_ptr,
    K,
    N,
    STRIDE_B,
    S0,
    S1,
    S2,
    S3,
    D1,
    D2,
    D3,
    SLAST,
    NDIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_I64: tl.constexpr,
    UPCAST: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    xbase = _x_batch_off(b, STRIDE_B, S0, S1, S2, S3, D1, D2, D3, NDIM)
    raw = tl.load(x_ptr + xbase + offs_k.to(tl.int64) * SLAST, mask=mask_k, other=1.0)
    if UPCAST:
        xv = raw.to(tl.float32)
    else:
        xv = raw

    n0 = pid_n * BLOCK_N

    # One log2 per element of x. The power axis is then covered as
    # x**(n0 + q*BLOCK_R + s) = base_q * inner_s, where `inner` is shared by all
    # Q = BLOCK_N/BLOCK_R sub-tiles, so the transcendental count per output element
    # drops from one to about (BLOCK_R + 2) / (BLOCK_R * Q).
    lx = tl.log2(tl.abs(xv))
    neg = xv < 0

    s = tl.arange(0, BLOCK_R)
    sign_s = tl.where((s & 1) == 0, 1.0, -1.0).to(xv.dtype)
    inner = tl.exp2(s.to(xv.dtype)[None, :] * lx[:, None])
    inner = inner * tl.where(neg[:, None], sign_s[None, :], 1.0)
    inner = tl.where(s[None, :] == 0, 1.0, inner)

    pow_r = tl.exp2(lx * BLOCK_R)
    if n0 == 0:
        base = tl.full((BLOCK_K,), 1.0, xv.dtype)
    else:
        base = tl.exp2(n0.to(xv.dtype) * lx)

    if USE_I64:
        row = b * K.to(tl.int64) + offs_k.to(tl.int64)
        obase = row * N.to(tl.int64)
    else:
        obase = (b.to(tl.int32) * K + offs_k) * N
    o_row = o_ptr + obase[:, None]

    for q in tl.static_range(BLOCK_N // BLOCK_R):
        offs_n = n0 + q * BLOCK_R + s
        tl.store(
            o_row + offs_n[None, :],
            (base[:, None] * inner).to(o_ptr.dtype.element_ty),
            mask=mask_k[:, None] & (offs_n < N)[None, :],
        )
        base = base * pow_r


# Bounded configuration search for the measured float32 path. The winning tile
# genuinely depends on the size regime (small totals favour few CTAs with many
# elements per thread, large totals favour many CTAs), so let the device decide
# instead of hard-coding one formula. Half/integer/complex keep fixed configs.
_F32_CONFIGS = [
    triton.Config(
        {"BLOCK_K": bk, "BLOCK_N": bn, "BLOCK_R": br}, num_warps=nw, num_stages=1
    )
    for bk, bn, br, nw in [
        (1, 128, 32, 4),
        (4, 128, 32, 4),
        (16, 128, 32, 4),
        (32, 128, 32, 4),
        (16, 256, 32, 4),
        (32, 256, 32, 4),
        (8, 64, 16, 4),
        (16, 64, 16, 4),
        (32, 32, 8, 4),
        (16, 16, 4, 4),
        (4, 16, 4, 2),
        (4, 4, 4, 1),
    ]
]


@triton.autotune(configs=_F32_CONFIGS, key=["K", "N"])
@triton.jit
def _vander_f32_auto(
    x_ptr,
    o_ptr,
    K,
    N,
    STRIDE_B,
    S0,
    S1,
    S2,
    S3,
    D1,
    D2,
    D3,
    SLAST,
    NDIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_I64: tl.constexpr,
    UPCAST: tl.constexpr,
):
    _vander_pow_tile(
        x_ptr,
        o_ptr,
        K,
        N,
        STRIDE_B,
        S0,
        S1,
        S2,
        S3,
        D1,
        D2,
        D3,
        SLAST,
        NDIM,
        BLOCK_K,
        BLOCK_N,
        BLOCK_R,
        USE_I64,
        UPCAST,
    )


@triton.jit
def _vander_float_kernel(
    x_ptr,
    o_ptr,
    K,
    N,
    STRIDE_B,
    S0,
    S1,
    S2,
    S3,
    D1,
    D2,
    D3,
    SLAST,
    NDIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_I64: tl.constexpr,
    UPCAST: tl.constexpr,
):
    _vander_pow_tile(
        x_ptr,
        o_ptr,
        K,
        N,
        STRIDE_B,
        S0,
        S1,
        S2,
        S3,
        D1,
        D2,
        D3,
        SLAST,
        NDIM,
        BLOCK_K,
        BLOCK_N,
        BLOCK_R,
        USE_I64,
        UPCAST,
    )


@triton.jit
def _vander_int_kernel(
    x_ptr,
    o_ptr,
    K,
    N,
    STRIDE_B,
    S0,
    S1,
    S2,
    S3,
    D1,
    D2,
    D3,
    SLAST,
    NDIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_I64: tl.constexpr,
    BITS: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    xbase = _x_batch_off(b, STRIDE_B, S0, S1, S2, S3, D1, D2, D3, NDIM)
    xv = tl.load(x_ptr + xbase + offs_k.to(tl.int64) * SLAST, mask=mask_k, other=1).to(
        tl.int64
    )

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # exact (wrapping) integer power by binary exponentiation, one bit at a time
    ex = tl.broadcast_to(offs_n[None, :], (BLOCK_K, BLOCK_N))
    res = tl.full((BLOCK_K, BLOCK_N), 1, tl.int64)
    cur = tl.broadcast_to(xv[:, None], (BLOCK_K, BLOCK_N))
    for i in tl.static_range(BITS):
        bit = (ex >> i) & 1
        res = tl.where(bit == 1, res * cur, res)
        cur = cur * cur

    if USE_I64:
        row = b * K.to(tl.int64) + offs_k.to(tl.int64)
        obase = row * N.to(tl.int64)
    else:
        obase = (b.to(tl.int32) * K + offs_k) * N
    tl.store(
        o_ptr + obase[:, None] + offs_n[None, :],
        res,
        mask=mask_k[:, None] & mask_n[None, :],
    )


@triton.jit
def _vander_cplx_kernel(
    xr_ptr,
    o_ptr,
    K,
    N,
    STRIDE_B,
    S0,
    S1,
    S2,
    S3,
    D1,
    D2,
    D3,
    SK,
    NDIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_I64: tl.constexpr,
):
    """Complex powers via the multiplication recurrence along the power axis.

    ``xr_ptr`` and ``o_ptr`` are float views of complex data, so the real part of
    element ``k`` sits at offset ``i`` and the imaginary part at ``i + 1``.
    """
    pid_k = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    xbase = _x_batch_off(b, STRIDE_B, S0, S1, S2, S3, D1, D2, D3, NDIM)
    idx = xbase + offs_k.to(tl.int64) * SK
    re = tl.load(xr_ptr + idx, mask=mask_k, other=1.0)
    im = tl.load(xr_ptr + idx + 1, mask=mask_k, other=0.0)

    cr = tl.full((BLOCK_K,), 1.0, re.dtype)
    ci = tl.zeros((BLOCK_K,), re.dtype)

    if USE_I64:
        row = b * K.to(tl.int64) + offs_k.to(tl.int64)
        obase = row * (N.to(tl.int64) * 2)
    else:
        obase = (b.to(tl.int32) * K + offs_k) * (N * 2)
    for j in range(N):
        cj = j * 2
        tl.store(o_ptr + obase + cj, cr, mask=mask_k)
        tl.store(o_ptr + obase + cj + 1, ci, mask=mask_k)
        cr, ci = cr * re - ci * im, cr * im + ci * re


def _next_pow2(v):
    return 1 << max(0, int(v) - 1).bit_length()


def linalg_vander(x, N=None):
    logger.debug("GEMS_ILUVATAR LINALG_VANDER")
    if not isinstance(x, torch.Tensor):
        raise TypeError("linalg_vander(): argument 'x' must be a Tensor")
    dtype = x.dtype
    if dtype is torch.bool:
        raise RuntimeError(
            "linalg.vander supports floating point, complex, and integer tensors, but got Bool"
        )

    if N is None:
        if x.dim() == 0:
            raise RuntimeError("N must be greater than 1.")
        n_out = int(x.shape[-1])
    else:
        n_out = int(N)
    if n_out <= 1:
        raise RuntimeError("N must be greater than 1.")

    if x.dim() == 0:
        xin = x.view(1)
        out_shape = (1, n_out)
    else:
        xin = x
        out_shape = tuple(x.shape) + (n_out,)

    is_complex = dtype.is_complex
    is_int = (not is_complex) and (not dtype.is_floating_point)
    out = torch.empty(
        out_shape,
        dtype=torch.int64 if is_int else dtype,
        device=x.device,
    )
    if out.numel() == 0:
        return out

    k = int(xin.shape[-1])
    batch = xin.numel() // k
    ndim = xin.dim() - 1

    if is_complex:
        xr = torch.view_as_real(xin)
        orr = torch.view_as_real(out)
        s = list(xr.stride())
        sk = s[-2]
        if ndim == 0:
            stride_b, ndim_c = k * 2, 0
            s0 = s1 = s2 = s3 = d1 = d2 = d3 = 1
        else:
            lead = s[:ndim]
            dims = xin.shape[:ndim]
            stride_b = lead[0]
            ndim_c = min(ndim, 4)
            s0, s1, s2, s3 = (lead + [1, 1, 1, 1])[:4]
            d1, d2, d3 = (list(dims[1:]) + [1, 1, 1])[:3]
        use_i64 = 2 * batch * k * n_out >= 2147483647
        bk = min(_next_pow2(max(k, 1)), 512)
        _vander_cplx_kernel[(triton.cdiv(k, bk), batch)](
            xr,
            orr,
            k,
            n_out,
            stride_b,
            s0,
            s1,
            s2,
            s3,
            d1,
            d2,
            d3,
            sk,
            ndim_c,
            bk,
            use_i64,
            num_warps=4,
            num_stages=1,
        )
        return out

    s = list(xin.stride())
    slast = s[-1] if s else 1
    if ndim == 0:
        stride_b = k
        s0 = s1 = s2 = s3 = d1 = d2 = d3 = 1
        ndim_c = 0
    else:
        lead = s[:ndim]
        dims = xin.shape[:ndim]
        stride_b = lead[0]
        ndim_c = min(ndim, 4)
        s0, s1, s2, s3 = (lead + [1, 1, 1, 1])[:4]
        d1, d2, d3 = (list(dims[1:]) + [1, 1, 1])[:3]

    total = batch * k * n_out
    use_i64 = total >= 2147483647 or xin.numel() >= 2147483647

    if dtype is torch.float32:
        _vander_f32_auto[
            lambda meta: (
                triton.cdiv(k, meta["BLOCK_K"]),
                triton.cdiv(n_out, meta["BLOCK_N"]),
                batch,
            )
        ](
            xin,
            out,
            k,
            n_out,
            stride_b,
            s0,
            s1,
            s2,
            s3,
            d1,
            d2,
            d3,
            slast,
            ndim_c,
            USE_I64=use_i64,
            UPCAST=False,
        )
        return out

    bn = min(_next_pow2(n_out), 128)
    bk = max(8, min(_next_pow2(min(max(k, 1), 512)), max(8, 2048 // bn)))

    grid = (triton.cdiv(k, bk), triton.cdiv(n_out, bn), batch)
    if is_int:
        _vander_int_kernel[grid](
            xin,
            out,
            k,
            n_out,
            stride_b,
            s0,
            s1,
            s2,
            s3,
            d1,
            d2,
            d3,
            slast,
            ndim_c,
            bk,
            bn,
            use_i64,
            max(1, int(bn - 1).bit_length()),
            num_warps=4,
            num_stages=1,
        )
    else:
        _vander_float_kernel[grid](
            xin,
            out,
            k,
            n_out,
            stride_b,
            s0,
            s1,
            s2,
            s3,
            d1,
            d2,
            d3,
            slast,
            ndim_c,
            bk,
            bn,
            min(32, bn),
            use_i64,
            dtype in (torch.float16, torch.bfloat16),
            num_warps=4,
            num_stages=1,
        )
    return out
