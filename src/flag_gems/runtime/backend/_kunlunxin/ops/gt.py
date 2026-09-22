import logging
import math
import os
import struct

import numpy as np
import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


def _wrap_fp16(B):
    """Round python float B to fp16, returning (is_finite, fp16_value_as_float).
    Bit-identical to `torch.tensor(B, float16).item()` but without any tensor
    allocation (host-overhead trim for the small/mid shapes)."""
    v = float(np.float16(float(B)))
    return math.isfinite(v), v


def _wrap_to_dtype(B, dtype):
    """Round python float B to `dtype` (fp16/fp32/bf16) numeric value, returning
    (is_finite, value_as_float). Matches `torch.tensor(B, dtype).item()` exactly
    without any tensor allocation."""
    if dtype == torch.float16:
        v = float(np.float16(float(B)))
    elif dtype == torch.float32:
        v = float(np.float32(float(B)))
    else:  # bfloat16: RNE-truncate f32(B) to bf16, then read its numeric value
        u32 = struct.unpack("<I", struct.pack("<f", float(B)))[0]
        u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF
        v = struct.unpack("<f", struct.pack("<I", u16 << 16))[0]
    return math.isfinite(v), v


def _bf16_as_fp16_key(B):
    """Reinterpret the bits of bf16(B) as an fp16 python float.

    Reinterpreting a bf16 bit pattern as fp16 is order-preserving in the real
    value: both IEEE formats order by the same low-15-bit integer for a fixed
    sign, and the sign bit lines up, so `x >bf16 c  <=>  fp16bits(x) >fp16
    fp16bits(c)` exactly (negatives included). This lets a bf16 compare run on
    the native fp16 32-lane vcmpf with NO bf16->f32 vmerge widen.

    Returns (ok, fp16_key_float). ok=False when bf16(B) is non-finite or its
    bits reach the fp16 inf/nan exponent band (0x7C00 mask), i.e. |B| >~ 2.6e36,
    the only region where the reinterpret stops being order-preserving."""
    fb = float(B)
    if not math.isfinite(fb):
        return False, 0.0
    u32 = struct.unpack("<I", struct.pack("<f", fb))[0]
    u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF  # RNE f32->bf16 bits
    if (u16 & 0x7C00) == 0x7C00:  # fp16 exp all-ones => inf/nan reinterpret
        return False, 0.0
    return True, struct.unpack("<e", struct.pack("<H", u16))[0]


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logger.debug("GEMS_KUNLUNXIN GT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = gt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


config_scalar_bigtile_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    buffer_size_limit=16384,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar_bigtile_,
)
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


# ---------------------------------------------------------------------------
# gt_scalar SMALL/MID-shape fast paths (mirrors eq_scalar).
#
# The generic pointwise_dynamic path is dominated by its host dispatch layer
# (~60-180us/call for small shapes, ~28us/call at mid) which dwarfs the short
# device kernel. Bypassing it with hand-written kernels that keep the SAME
# CmpF-fusion device path (`x.to(f32) > wrapped_scalar` -> vcmpf, needs the
# TRITONXPU_COMPARE_FUSION/FP16_FAST env) recovers the framework overhead.
#
# SMALL (numel <= _GT_SCALAR_SMALL_NUMEL): single masked tile. MID (65536 <
# numel <= _GT_SCALAR_MID_NUMEL): multi-tile kernel with TILE=65536 +
# buffer_size_limit=16384 + unroll_num=16, matching the tuned pointwise DMA
# structure (ONE big gm2lm(65536) -> chunked compute -> ONE big lm2gm(65536))
# so the DMA is coalesced instead of many tiny transactions (see eq.py notes).
# fp32 is excluded from MID: its longer kernel already amortizes the host cost.
# NOTE bf16 does NOT reach here as bf16: gt_scalar() reinterprets bf16 tensors as
# fp16 up front (see _bf16_as_fp16_key), because a bf16 bit pattern read as fp16 is
# order-preserving in the real value, so bf16 rides the native fp16 32-lane vcmpf
# with no bf16->f32 widen. Only fp16/fp32 (and the rare non-finite/overflow bf16
# scalar that fails the bitcast guard) fall through to these paths as-is.
# Semantics identical to gt_func_scalar; the scalar is wrapped to the tensor
# dtype for bit-exact parity with torch (non-finite scalars keep the generic
# path).
_GT_SCALAR_SMALL_NUMEL = 65536
_GT_SCALAR_MID_NUMEL = 4194304
_GT_SCALAR_MID_TILE = 65536
_GT_SCALAR_MID_BSL = 16384
_GT_SCALAR_MID_UNROLL = 16


@triton.jit
def gt_scalar_small_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    tid = tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    tl.store(out_ptr + tid, (x > scalar).to(tl.int8), mask=mask)


def _gt_scalar_small(A, wrapped):
    numel = A.numel()
    out = torch.empty(numel, dtype=torch.int8, device=A.device)
    TILE = triton.next_power_of_2(numel)
    gt_scalar_small_kernel[(1,)](
        out,
        A.reshape(-1),
        wrapped,
        numel,
        TILE=TILE,
        num_warps=4,
        isCloseMemoryAsync=False,
    )
    return out.view(torch.bool).reshape(A.shape)


@triton.jit
def gt_scalar_tile_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    mask = off < numel
    x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
    tl.store(out_ptr + off, (x > scalar).to(tl.int8), mask=mask)


def _gt_scalar_tiled(A, wrapped):
    numel = A.numel()
    out = torch.empty(numel, dtype=torch.int8, device=A.device)
    grid = (triton.cdiv(numel, _GT_SCALAR_MID_TILE),)
    gt_scalar_tile_kernel[grid](
        out,
        A.reshape(-1),
        wrapped,
        numel,
        TILE=_GT_SCALAR_MID_TILE,
        num_warps=4,
        isCloseMemoryAsync=False,
        buffer_size_limit=_GT_SCALAR_MID_BSL,
        unroll_num=_GT_SCALAR_MID_UNROLL,
    )
    return out.view(torch.bool).reshape(A.shape)


def gt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_SCALAR")
    # bf16 fast path via BITCAST bf16->fp16 (no widen). Reinterpreting a bf16 bit
    # pattern as fp16 is order-preserving in the real value (see _bf16_as_fp16_key),
    # so `x >bf16 B  <=>  fp16bits(x) >fp16 fp16bits(B)` exactly. Viewing A as fp16
    # and comparing against the fp16-reinterpreted scalar key routes bf16 through
    # the native fp16 32-lane vcmpf (NO bf16->f32 vmerge widen -> ~1.5x faster),
    # while every downstream fp16 path stays bit-exact vs torch.gt on bf16.
    if A.dtype == torch.bfloat16 and A.is_contiguous():
        ok, key = _bf16_as_fp16_key(B)
        if ok:
            A = A.view(torch.float16)
            B = key
    # Small-shape fast path: bypass the pointwise_dynamic host layer with a
    # single-tile hand kernel (finite wrapped scalar only, for bit-exact parity).
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < A.numel() <= _GT_SCALAR_SMALL_NUMEL
    ):
        wrapped_finite, wrapped = _wrap_to_dtype(B, A.dtype)
        if wrapped_finite:
            os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
            os.environ["TRITONXPU_FP16_FAST"] = "1"
            res = _gt_scalar_small(A, wrapped)
            del os.environ["TRITONXPU_COMPARE_FUSION"]
            del os.environ["TRITONXPU_FP16_FAST"]
            return res
    # Mid-shape fast path (fp16/fp32, 65536 < numel <= _GT_SCALAR_MID_NUMEL):
    # multi-tile hand kernel with the coalesced-DMA tile config. bf16 arrives here
    # already reinterpreted as fp16 (see top of gt_scalar). fp32 is included too:
    # at mid shapes the generic pointwise_dynamic path is dominated by its host
    # dispatch layer, so the hand kernel that bypasses it recovers that overhead
    # (fp32 x.to(f32) is a no-op; the compare is the native 16-lane f32 vcmpf).
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _GT_SCALAR_SMALL_NUMEL < A.numel() <= _GT_SCALAR_MID_NUMEL
    ):
        wrapped_finite, wrapped = _wrap_to_dtype(B, A.dtype)
        if wrapped_finite:
            os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
            os.environ["TRITONXPU_FP16_FAST"] = "1"
            res = _gt_scalar_tiled(A, wrapped)
            del os.environ["TRITONXPU_COMPARE_FUSION"]
            del os.environ["TRITONXPU_FP16_FAST"]
            return res
    # Full-precision float CmpF-fusion path. torch compares `x > wrapped(B)`
    # where wrapped(B) is B rounded to the tensor dtype; passing the wrapped
    # scalar keeps `x.to(f32) > wrapped` bit-identical to torch for fp16/bf16
    # (a raw B differs when B is not representable in the tensor dtype).
    if A.dtype in (torch.float16, torch.bfloat16, torch.float32):
        B = _wrap_to_dtype(B, A.dtype)[1]
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = gt_func_scalar(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def gt_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def gt_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def gt_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    gt_func_tensor_inplace(A, B, out0=A)
    return A


def gt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_ SCALAR")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == _wrap_to_dtype(B, A.dtype)[1]
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel >= _GT_SCALAR_INPLACE_FAST_TILE * _GT_SCALAR_INPLACE_MIN_GRID
            and numel % _GT_SCALAR_INPLACE_FAST_TILE == 0
        ):
            return _gt_scalar_inplace_fast(A, float(B))
        gt_func_scalar_inplace(A, B, out0=A)
        return A
    return gt_func_scalar(A, B, out0=A)


_GT_SCALAR_INPLACE_FAST_TILE = 131072
_GT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def gt_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _gt_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _GT_SCALAR_INPLACE_FAST_TILE,)
    gt_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_GT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
