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
# gt_scalar flat fast paths (bypass pointwise_dynamic).
#
# The generic pointwise_dynamic path is dominated by its host dispatch layer
# (~60-180us/call for small shapes, ~28us/call at mid) which dwarfs the short
# device kernel; on large shapes it also pays for a masked program shape. Both
# are recovered by hand-written flat kernels that keep the SAME CmpF-fusion
# device path (needs TRITONXPU_COMPARE_FUSION/FP16_FAST around the launch):
#   * fp32 input       -> `x.to(f32) > wrapped` on the native 16-lane f32 vcmpf
#   * fp16 input       -> `x > fp16(wrapped)` on the native 32-lane fp16 vcmpf
#   * bf16 input       -> load bf16, convert to fp16 (saturating), same native
#     32-lane vcmpf. A bf16 significand is a subset of fp16's, so the
#     conversion is exact inside fp16's exponent range and the compare matches
#     torch bit-for-bit (bf16 NaN -> fp16 NaN -> False, +-inf and huge finite
#     saturate to +-inf -> correct `>`), which a raw bf16->fp16 BITCAST does not
#     (bf16 |x| >= ~2.66e36 / +inf land in the fp16 NaN band and would answer
#     False). Outside fp16's exact range for the scalar, bf16 falls back to the
#     generic path (see _GT_SCALAR_BF16_FP16_MIN/MAX).
# MASKED is a constexpr so shapes that divide the tile skip the XPU
# masked-memory path entirely (that path is ~40% slower on 16.7M: 38us vs 54us).
#
# TILE is picked per numel from measured do_bench sweeps (KERNEL mode) --
# see harness/perf_ir/probe_gt_scalar_{tiny,table,cfg,split}.log:
#   half domain: single program up to 4K | 16K up to 128K | 32K up to 256K |
#                128K above (16.7M 0.82x, 268M 0.79x, 2.56M 0.84x, 67M 0.79x,
#                vs 0.56/0.52/0.63/0.53 for the generic path)
#   fp32 domain: single program up to 4K | 8K up to 64K | 16K up to 128K |
#                32K up to 1M | 64K above (16.7M 0.75x, 268M 0.73x, 2.56M
#                0.76x, 67M 0.73x vs 0.71/0.69/0.70/0.70 generic)
# The compare itself is free on XPU3 (a load + int8-store kernel without any
# compare takes the same time), so this is a pure DMA-shape problem: bigger
# unmasked tiles, never a per-lane path.
# Semantics identical to gt_func_scalar; the scalar is wrapped to the tensor
# dtype for bit-exact parity with torch (non-finite scalars keep the generic
# path).
_GT_SCALAR_TILE_MAX_TINY = 4096
_GT_SCALAR_BSL = 16384
_GT_SCALAR_UNROLL = 16
# bf16 takes the fp16-conversion path only while the wrapped scalar is inside
# fp16's exactly-representable range (2^-14 .. 65504); outside it, bf16 overflow
# to inf / underflow to 0 would change the comparison result.
_GT_SCALAR_BF16_FP16_MIN = 2.0**-14
_GT_SCALAR_BF16_FP16_MAX = 65504.0
# half-domain (fp16 / bf16-converted) tile table
_GT_SCALAR_HALF_TILE_TINY = 16384
_GT_SCALAR_HALF_TILE_MID = 32768
_GT_SCALAR_HALF_TILE_LARGE = 131072
_GT_SCALAR_HALF_MID_MIN = 131072
_GT_SCALAR_HALF_LARGE_MIN = 262144
# fp32-domain tile table
_GT_SCALAR_F32_TILE_TINY = 8192
_GT_SCALAR_F32_TILE_MID = 16384
_GT_SCALAR_F32_TILE_LARGE = 32768
_GT_SCALAR_F32_TILE_HUGE = 65536
_GT_SCALAR_F32_MID_MIN = 65536
_GT_SCALAR_F32_LARGE_MIN = 131072
_GT_SCALAR_F32_HUGE_MIN = 1048576


def _gt_scalar_tile(numel, dtype):
    if dtype == torch.float32:
        if numel <= _GT_SCALAR_F32_MID_MIN:
            return _GT_SCALAR_F32_TILE_TINY
        if numel <= _GT_SCALAR_F32_LARGE_MIN:
            return _GT_SCALAR_F32_TILE_MID
        if numel <= _GT_SCALAR_F32_HUGE_MIN:
            return _GT_SCALAR_F32_TILE_LARGE
        return _GT_SCALAR_F32_TILE_HUGE
    if numel <= _GT_SCALAR_HALF_MID_MIN:
        return _GT_SCALAR_HALF_TILE_TINY
    if numel <= _GT_SCALAR_HALF_LARGE_MIN:
        return _GT_SCALAR_HALF_TILE_MID
    return _GT_SCALAR_HALF_TILE_LARGE


@triton.jit
def gt_scalar_tile_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, MASKED: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    if MASKED:
        mask = off < numel
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        tl.store(out_ptr + off, (x > scalar).to(tl.int8), mask=mask)
    else:
        x = tl.load(x_ptr + off).to(tl.float32)
        tl.store(out_ptr + off, (x > scalar).to(tl.int8))


@triton.jit
def gt_scalar_tile_half_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, MASKED: tl.constexpr
):
    # fp16 loads keep fp16 (the explicit .to(tl.float16) folds away); bf16
    # loads are converted for real -- bf16 -> fp16 is a saturating conversion
    # (huge finite -> inf, NaN -> NaN) and a bf16 significand is a subset of
    # fp16's, so `x_fp16 > fp16(wrapped)` is exactly torch.gt for every bf16
    # value as long as the wrapped scalar is inside fp16's exact range (see
    # _GT_SCALAR_BF16_FP16_MIN/MAX in gt_scalar).
    half_scalar = scalar.to(tl.float16)
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    if MASKED:
        mask = off < numel
        x = tl.load(x_ptr + off, mask=mask).to(tl.float16)
        tl.store(out_ptr + off, (x > half_scalar).to(tl.int8), mask=mask)
    else:
        x = tl.load(x_ptr + off).to(tl.float16)
        tl.store(out_ptr + off, (x > half_scalar).to(tl.int8))


def _gt_scalar_flat(A, wrapped):
    numel = A.numel()
    out = torch.empty(numel, dtype=torch.int8, device=A.device)
    if numel <= _GT_SCALAR_TILE_MAX_TINY:
        # one program covering the whole tensor; a wider grid is slower here
        tile = triton.next_power_of_2(numel)
    else:
        tile = _gt_scalar_tile(numel, A.dtype)
    masked = (numel % tile) != 0
    grid = (triton.cdiv(numel, tile),)
    kernel = (
        gt_scalar_tile_kernel
        if A.dtype == torch.float32
        else gt_scalar_tile_half_kernel
    )
    kernel[grid](
        out,
        A.reshape(-1),
        wrapped,
        numel,
        TILE=tile,
        MASKED=masked,
        num_warps=4,
        isCloseMemoryAsync=False,
        buffer_size_limit=_GT_SCALAR_BSL,
        unroll_num=_GT_SCALAR_UNROLL,
    )
    return out.view(torch.bool).reshape(A.shape)


def gt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_SCALAR")
    # Flat fast path for contiguous fp16/bf16/fp32 with a finite wrapped scalar
    # (non-finite scalars, other dtypes and non-contiguous inputs keep the
    # generic pointwise_dynamic path below).
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and A.numel() > 0
    ):
        wrapped_finite, wrapped = _wrap_to_dtype(B, A.dtype)
        if wrapped_finite and (
            A.dtype != torch.bfloat16
            or _GT_SCALAR_BF16_FP16_MIN <= abs(wrapped) <= _GT_SCALAR_BF16_FP16_MAX
        ):
            os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
            os.environ["TRITONXPU_FP16_FAST"] = "1"
            res = _gt_scalar_flat(A, wrapped)
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
    logger.debug("GEMS_KUNLUNXIN GT__TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    gt_func_tensor_inplace(A, B, out0=A)
    return A


def gt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT__SCALAR")
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
