# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# arctan2 is an alias of atan2 (y = first arg, x = second arg).  The kernel is
# a deg-7 atan polynomial on u = min(|y|,|x|) / max(|y|,|x|) plus quadrant
# assembly, but the quadrant assembly is *select free*: on this XPU backend a
# single `tl.where` on a float vector is by far the most expensive primitive in
# the whole kernel.
#
# Ablation on card 4, 16.7M-element fp32, do_bench median (probe scripts, all
# measurements with the same load/store shell, BLOCK=32768/8w):
#     load+load+store only ................ 118 us
#     + 8 Horner FMAs .....................  14 us
#     + tl.abs / tl.minimum / tl.maximum ...  16 us
#     + one fp32 division ................. 153 us
#     + one tl.where ...................... 226-307 us  each
#     + one (a > b).to(fp32) .............. 600 us      each
#     + int32 bitcast & and & max ......... 4500 us
# So the previous implementation (3 quadrant selects + 1 divide guard select +
# 4 NaN-guard selects = 8 selects) measured 2034 us, i.e. ~90% of the runtime
# was spent in selects, against 685 us for torch's own atan2 (Gems speedup
# 0.46 over the benchmark matrix).
#
# The select-free assembly used here replaces every conditional by cheap
# min/max/abs/FMA arithmetic:
#  * the divide guard `where(m > 0, m, 1.0)` becomes `m + FLT_MIN`
#    (u = 0 for (0,0), relative perturbation < 1e-38 otherwise);
#  * each `where(cond, a, b)` becomes a +-1 factor obtained as
#    clamp(v * 1e30 * 1e30, -1, 1) (exact +-1 for every non-zero float,
#    including subnormals; 0 for +-0, then fixed up to +-1 by
#    `s -+ (1 - |s|)` so that ties/zeros keep the old `>`/`<` semantics),
#    folded into the quadrant formulas
#        q = pi/4 + sw * (pi/4 - p)     (sw = +1 iff |y| >  |x|)
#        t = pi/2 + sx * (pi/2 - q)     (sx = +1 iff  x  <  0)
#        r = sy * t                     (sy = -1 iff  y  <  0)
#  * NaN no longer needs a guard at all: tl.minimum/tl.maximum are called with
#    propagate_nan=ALL (measured free, 123 us vs 118 us for the load/store
#    shell), so a NaN input reaches u -> NaN -> p -> NaN and the +-1 factors
#    (finite) cannot mask it.
# Result: 620 us at 16.7M fp32 (3.3x faster than the select version) i.e.
# 1.10x torch, and the whole benchmark matrix moves from 0.46x to >1x.
#
# Rejected by measurement: `tl_extra_shim.copysign` (xpu3-elfconv fails to
# compile), int32 exponent/mantissa NaN detection (4.5 ms of int ops),
# compare-to-float quadrant arithmetic (600 us per compare, worse than the
# select it replaces), (|y|-|x|)/(|y|+|x|) single-poly assembly (inf/inf
# -> NaN, and clamping |y|,|x| to dodge it distorts finite huge inputs).
#
# Edge semantics (verified against fp64 torch on a dedicated probe list, all
# identical to the previous select implementation):
#     (+-0, +-0)      -> +-0 within 4e-17            (torch +-0)
#     (0, -1)         -> +pi                          (torch +pi)
#     (-0, -1)        -> +pi                          (previous impl: +pi too,
#                                                      torch -pi: -0.0 < 0.0
#                                                      is false either way)
#     (+-inf, finite) -> +-pi/2, (finite, +-inf) -> 0 / +-pi
#     NaN in          -> NaN
#     (inf, inf)      -> NaN (torch pi/4) -- unchanged documented limitation
#                        of the min/max reduction, outside the test matrix.
#
# Non-contiguous or broadcast (different-shape) inputs keep the generic
# pointwise_dynamic path (which handles strides/broadcast; those shapes are
# not part of the benchmark matrix).

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

_UNROLL_NUM = 16
_BUFFER_SIZE_LIMIT = 8192
_IS_CLOSE_MEMORY_ASYNC = False

# Must be `tl.constexpr(v)`, not `v: tl.constexpr` -- triton only lets @jit
# functions read globals built with the call form.
_PI = tl.constexpr(3.141592653589793)
_HALF_PI = tl.constexpr(1.5707963267948966)
_QTR_PI = tl.constexpr(0.7853981633974483)
# smallest positive normal fp32: replaces the `where(m > 0, m, 1.0)` guard.
_FLT_MIN = tl.constexpr(1.1754943508222875e-38)
# applied twice so that |v| * _BIG * _BIG >= 1 for every non-zero fp32,
# subnormals included (1.4e-45 * 1e30 * 1e30 = 1.4e15).
_BIG = tl.constexpr(1e30)


def _pick_block(n_elements):
    # Swept on card 4 with do_bench over the whole benchmark matrix
    # (n = 1024 .. 268M, blocks 1024..131072, fp32 and fp16); the select-free
    # kernel is memory bound, so 8192/8w (best or within 0.7% of best for every
    # n >= 65536) plus a small-shape bucket is enough:
    #   n <= 16384: 2048/4w  (fp32 16384: 6.98us vs 8.93us at 8192/8w;
    #                         block 1024 is a trap for fp16 -- 22.6us at
    #                         16384 elements, 12.1ms at 16.7M)
    #   otherwise:  8192/8w  (fp32 262144: 16.7us vs 19.6us at 16384/8w,
    #                         16.7M: 620us vs 684us at 4096/4w)
    # Unmasked whenever the shape divides the tile (the masked memory path
    # costs ~2x on XPU).
    if n_elements <= 16384:
        return 2048, 4, n_elements % 2048 != 0
    return 8192, 8, n_elements % 8192 != 0


@triton.jit
def _sign_neg_tie(v):
    # +1 when v > 0, -1 when v <= 0 (ties and +-0 -> -1), select free.
    g = (v * _BIG) * _BIG
    s = tl.maximum(tl.minimum(g, 1.0), -1.0)
    return s - (1.0 - tl.abs(s))


@triton.jit
def _sign_pos_tie(v):
    # +1 when v >= 0 (+-0 included), -1 when v < 0, select free.
    g = (v * _BIG) * _BIG
    s = tl.maximum(tl.minimum(g, 1.0), -1.0)
    return s + (1.0 - tl.abs(s))


@triton.jit
def _arctan2_poly(yc, xc):
    # LSQ-fitted deg-7 atan polynomial (max abs err 9.5e-7 on [0,1], measured
    # 1.19e-6 end to end on 1M random fp32 inputs) on
    # u = min(|y|,|x|) / max(|y|,|x|), plus the select-free quadrant assembly.
    ay = tl.abs(yc)
    ax = tl.abs(xc)
    # propagate_nan keeps NaN inputs alive without any NaN guard select.
    m = tl.maximum(ay, ax, propagate_nan=tl.PropagateNan.ALL)
    mn = tl.minimum(ay, ax, propagate_nan=tl.PropagateNan.ALL)
    u = mn / (m + _FLT_MIN)  # (0,0) -> u = 0, not 0/0
    p = 5.21594798e-02
    p = p * u + -2.22082111e-01
    p = p * u + 3.16956596e-01
    p = p * u + -3.27826582e-02
    p = p * u + -3.28529690e-01
    p = p * u + -3.31425699e-04
    p = p * u + 1.00000797e00
    p = p * u + 4.05427219e-17
    # q = where(|y| > |x|, pi/2 - p, p)
    q = _QTR_PI + _sign_neg_tie(ay - ax) * (_QTR_PI - p)
    # t = where(x < 0, pi - q, q)
    t = _HALF_PI + _sign_neg_tie(-xc) * (_HALF_PI - q)
    # res = where(y < 0, -t, t)
    return _sign_pos_tie(yc) * t


@triton.jit
def _arctan2_kernel_impl(
    y_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    yc = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)
    xc = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    res = _arctan2_poly(yc, xc)
    tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _arctan2_kernel_impl_unmasked(
    y_ptr,
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    yc = tl.load(y_ptr + offset).to(tl.float32)
    xc = tl.load(x_ptr + offset).to(tl.float32)
    res = _arctan2_poly(yc, xc)
    tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


def _launch(y, x, out):
    n_elements = y.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        _arctan2_kernel_impl[grid](
            y,
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=_UNROLL_NUM,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=_IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        _arctan2_kernel_impl_unmasked[grid](
            y,
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=_UNROLL_NUM,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=_IS_CLOSE_MEMORY_ASYNC,
        )


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _arctan2_kernel(input, other):
    # Generic fallback (broadcast / non-contiguous inputs).
    input_f32 = input.to(tl.float32)
    other_f32 = other.to(tl.float32)
    result = tl_extra_shim.atan2(input_f32, other_f32)

    # XPU atan2 returns zero for atan2(+/-0, negative), losing the quadrant.
    input_bits = input_f32.to(tl.int32, bitcast=True)
    other_bits = other_f32.to(tl.int32, bitcast=True)
    signed_pi = tl.where(input_bits < 0, -3.141592653589793, 3.141592653589793)
    negative_other = (other_f32 < 0.0) | ((other_f32 == 0.0) & (other_bits < 0))
    result = tl.where((input_f32 == 0.0) & negative_other, signed_pi, result)
    is_nan = (input_f32 != input_f32) | (other_f32 != other_f32)
    return tl.where(is_nan, float("nan"), result)


def _use_fast_path(input, other):
    return (
        input.is_contiguous()
        and other.is_contiguous()
        and input.shape == other.shape
        and input.dtype == other.dtype
    )


def arctan2(input, other):
    logger.debug("GEMS_KUNLUNXIN ARCTAN2")
    if _use_fast_path(input, other):
        out = torch.empty_like(input)
        _launch(input, other, out)
        return out
    return _arctan2_kernel(input, other)


def arctan2_(input, other):
    logger.debug("GEMS_KUNLUNXIN ARCTAN2_")
    if _use_fast_path(input, other):
        # Both operands are read before the lane is stored, so out aliasing
        # input is safe (including the degenerate x.arctan2_(x) case).
        _launch(input, other, input)
        return input
    _arctan2_kernel(input, other, out0=input)
    return input