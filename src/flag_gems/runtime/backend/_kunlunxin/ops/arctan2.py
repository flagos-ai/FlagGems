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

_PI = tl.constexpr(3.141592653589793)
_HALF_PI = tl.constexpr(1.5707963267948966)
_QTR_PI = tl.constexpr(0.7853981633974483)
_FLT_MIN = tl.constexpr(1.1754943508222875e-38)
_BIG = tl.constexpr(1e30)


def _pick_block(n_elements):
    if n_elements <= 16384:
        return 2048, 4, n_elements % 2048 != 0
    return 8192, 8, n_elements % 8192 != 0


@triton.jit
def _sign_neg_tie(v):
    g = (v * _BIG) * _BIG
    s = tl.maximum(tl.minimum(g, 1.0), -1.0)
    return s - (1.0 - tl.abs(s))


@triton.jit
def _sign_pos_tie(v):
    g = (v * _BIG) * _BIG
    s = tl.maximum(tl.minimum(g, 1.0), -1.0)
    return s + (1.0 - tl.abs(s))


@triton.jit
def _arctan2_poly(yc, xc):
    ay = tl.abs(yc)
    ax = tl.abs(xc)
    m = tl.maximum(ay, ax, propagate_nan=tl.PropagateNan.ALL)
    mn = tl.minimum(ay, ax, propagate_nan=tl.PropagateNan.ALL)
    u = mn / (m + _FLT_MIN)
    p = 5.21594798e-02
    p = p * u + -2.22082111e-01
    p = p * u + 3.16956596e-01
    p = p * u + -3.27826582e-02
    p = p * u + -3.28529690e-01
    p = p * u + -3.31425699e-04
    p = p * u + 1.00000797e00
    p = p * u + 4.05427219e-17
    q = _QTR_PI + _sign_neg_tie(ay - ax) * (_QTR_PI - p)
    t = _HALF_PI + _sign_neg_tie(-xc) * (_HALF_PI - q)
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
        _launch(input, other, input)
        return input
    _arctan2_kernel(input, other, out0=input)
    return input
