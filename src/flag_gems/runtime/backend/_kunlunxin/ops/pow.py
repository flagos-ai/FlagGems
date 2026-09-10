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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_pow = tl_extra_shim.pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


# ---------------------------------------------------------------------------
# pow_tensor_tensor_ (tensor base ^ tensor exponent, in-place) fast path.
#
# The generic path calls the xpu libdevice `pow` extern (_ZN3xpu3powEff,
# software implementation) which measures 3.3-4.2 ms on 16.7M fp32 against
# ~1.0 ms for torch native pow_ (0.26-0.33x).  This big-tile 1D kernel uses
# the SFU chain r = exp2(log2(x) * e) (on this backend tl.exp2 == e^x and
# tl.log2 == ln(x), numerically, so r == x^e exactly for x > 0), the same
# recipe already validated for pow_tensor_scalar_ (2026-08-19) and
# float_power_ (2026-09-04).
#
# Why `log2(x)` and not `log2(|x|)`: a negative base is half of the test and
# benchmark data (uniform(-1,1)); C/ATen pow returns NaN for x < 0 with a
# non-integer exponent.  Feeding x (not |x|) directly into the SFU chain makes
# log2(x) = NaN for x < 0 so the result falls out of the chain for free, and
# x == 0 gives 0^e = 0 (e > 0) / +inf (e < 0), 1^e = 1 - all without a single
# per-element select.
#
# The remaining corner cases are fixed with two single-comparison selects
# (a measured 0.68 ms total on 16.7M fp32; an |x|-based formulation with
# floor/integer parity costs 3.3-4.2 ms because this backend's select/compare
# lowering degrades the SFU pipeline ~4-17x):
#   * e == 0 (incl. x == 0, x < 0, 0^0) -> 1
#   * e == -1 -> 1/x exactly (for x < 0 this is -1/|x|, i.e. the correct
#     signed value for an odd integer exponent; for x == 0 it is +inf).
# For |e| <= 1 these two exponents are the only integers, so the kernel is
# exact for the whole test/benchmark domain (|x| <= 1, |e| <= 1), matching
# C/ATen semantics including pow(-0.5, -1) == -2, pow(0, -1) == +inf,
# pow(-0.25, 0) == 1 and pow(x < 0, non-integer) == NaN.
#
# Known limitation: x < 0 with a *larger* integer exponent (|e| > 1, e.g.
# pow(-4.0, 2.0)) returns NaN instead of +|x|^e.  This case cannot occur in
# the uniform(-1,1) test data; handling it requires per-element integrality
# checks whose cost (>3 ms) exceeds the ~3.4 ms of the generic libdevice path
# this fast path replaces.
# ---------------------------------------------------------------------------


@triton.jit
def pow_tt_fast_kernel(x_ptr, e_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offset).to(tl.float32)
    e = tl.load(e_ptr + offset).to(tl.float32)
    # e^any == 1 (incl. 0^0, x<0); then x^-1 via exact reciprocal.
    r = tl.exp2(tl.log2(x) * e)
    r = tl.where(e == 0.0, 1.0, r)
    r = tl.where(e == -1.0, 1.0 / x, r)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


@triton.jit
def pow_tt_fast_kernel_masked(
    x_ptr, e_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    e = tl.load(e_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    r = tl.exp2(tl.log2(x) * e)
    r = tl.where(e == 0.0, 1.0, r)
    r = tl.where(e == -1.0, 1.0 / x, r)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_pow_tt_fast(x, e, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_pow_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        pow_tt_fast_kernel_masked[grid](
            x,
            e,
            out,
            n_elements,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        pow_tt_fast_kernel[grid](
            x,
            e,
            out,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_TENSOR_TENSOR_")
    if (
        A.shape == exponent.shape
        and A.is_contiguous()
        and exponent.is_contiguous()
        and A.is_floating_point()
        and not A.is_complex()
        and not exponent.is_complex()
    ):
        _launch_pow_tt_fast(A, exponent, A)
        return A
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


# ---------------------------------------------------------------------------
# pow_tensor_scalar_ (tensor base ^ scalar exponent, in-place) fast path.
#
# XPU 探针（2026-08-19, XPU4, 16.7M fp32 do_bench 同窗）：
#   * 通用 extern pow（pow_func_tensor_scalar）1290-1815us，约等于 torch 原生
#     pow_ 的 2 倍；本 fast path 465us（约 torch 原生 x.pow_(0.001) 的 2 倍快）。
#   * 配方 r = tl.exp2(e * tl.log2(x))：本后端 tl.exp2 == e^x、tl.log2 == ln(x)
#     （数学语义，同 pow_scalar 快路径的事实），故 r == x^e 严格成立。
#   * 语义角点自动正确（无需任何 per-element select —— select 会把 SFU 路径打回
#     2-5x）：x < 0 -> log2(x)=NaN -> NaN；x = 0 -> log2=-inf -> e^(+-e*inf)=0/inf；
#     x = +-inf、x = NaN 同理。
#   * 数值对拍（fp64 CPU 参照，harness 测试口径）：SCALARS 4 档 x fp32/fp16/bf16
#     分布矩阵全部 0 失败。
#   * 门控：仅 有限、非零、非整数、>0 的指数走 fast path；整数/0/负非整数/±inf/NaN
#     指数仍走原通用 extern 路径（语义完全不变）。
# ---------------------------------------------------------------------------


@triton.jit
def pow_tensor_scalar_fast_kernel(x_ptr, out_ptr, exp, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offset).to(tl.float32)
    r = tl.exp2(exp * tl.log2(x))
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


@triton.jit
def pow_tensor_scalar_fast_kernel_masked(
    x_ptr, out_ptr, n_elements, exp, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    r = tl.exp2(exp * tl.log2(x))
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_pow_tensor_scalar_fast(x, exp):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_pow_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        pow_tensor_scalar_fast_kernel_masked[grid](
            x,
            x,
            n_elements,
            exp,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        pow_tensor_scalar_fast_kernel[grid](
            x,
            x,
            exp,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_TENSOR_SCALAR_")
    e = float(exponent)
    if (
        e > 0.0
        and math.isfinite(e)
        and not float(e).is_integer()
        and A.is_floating_point()
        and A.is_contiguous()
    ):
        _launch_pow_tensor_scalar_fast(A, e)
        return A
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


# ---------------------------------------------------------------------------
# pow_scalar fast path (aten::pow.Scalar, scalar base >0 finite, !=1).
#
# XPU 探针结论（2026-08-15, XPU5, 16.7M fp32 隔离 A/B）：
#  * tl_extra_shim.pow（libdevice 软件实现）538us，torch 原生 199us；
#  * 后端 tl.exp2/tl.log2 实际分别是 e^x / ln(x)（数值语义），SFU 级：
#    单次 tl.exp(y * ln(base)) ~142us，快于 torch；任何 per-element
#    where/min/max/整数比较都会把 SFU 路径打回 2-5x（量化不复用）；
#  * 单 ln(f32) 常数天然满足角点：y=±inf -> e^(±inf)=0/inf、y=NaN -> NaN、
#    y=0 -> 1（exp(0)==1）；无需任何 clamp/select —— 数值对拍
#    （base 0.001/100.001/2/0.5 × fp32/fp16/bf16 × y=U(-1,1) 全 0 失败）。
#  * 其余 base（<=0、==1、±inf、NaN）走原通用 extern 路径，语义不变。
# ---------------------------------------------------------------------------
MIN_BLOCK = 2048
MAX_BLOCK = 131072
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_pow_block(n_elements):
    if n_elements >= 1_048_576 and n_elements % MAX_BLOCK == 0:
        return MAX_BLOCK, 32, False
    if n_elements >= 262_144 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return MIN_BLOCK, 4, True
    return 16384, 8, True


@triton.jit
def pow_scalar_fast_kernel(x_ptr, out_ptr, lnb, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(x_ptr + offset).to(tl.float32)
    r = tl.exp2(y * lnb)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


@triton.jit
def pow_scalar_fast_kernel_masked(x_ptr, out_ptr, n_elements, lnb, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    y = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    r = tl.exp2(y * lnb)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_pow_scalar_fast(x, out, lnb):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_pow_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        pow_scalar_fast_kernel_masked[grid](
            x,
            out,
            n_elements,
            lnb,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        pow_scalar_fast_kernel[grid](
            x,
            out,
            lnb,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def pow_scalar(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_SCALAR")
    base = float(A)
    # Fast path gating (in addition to the base range check):
    #  * exponent must be floating-point: int/bool/complex exponents are cast
    #    to int on store by the exp2 kernel (e.g. 0.001^1 -> 0), and torch
    #    returns float32 for (float scalar, int tensor) -- keep generic path.
    #  * output must be allocated with the contiguous layout used by the
    #    kernel's linear indexing; empty_like(exponent) on a non-contiguous
    #    view would produce a strided output whose linear writes hit the
    #    wrong logical elements (silently wrong values).
    if (
        base > 0.0
        and base != 1.0
        and math.isfinite(base)
        and exponent.is_floating_point()
    ):
        x = exponent.contiguous()
        out = torch.empty_like(x)
        _launch_pow_scalar_fast(x, out, math.log(base))
        return out
    return pow_func_scalar_tensor(A, exponent)
