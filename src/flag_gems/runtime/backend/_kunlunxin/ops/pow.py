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

from flag_gems.utils import triton_lang_extension as ext, tl_extra_shim

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


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_TENSOR_TENSOR_")
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
            x, x, n_elements, exp, BLOCK=block_size, num_warps=num_warps,
            unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        pow_tensor_scalar_fast_kernel[grid](
            x, x, exp, BLOCK=block_size, num_warps=num_warps,
            unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
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
def pow_scalar_fast_kernel_masked(
    x_ptr, out_ptr, n_elements, lnb, BLOCK: tl.constexpr
):
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
            x, out, n_elements, lnb, BLOCK=block_size, num_warps=num_warps,
            unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        pow_scalar_fast_kernel[grid](
            x, out, lnb, BLOCK=block_size, num_warps=num_warps,
            unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def pow_scalar(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_SCALAR")
    base = float(A)
    if base > 0.0 and base != 1.0 and math.isfinite(base):
        x = exponent.contiguous()
        out = torch.empty_like(exponent)
        _launch_pow_scalar_fast(x, out, math.log(base))
        return out
    return pow_func_scalar_tensor(A, exponent)
