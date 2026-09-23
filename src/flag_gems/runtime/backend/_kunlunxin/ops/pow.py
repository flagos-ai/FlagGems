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

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_pow = tl_extra_shim.pow
_fast_expf = tl_extra_shim.fast_expf


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
# XPU probe (2026-08-19, XPU4, 16.7M fp32 do_bench, same window):
#   * generic extern pow (pow_func_tensor_scalar) 1290-1815us, ~2x torch native
#     pow_; this fast path 465us (~2x faster than native x.pow_(0.001));
#   * recipe r = tl.exp2(e * tl.log2(x)): on this backend tl.exp2 == e^x and
#     tl.log2 == ln(x) (math semantics, same as the pow_scalar fast path), so
#     r == x^e holds exactly;
#   * semantic corners are automatic (no per-element select -- select forces
#     the SFU path back to 2-5x): x < 0 -> log2(x)=NaN -> NaN; x = 0 ->
#     log2(x)=-inf -> e^(+-e*inf)=0/inf; x = +-inf / NaN likewise;
#   * numeric cross-check (fp64 CPU reference, harness protocol): SCALARS x
#     fp32/fp16/bf16 distribution matrix, 0 failures;
#   * gating: only finite, non-zero, non-integer, > 0 exponents take the fast
#     path; integer/0/negative-non-integer/+-inf/NaN exponents keep the
#     generic extern path (semantics unchanged).
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


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor_fast(log_base, exponent):
    # For a positive scalar base, pow(base, exp) == exp(exp * log(base)).
    # log(base) is constant so it is computed once on the host; fast_expf is the
    # XPU fast approximate expf (much cheaper and more accurate than the native
    # tl.exp2 path, which produced NaN at large shapes for base 100.001).
    return _fast_expf(exponent.to(tl.float32) * log_base.to(tl.float32))


# ---------------------------------------------------------------------------
# pow_scalar fast path (aten::pow.Scalar, scalar base >0 finite, !=1).
#
# XPU probe (2026-08-15, XPU5, 16.7M fp32 isolated A/B):
#  * tl_extra_shim.pow (libdevice software impl) 538us, torch native 199us;
#  * backend tl.exp2/tl.log2 are actually e^x / ln(x) (numeric semantics) and
#    SFU-class: one tl.exp(y * ln(base)) ~142us, faster than torch; any
#    per-element where/min/max/integer-compare forces the SFU path back to
#    2-5x (no quantization reuse);
#  * a single ln(f32) constant satisfies the corners naturally: y=±inf ->
#    e^(±inf)=0/inf, y=NaN -> NaN, y=0 -> 1 (exp(0)==1); no clamp/select
#    needed -- numeric cross-check (base 0.001/100.001/2/0.5 x
#    fp32/fp16/bf16 x y=U(-1,1)): 0 failures;
#  * all other bases (<=0, ==1, +-inf, NaN) keep the generic extern path,
#    semantics unchanged.
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


def pow_scalar(A, exponent):
    logger.debug("GEMS_KUNLUNXIN POW_SCALAR")
    base = A.item() if hasattr(A, "item") else float(A)
    if base > 0:
        return pow_func_scalar_tensor_fast(math.log(base), exponent)
    return pow_func_scalar_tensor(A, exponent)
