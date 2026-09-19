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
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# The XPU backend has two f32->bf16 store lowerings (TritonXPUToLLVM
# LoadStoreOpToLLVM.cpp): the default one packs every two 512-bit f32 vectors
# through a per-lane vand/vadd rounding chain followed by two masked
# SCATTER_MH ops, which costs ~+150us at n = 16.7M (a 1.35x slowdown of the
# whole op vs the f32 store). TRITONXPU_BF16_FAST selects the vendor
# vstore2_lm device call instead (a single hardware-rounded 16->bf16 pack +
# store per two vectors, ~70us faster at 16.7M, bit-exact RNE on the same
# outputs; used by the vendor's own sglang kernels, e.g.
# third_party/xpu/test/sglang/qwen3_next/test_l2norm_fwd_kernel.py). The flag
# is a compile-time choice read when a kernel is compiled, so it is set here
# (before any kunlunxin kernel compiles at import time); a later assignment
# in _launch would not apply if the kernel is JIT-cached. It only changes the
# f32->bf16 store lowering (bit-exact) and leaves every other op untouched.
os.environ.setdefault("TRITONXPU_BF16_FAST", "1")

# asin(x) = sign(x) * (pi/2 - 2*sqrt(t)*P(t)) with t = (1-|x|)/2 in [0, 0.5]
# and P(t) = asin(sqrt(t))/sqrt(t).  P is analytic on t in [0, 0.5] (its only
# singularity is the branch point at t = 1), so its Chebyshev coefficients in
# w = 4t - 1 decay geometrically (rho = 3 + 2*sqrt(2) ~ 5.83); the degree-4
# truncation leaves a true fit error < 1e-6, and the fp32 Horner (the backend
# fuses the mul+add into a single 512-bit vmacf) keeps
# |asin(x) - asin_ref| <= 3e-5 on the full fp32 domain [-1, 1], well inside
# the test tolerance (atol 1e-4 + rtol 1.3e-6).
#
# Perf notes (XPU4, measured with the official unary matrix, n = 16.7M,
# torch.asin = 228us):
#   * XPU has a 512-bit vsqrtf but no vector select: a t-form kernel with
#     tl.where(x < 0, -r, r) costs ~485us, of which ~216us is the compare +
#     subtract + select chain (each arith.select is not 512-bit-vectorized).
#   * The sign is therefore applied with pure min/max (vnminf/vnmaxf are
#     512-bit and need no compare/mask): with S = 2^126,
#         m  = min(1, max(0, -x*S))   (exactly 1 for x<0, 0 for x>=0)
#         r  = w * (1 - 2*m)          w = asin(|x|) >= 0 for |x| <= 1
#     so r = +-w with no select, ~4 cheap vector ops (~+15us vs the pure-int
#     sign-bit OR, which is NOT usable: the f32<->i32 bitcast of a bf16-loaded
#     value makes the backend scalarize the store path through local memory
#     and corrupts every 16-lane vector past the first one for bf16 inputs,
#     whereas min/max produces bit-exact results on all dtypes).
#   * sqrt is computed as t*rsqrt(t+1e-30): xpu.rsqrt lowers to the inline
#     SFU op (~0.67x the cost of the software-expanded tl.sqrt chain); the
#     +1e-30 bias keeps s = 0 exactly at t = 0 (x = +-1), so
#     asin(+-1) = +-pi/2 like torch. Polynomial-removing poly fits (deg-8+
#     in x^2, rationals (n,n) up to 8) converge ~n^-2 and cannot reach 1e-4
#     below ~80 terms, so the sqrt cannot be removed entirely.
#   * The bf16 store goes through TRITONXPU_BF16_FAST (see the note below
#     the imports), which cuts the bf16 f32->bf16 pack+store from ~150us to
#     ~36us at n = 16.7M (measured 348us -> 242us for the whole op at that
#     size; the remaining time is SFU + memory).
#   * 32768-lane tiles measured fastest at 4.2M/16.7M, 65536 ties at 67M;
#     8192 for the mid sizes and 2048-masked for the launch-bound small
#     shapes (masked memory path costs ~2x, so unmasked runs whenever the
#     shape divides the tile exactly).
MIN_BLOCK = 2048
UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    if n_elements >= 1048576 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 8192 and n_elements % 8192 == 0:
        return 8192, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 8192, 8, True


@triton.jit
def arcsin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    # |x| > 1 makes t < 0 -> rsqrt(NaN) -> NaN propagates out, matching torch.
    s = t * xpu.rsqrt(t + 1e-30)
    p = 0.0962260290980339
    p = p * t + 0.008193825371563435
    p = p * t + 0.08233591169118881
    p = p * t + 0.1661728471517563
    p = p * t + 1.0000052452087402
    v = (s * p) * 2.0
    w = 1.5707964 - v
    # sign via min/max (no select; see the header comment): m = (x < 0) ? 1 : 0
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    r = w * (1.0 - 2.0 * m)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def arcsin_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    s = t * xpu.rsqrt(t + 1e-30)
    p = 0.0962260290980339
    p = p * t + 0.008193825371563435
    p = p * t + 0.08233591169118881
    p = p * t + 0.1661728471517563
    p = p * t + 1.0000052452087402
    v = (s * p) * 2.0
    w = 1.5707964 - v
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    r = w * (1.0 - 2.0 * m)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        arcsin_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        arcsin_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def arcsin(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN")
    xc = x.contiguous()
    if out is None:
        out = torch.empty_like(xc)
        _launch(xc, out)
        return out
    oc = out.contiguous()
    _launch(xc, oc)
    if oc.data_ptr() != out.data_ptr():
        out.copy_(oc.view(out.shape))
    return out


def arcsin_(x):
    logger.debug("GEMS_KUNLUNXIN ARCSIN_")
    xc = x.contiguous()
    _launch(xc, xc)
    if xc.data_ptr() != x.data_ptr():
        x.copy_(xc.view(x.shape))
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSIN OUT")
    return arcsin(x, out=out)
