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

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# asin(x) fast path (race: arcsin/asin, 2026-09-04): the previous kernel
# (tl.sqrt + tl.where sign reconstruction) was select/mask-bound: the
# x<0 where-compile emits an ~100-instruction i1->i32 mask extraction
# (llvm.xpu.vvor_f_mh_rn) costing ~0.245 ms at 16.7M elements, 2.2x the
# whole torch.asin time. This rewrite replaces both with XPU-friendly ops:
#
#   asin(x) = sgn(x) * (pi/2 - 2*s*P(t)),  t=(1-|x|)/2 in [0,0.5],
#   s = t*rsqrt(t+eps) ~ sqrt(t),  P = degree-8 LSQ fit of asin(s)/s
#   sgn(x) = 1 - 2*min(1, max(0, -x*2^126))      (no select/compare)
#
#   - rsqrt: xpu.rsqrt lowers to the inline hardware SFU op
#     (tt.extern_elementwise _ZN3xpu6rsqrtfEf, ~0.67x the cost of the
#     software-expanded tl.sqrt chain). The +1e-30 bias keeps s=0 exactly
#     at t=0 (x = +-1): 0*rsqrt(1e-30) = 0, so asin(+-1) = +-pi/2 like
#     torch. The bias is a bit-exact no-op for every other representable
#     t (no fp32 value falls in (1, 1+2e-30]).
#   - sign: pure min/max/fma, no ordered-compare -> no mask extraction.
#     The 2^126 scale maps x<0 -> m=1, x>0 -> m=0, and the min(1,..)
#     clamps the product; for normal |x| the result is exactly +-1.
#     (min/max on XPU drop NaN, which is fine: q is already NaN for
#     |x|>1, so r = NaN*(1-2m) = NaN.)
#   - poly2 = 2*P folds the *2.0 into the coefficients (one mul saved).
#
# Accuracy: fp32 Horner keeps |asin(x)-ref| <= 4e-5 on [-1,1] (measured
# 3.70e-5 on randn), inside atol 1e-4 + rtol (1.3e-6 / 1e-3 / 1e-3).
# Coeffs (fp32-rounded, Horner order high -> low), shared with acos.py:
#   [-246.59942627, 530.01574707, -470.57415771, 222.85160828, -60.52576828,
#     9.49576759, -0.72389036, 0.19823363, 0.99959993]
# Known limit: bf16 output stores pay a software-expanded f32->bf16
# convert (~0.07 ms at 16.7M elements, ~2x the fp32 store cost); fp16 and
# fp32 are unaffected.
MIN_BLOCK = 2048
# unroll 8 beats 16 on the official unary matrix (acos family sweep,
# arccos/arccos_ closure 2026-08-16: u16 -> u8 gained ~4%).
UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into a few unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~6 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    # Larger blocks win once there are >= ~128 programs (16.7M+ elements);
    # mid sizes prefer 32768 (>=16 programs); small sizes 8192; tiny shapes
    # are launch-bound and use the 2048/4w masked kernel.
    if n_elements >= (1 << 24) and n_elements % 131072 == 0:
        return 131072, 8, False
    if n_elements >= (1 << 19) and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= (1 << 14) and n_elements % 8192 == 0:
        return 8192, 8, False
    if n_elements <= 16384:
        return 2048, 4, True
    if n_elements % 16384 == 0:
        return 16384, 8, False
    return 16384, 8, True


@triton.jit
def _asin_body(x):
    t = 0.5 - 0.5 * tl.abs(x)
    # |x| > 1 makes t < 0 -> rsqrt(NaN) -> NaN propagates out, matching torch.
    s = t * xpu.rsqrt(t + 1e-30)
    p = -493.19885254
    p = p * t + 1060.03149414
    p = p * t + -941.14831543
    p = p * t + 445.70321655
    p = p * t + -121.05153656
    p = p * t + 18.99153519
    p = p * t + -1.44778073
    p = p * t + 0.39646727
    p = p * t + 1.99919987
    q = 1.5707964 - s * p
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    return q * (1.0 - 2.0 * m)


@triton.jit
def asin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    r = _asin_body(x)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def asin_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    r = _asin_body(x)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        asin_kernel[grid](
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
        asin_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def asin(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ASIN")
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


def asin_(x):
    logger.debug("GEMS_KUNLUNXIN ASIN_")
    xc = x.contiguous()
    _launch(xc, xc)
    if xc.data_ptr() != x.data_ptr():
        x.copy_(xc.view(x.shape))
    return x


def asin_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ASIN OUT")
    return asin(x, out=out)