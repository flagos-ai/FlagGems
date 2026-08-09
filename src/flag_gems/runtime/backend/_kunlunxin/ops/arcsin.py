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

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# asin(x) fast path: replace the XPU software atan2/acosf external calls
# (measured ~0.18x torch on the official unary matrix) with a pure
# polynomial in the stable region.
#
#   asin(x) = pi/2 - acos(x)
#
# acos(x) = 2 * asin(sqrt((1-|x|)/2)) for x >= 0, pi - acos(-x) for x < 0;
# with t = (1-|x|)/2 in [0, 0.5], s = sqrt(t), asin(s)/s = P(t) with P the
# same degree-8 LSQ fit as the acos family (acos.py). fp32 Horner keeps
# |asin(x) - asin_ref| <= 4e-5 on the full fp32 domain [-1, 1] (numpy fp32
# emulation, no-FMA), inside the test tolerance (atol 1e-4 + rtol 1.3e-6).
# NaN/Inf semantics: |x| > 1 makes t < 0 -> sqrt(NaN) -> the single
# x < 0 where-chain keeps NaN like torch (|x|>1 gives NaN, NaN input gives
# NaN, ±1 give ±pi/2 exactly). Only ONE ordered comparison (x < 0) remains;
# compound boolean compares ((x<=1)&(x>=-1)) compile to the slow XPU path.
# Coeffs (fp32-rounded, Horner order high -> low), shared with acos.py:
#   [-246.59942627, 530.01574707, -470.57415771, 222.85160828, -60.52576828,
#     9.49576759, -0.72389036, 0.19823363, 0.99959993]
MIN_BLOCK = 2048
# unroll 8 beats 16 on the official unary matrix (acos family sweep on XPU2,
# arccos/arccos_ closure 2026-08-16: u16 -> u8 gained ~4%： 0.6834 -> 0.7170;
# probe on this operator's own matrix stays within noise).
UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into a few unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~4 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    if n_elements >= 16384 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


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
    # |x| > 1 makes t < 0 -> sqrt(NaN) -> NaN propagates out, matching torch.
    s = tl.sqrt(t)
    p = -246.59942627
    p = p * t + 530.01574707
    p = p * t + -470.57415771
    p = p * t + 222.85160828
    p = p * t + -60.52576828
    p = p * t + 9.49576759
    p = p * t + -0.72389036
    p = p * t + 0.19823363
    p = p * t + 0.99959993
    v = (s * p) * 2.0
    # asin(x) = pi/2 - acos(x); acos(x) = x<0 ? pi-v : v  (v == acos(|x|))
    r = tl.where(x < 0.0, v - 1.5707964, 1.5707964 - v)
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
    s = tl.sqrt(t)
    p = -246.59942627
    p = p * t + 530.01574707
    p = p * t + -470.57415771
    p = p * t + 222.85160828
    p = p * t + -60.52576828
    p = p * t + 9.49576759
    p = p * t + -0.72389036
    p = p * t + 0.19823363
    p = p * t + 0.99959993
    v = (s * p) * 2.0
    r = tl.where(x < 0.0, v - 1.5707964, 1.5707964 - v)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        arcsin_kernel[grid](
            x, out, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps,
            unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        arcsin_kernel_unmasked[grid](
            x, out, BLOCK_SIZE=block_size, num_warps=num_warps,
            unroll_num=UNROLL_NUM, buffer_size_limit=BUFFER_SIZE_LIMIT,
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