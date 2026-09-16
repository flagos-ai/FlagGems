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

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag


@triton.jit
def _polar_fused_kernel(a_ptr, t_ptr, out64, n, BLOCK: tl.constexpr):
    # One fused kernel: r = abs*cos(angle), im = abs*sin(angle), then pack
    # each (r, im) f32 pair into one u64 slot of the complex storage
    # (little-endian: low 32 bits = real, high 32 bits = imag), so both the
    # loads and the store are dense stride-1.  Any interleaved (stride-2)
    # or gathered access is not handled by the XPU block-DMA engine and
    # collapses to per-element scalar access (~40s/16M elts through the
    # pointwise rank-2 path; ~7ms gathered; ~40ms strided), and the ATen
    # torch.complex interleave lowers to two stride-2 copies.
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(a_ptr + i, mask=m)
    t = tl.load(t_ptr + i, mask=m)
    r = (x * tl.cos(t)).to(tl.int32, bitcast=True).to(tl.uint32).to(tl.uint64)
    im = (x * tl.sin(t)).to(tl.int32, bitcast=True).to(tl.uint32).to(tl.uint64)
    tl.store(out64 + i, (im << 32) | r, mask=m)


@triton.jit
def _polar_pack_f64_kernel(real, imag, out, n2, BLOCK: tl.constexpr):
    # float64 path: a complex128 element is two u64 slots (no 128-bit pack
    # on this backend), so interleave the two u64 streams with a flat
    # kernel: dense u64 store, value selected by lane parity (loads at
    # j>>1 gather, but float64 is unreachable on most XPU stacks where
    # f64 silently downcasts -- correctness over speed here).
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < n2
    even = (j & 1) == 0
    half = j >> 1
    r = tl.load(real + half, mask=m & even, other=0)
    im = tl.load(imag + half, mask=m & (~even), other=0)
    tl.store(out + j, tl.where(even, r, im), mask=m)


def polar(abs, angle):
    logger.debug("GEMS_KUNLUNXIN POLAR")
    if abs.dtype == torch.float32 and abs.numel() > 0:
        if not abs.is_contiguous():
            abs = abs.contiguous()
            angle = angle.contiguous()
        out = torch.empty(abs.shape, dtype=torch.complex64, device=abs.device)
        n = abs.numel()

        BLOCK = 8192
        grid = (triton.cdiv(n, BLOCK),)
        with torch_device_fn.device(abs.device):
            _polar_fused_kernel[grid](
                abs, angle, out.view(torch.int64), n, BLOCK=BLOCK, num_warps=8
            )
        return out

    # float64 (two u64 per element) and empty tensors: two-phase compute +
    # a u64 interleave kernel (bit-pattern preserving, no ATen fallback).
    real = torch.empty(abs.shape, dtype=abs.dtype, device=abs.device)
    imag = torch.empty(abs.shape, dtype=abs.dtype, device=abs.device)

    polar_kernel(abs, angle, out0=real, out1=imag)

    cplx_dtype = torch.complex128 if abs.dtype == torch.float64 else torch.complex64
    out = torch.empty(abs.shape, dtype=cplx_dtype, device=abs.device)
    if abs.dtype == torch.float64 and abs.numel() > 0:
        n2 = 2 * out.numel()
        BLOCK = 8192
        grid = (triton.cdiv(n2, BLOCK),)
        with torch_device_fn.device(abs.device):
            _polar_pack_f64_kernel[grid](
                real.view(torch.int64),
                imag.view(torch.int64),
                out.view(torch.int64),
                n2,
                BLOCK=BLOCK,
                num_warps=8,
            )
    return out
