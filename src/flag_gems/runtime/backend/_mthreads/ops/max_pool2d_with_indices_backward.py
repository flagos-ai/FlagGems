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

from flag_gems.ops.max_pool2d_with_indices import (
    max_pool2d_with_indices_backward as default_max_pool2d_with_indices_backward,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# General kernel: any (kernel, stride, padding, dilation). Flat 1D grid over
# N*C*H*W, per-lane decomposition, full (kH x kW) masked scan of candidate
# windows, FP32 accumulation, implicit cast to the output dtype on store.
# ---------------------------------------------------------------------------
@triton.jit
def _maxpool2d_bwd_general_kernel(
    grad_ptr,  # (N*C, H_out, W_out)
    idx_ptr,  # (N*C, H_out, W_out) int64 flat input indices
    out_ptr,  # (N*C, H*W)
    total_n,  # N*C*H*W total input elements
    W: tl.constexpr,
    total_in: tl.constexpr,  # H*W per image
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    pH: tl.constexpr,
    pW: tl.constexpr,
    dH: tl.constexpr,
    dW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    m = flat < total_n
    nch = flat // total_in
    t = flat - nch * total_in
    h = t // W
    w = t - h * W

    grad_base = grad_ptr + nch * (H_out * W_out)
    idx_base = idx_ptr + nch * (H_out * W_out)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kh in tl.static_range(kH):
        oh_num = h + pH - kh * dH
        oh = oh_num // sH
        okh = (oh_num % sH == 0) & (oh_num >= 0) & (oh < H_out)
        for kw in tl.static_range(kW):
            ow_num = w + pW - kw * dW
            ow = ow_num // sW
            okw = (ow_num % sW == 0) & (ow_num >= 0) & (ow < W_out)
            valid = m & okh & okw
            off = oh * W_out + ow
            idx_v = tl.load(idx_base + off, mask=valid, other=-1)
            g = tl.load(grad_base + off, mask=valid, other=0.0)
            acc += tl.where(idx_v == t, g, 0.0)
    tl.store(out_ptr + flat, acc, mask=m)


# ---------------------------------------------------------------------------
# Fast kernels specialized to (kH,kW,sH,sW,pH,pW,dH,dW) == (3,3,2,2,1,1,1,1),
# the exact configuration of all five timed benchmark shapes.
#
# For this config the valid output row for input row h satisfies
#   2*oh in {h-1, h, h+1}   <=>   |h - 2*oh| <= 1
# with candidates oh_lo = (h-1)>>1 and oh_hi = oh_lo+1. The 3x3=9 masked
# scan collapses to 4 explicit candidate checks. oh_hi is always >= 0 and
# always satisfies 2*oh_hi >= h-1, so its guards reduce to oh_hi < H_out;
# only oh_lo needs the full parity/bounds guard (same for w). When the grid
# divides the total element count exactly (all timed shapes do), the global
# boundary mask is dropped entirely (EXACT specialization).
# ---------------------------------------------------------------------------
@triton.jit
def _maxpool2d_bwd_fast_kernel(
    grad_ptr,  # (N*C, H_out, W_out)
    idx_ptr,  # (N*C, H_out, W_out) int64 flat input indices
    out_ptr,  # (N*C, H*W)
    total_n,  # N*C*H*W total input elements
    W: tl.constexpr,
    total_in: tl.constexpr,  # H*W per image
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    flat = pid * BLOCK + tl.arange(0, BLOCK)
    nch = flat // total_in
    t = flat - nch * total_in
    h = t // W
    w = t - h * W

    grad_base = grad_ptr + nch * (H_out * W_out)
    idx_base = idx_ptr + nch * (H_out * W_out)

    oh_lo = (h - 1) >> 1
    oh_hi = oh_lo + 1
    ow_lo = (w - 1) >> 1
    ow_hi = ow_lo + 1

    h1 = h - 1
    w1 = w - 1
    hm_lo = (oh_lo >= 0) & (oh_lo < H_out) & ((oh_lo << 1) >= h1)
    hm_hi = oh_hi < H_out
    wm_lo = (ow_lo >= 0) & (ow_lo < W_out) & ((ow_lo << 1) >= w1)
    wm_hi = ow_hi < W_out

    off_lo = oh_lo * W_out
    off_hi = oh_hi * W_out

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    if EXACT:
        valid = hm_lo & wm_lo
        off = off_lo + ow_lo
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = hm_hi & wm_lo
        off = off_hi + ow_lo
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = hm_lo & wm_hi
        off = off_lo + ow_hi
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = hm_hi & wm_hi
        off = off_hi + ow_hi
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        tl.store(out_ptr + flat, acc)
    else:
        m = flat < total_n
        valid = m & hm_lo & wm_lo
        off = off_lo + ow_lo
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = m & hm_hi & wm_lo
        off = off_hi + ow_lo
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = m & hm_lo & wm_hi
        off = off_lo + ow_hi
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = m & hm_hi & wm_hi
        off = off_hi + ow_hi
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        tl.store(out_ptr + flat, acc, mask=m)


# ---------------------------------------------------------------------------
# Fast kernel for small images (one 2D tile covers the whole image):
# h/w come from block indices so there are no per-lane divisions and all
# lanes of a program share one image's index/grad planes (L1-resident).
# Each program processes IMG consecutive images via a static unrolled loop;
# the whole spatial setup (t, oh_lo, ow_lo, masks, offsets) is loop-invariant
# and hoisted, so per-element setup cost shrinks by IMG.
# ---------------------------------------------------------------------------
@triton.jit
def _maxpool2d_bwd_fast_tile_kernel(
    grad_ptr,  # (N*C, H_out, W_out)
    idx_ptr,  # (N*C, H_out, W_out) int64 flat input indices
    out_ptr,  # (N*C, H*W)
    num_nc,  # N*C
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    IMG: tl.constexpr,
):
    pid = tl.program_id(0)

    h = tl.arange(0, BH)
    w = tl.arange(0, BW)
    hm = h < H
    wm = w < W
    mm = hm[:, None] & wm[None, :]

    hh = h[:, None]
    ww = w[None, :]
    t = hh * W + ww

    oh_lo = (hh - 1) >> 1
    oh_hi = oh_lo + 1
    ow_lo = (ww - 1) >> 1
    ow_hi = ow_lo + 1

    h1 = hh - 1
    w1 = ww - 1
    hm_lo = (oh_lo >= 0) & (oh_lo < H_out) & ((oh_lo << 1) >= h1)
    hm_hi = oh_hi < H_out
    wm_lo = (ow_lo >= 0) & (ow_lo < W_out) & ((ow_lo << 1) >= w1)
    wm_hi = ow_hi < W_out

    off_lo = oh_lo * W_out
    off_hi = oh_hi * W_out

    nc_base = pid * IMG
    for img in tl.static_range(IMG):
        nc = nc_base + img
        grad_base = grad_ptr + nc * (H_out * W_out)
        idx_base = idx_ptr + nc * (H_out * W_out)

        acc = tl.zeros((BH, BW), dtype=tl.float32)

        valid = hm_lo & wm_lo
        off = off_lo + ow_lo
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = hm_hi & wm_lo
        off = off_hi + ow_lo
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = hm_lo & wm_hi
        off = off_lo + ow_hi
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )
        valid = hm_hi & wm_hi
        off = off_hi + ow_hi
        acc += tl.where(
            tl.load(idx_base + off, mask=valid, other=-1) == t,
            tl.load(grad_base + off, mask=valid, other=0.0),
            0.0,
        )

        tl.store(out_ptr + nc * (H * W) + t, acc, mask=mm)


def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _specialized_max_pool2d_with_indices_backward(
    grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    N, C, H, W = self.shape
    kH, kW = _pair(kernel_size)
    if stride is None or (isinstance(stride, (tuple, list)) and len(stride) == 0):
        stride = kernel_size
    sH, sW = _pair(stride)
    if sH == 0:
        sH = kH
    if sW == 0:
        sW = kW
    pH, pW = _pair(padding)
    dH, dW = _pair(dilation)

    H_out, W_out = grad_output.shape[2], grad_output.shape[3]
    NCH = N * C
    total_in = H * W

    out = torch.empty_like(self)

    if NCH == 0 or total_in == 0:
        return out

    fast = (
        kH == 3
        and kW == 3
        and sH == 2
        and sW == 2
        and pH == 1
        and pW == 1
        and dH == 1
        and dW == 1
    )

    if fast and total_in < 128:
        BH = triton.next_power_of_2(H)
        BW = triton.next_power_of_2(W)
        IMG = max(1, min(4, 128 // total_in))
        grid = (triton.cdiv(NCH, IMG),)
        num_warps = 2 if BH * BW <= 64 else 4
        _maxpool2d_bwd_fast_tile_kernel[grid](
            grad_output,
            indices,
            out,
            NCH,
            H,
            W,
            H_out,
            W_out,
            BH,
            BW,
            IMG,
            num_warps=num_warps,
        )
    elif fast:
        total_n = NCH * total_in
        BLOCK = 256
        exact = total_n % BLOCK == 0
        grid = (total_n // BLOCK,) if exact else (triton.cdiv(total_n, BLOCK),)
        _maxpool2d_bwd_fast_kernel[grid](
            grad_output,
            indices,
            out,
            total_n,
            W,
            total_in,
            H_out,
            W_out,
            BLOCK,
            exact,
        )
    else:
        total_n = NCH * total_in
        BLOCK = 256
        grid = (triton.cdiv(total_n, BLOCK),)
        _maxpool2d_bwd_general_kernel[grid](
            grad_output,
            indices,
            out,
            total_n,
            W,
            total_in,
            H_out,
            W_out,
            kH,
            kW,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            BLOCK,
        )
    return out


def max_pool2d_with_indices_backward(
    grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    logger.debug("GEMS_MTHREADS MAX_POOL2D_WITH_INDICES_BACKWARD")
    if (
        isinstance(grad_output, torch.Tensor)
        and grad_output.device.type == "musa"
        and grad_output.dtype in _SUPPORTED_DTYPES
        and isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
        and isinstance(indices, torch.Tensor)
        and indices.device.type == "musa"
        and indices.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_max_pool2d_with_indices_backward(
            grad_output,
            self,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
        )
    return default_max_pool2d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices
    )
