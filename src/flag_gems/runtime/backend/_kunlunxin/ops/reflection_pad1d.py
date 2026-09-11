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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Flat 1D kernel over the ENTIRE output (all batch rows at once).
#
# ROOT CAUSE of the old slowness: the previous kernel wrapped every store index
# with `% W_out` ("modulo wrap") to avoid masked stores. On KunlunXin XPU that
# runtime modulo defeats OffsetAnalysis, so EVERY load/store degrades to the
# discrete per-element path (~1.2 GB/s), a ~470x penalty vs mask-based
# contiguous stores (see reflection_pad2d_perf_fix.md). Baseline big shape
# [32,64,2048] pad[3,5] measured ~14ms / speedup 0.002.
#
# Fix: flatten (b, w_out) into one linear output index `o` and store to `o`
# directly (provably stride-1 -> block DMA). A single boolean mask
# `o < total_out` handles the tail. Because the layout is one flat contiguous
# buffer, the only masked-out threads sit at the very end (o >= total_out) and
# could not corrupt a valid element even if not suppressed (and it is in fact
# suppressed here). This removes the "adjacent batch corruption" hazard that
# motivated the modulo wrap.
@triton.jit
def reflection_pad1d_kernel(
    in_ptr, out_ptr, W_in, pad_left, W_out, total_out, BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat output index -> (batch row, w_out)
    b = o // W_out
    w_idx = o % W_out

    # Reflected width index. pad_left < W_in is validated on the host, so a
    # single period (abs + where) is exact -- no `% (2*(W_in-1))` needed.
    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t = tl.abs(x)
    iw = tl.where(t < W_in, t, pW - t)

    in_offs = b * W_in + iw
    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    # Flat contiguous copy (no padding path). Mask-based, contiguous offsets ->
    # block DMA, same as the padded kernel's store side.
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


def _launch_reflection_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension")

    x = input.contiguous()
    W_in = int(x.shape[-1])
    if W_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if out is None:
        out = torch.empty((*leading_shape, W_out), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (*leading_shape, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    # BLOCK=1024 is the best all-round tile on XPU (measured sweep in the
    # reflection_pad2d fix): small shapes avoid huge-block launch waste, and
    # medium/large shapes still get enough work per program.
    BLOCK = 1024

    # No padding: just copy
    if pad_left == 0 and pad_right == 0:
        total = B * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    # Validate reflection padding constraints
    if W_in < 2:
        raise ValueError(
            "input width must be at least 2 for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in:
        raise ValueError(
            "padding values must be less than the input width for reflection padding"
        )

    total_out = B * W_out
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad1d_kernel[grid](
            x, out, W_in, pad_left, W_out, total_out, BLOCK=BLOCK
        )
    return out


def reflection_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D")
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_OUT")
    return _launch_reflection_pad1d(input, padding, out=out)
