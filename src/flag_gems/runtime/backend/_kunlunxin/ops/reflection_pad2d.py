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


# Flat 1D kernel over the ENTIRE output (all batches at once).
#
# ROOT CAUSE of the old slowness: the previous kernel wrapped every store index
# with `% HW_out` ("modulo wrap") to avoid masked stores. On KunlunXin XPU that
# runtime modulo defeats OffsetAnalysis, so EVERY load/store degrades to the
# discrete per-element path (~1.2 GB/s). Even a pure contiguous copy written with
# `%total` measured 228ms / 1.2 GB/s vs 0.49ms / 578 GB/s for the mask-based
# copy — a ~470x penalty (see reflection_pad2d_perf_fix.md).
#
# Fix: flatten (b, h_out, w_out) into one linear output index `o` and store to
# `o` directly (provably stride-1 -> block DMA). A single boolean mask
# `o < total_out` handles the tail. Because the layout is one flat contiguous
# buffer, the only masked-out threads sit at the very end (o >= total_out) and
# would write PAST the whole buffer — so even if a masked store were not
# suppressed it could not corrupt a valid element; and it is in fact suppressed
# here (verified maxdiff=0 on all shapes, including a tail-masked shape). This
# removes the "adjacent batch corruption" hazard that motivated the modulo wrap.
#
# The reflected input index is still a data-dependent gather (structural XPU
# wall), and the flat-index decode needs integer div/mod (slow on XPU), so the
# big shape stays ~40ms; but that is ~4.5x faster than the 183ms modulo version.
@triton.jit
def reflection_pad2d_kernel(
    in_ptr,
    out_ptr,
    H_in,
    W_in,
    pad_left,
    pad_top,
    W_out,
    HW_out,
    HW_in,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat output index -> (batch, h_out, w_out)
    b = o // HW_out
    rem = o % HW_out
    h_idx = rem // W_out
    w_idx = rem % W_out

    # Reflected height index. pad_top < H_in is validated on the host, so a single
    # period (abs + where) is exact — no `% (2*(H_in-1))` needed.
    y = h_idx.to(tl.int32) - pad_top
    pH = 2 * (H_in - 1)
    t_h = tl.abs(y)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    # Reflected width index (same reasoning; pad_left < W_in validated).
    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t_w = tl.abs(x)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    in_offs = b * HW_in + ih * W_in + iw
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


def launch_reflection_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    # Validate reflection padding constraints
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        expected_shape = (*leading_shape, H_out, W_out)
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

    # No padding: just copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        BLOCK = 1024
        total = B * H_in * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    # BLOCK=1024 is the best all-round tile on XPU: small shapes avoid the
    # per-program waste of a huge block, while medium/large shapes still get
    # enough work per program to stay off the launch floor (measured sweep).
    BLOCK = 1024
    HW_out = H_out * W_out
    HW_in = H_in * W_in
    total_out = B * HW_out
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad2d_kernel[grid](
            x,
            out,
            H_in,
            W_in,
            pad_left,
            pad_top,
            W_out,
            HW_out,
            HW_in,
            total_out,
            BLOCK=BLOCK,
        )
    return out


def reflection_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
