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

"""KunlunXin (XPU) implementation of `upsample_linear1d`.

The dominant workload is an integer 2x upsampling
(``output_size == 2 * input_size``) with ``align_corners=False``, for which
the source coordinate ``src(w) = (w + 0.5) / 2 - 0.5`` makes every output pair
a fixed 2-tap stencil:

    out[2m]     = 0.25 * in[m-1] + 0.75 * in[m]
    out[2m + 1] = 0.75 * in[m]   + 0.25 * in[m+1]

(with the neighbour clamped to the row, which matches ATen's own coordinate
clamp bit for bit).  Two dedicated kernels exploit that:

* **2x fast paths** (``_upsample_linear1d_2x16_kernel`` /
  ``_upsample_linear1d_2x32_kernel``): one lane per input element, the two
  outputs are packed into a single wider store (``uint32`` for f16/bf16,
  ``uint64`` for f32), the ``row`` index comes from ``program_id(1)`` so there
  is **no per-element div/mod and no float->int conversion**, and the loads
  are unmasked (the clamped indices always land inside the row).

* **General kernel** (``_upsample_linear1d_row_kernel``): one program per
  output row (``nc = program_id(1)`` is a scalar, so per-element
  ``o // W_out`` is gone); the loads are unmasked because the clamped source
  position keeps every address inside the row, and the store keeps a pure
  tail mask so it stays an affine block DMA.

A 2x kernel is only legal when the *effective* scale is exactly 1/2: when
``scales`` is given ATen uses ``1/scales`` instead of the shape ratio, so a
size-2x request with ``scales=3`` must fall back to the general kernel.

The ``uint32``-domain RNE rounding for bf16 exists because this backend
rejects a bf16 bitcast (``Cannot bitcast size 32 to size 16``); f16 -> u16
bitcast is fine.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Lanes per program. 1-D blocks must be >= 256 on this backend.
_BLOCK_SIZE = 1024


@triton.jit
def _upsample_linear1d_2x16_kernel(
    input_ptr,
    output_ptr,
    W_in,
    BLOCK: tl.constexpr,
):
    # f16/bf16 2x fast path. One lane per input element; the row is a program
    # axis so the lane index needs no div/mod, and since W_out == 2 * W_in the
    # flat output index of pair (2m, 2m+1) is 2m in the input-space flat index.
    # The two f16/bf16 results are packed into one uint32 store.
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col < W_in

    in_base = row * W_in
    # Clamp the neighbours to [0, W_in-1] so every load is in-bounds and thus
    # maskless; the clamp reproduces ATen's coordinate clamp exactly.
    prev = tl.maximum(col - 1, 0)
    cur = tl.minimum(col, W_in - 1)
    nxt = tl.minimum(col + 1, W_in - 1)

    x_prev = tl.load(input_ptr + in_base + prev).to(tl.float32)
    x_cur = tl.load(input_ptr + in_base + cur).to(tl.float32)
    x_nxt = tl.load(input_ptr + in_base + nxt).to(tl.float32)

    o0 = 0.25 * x_prev + 0.75 * x_cur
    o1 = 0.75 * x_cur + 0.25 * x_nxt

    if input_ptr.dtype.element_ty == tl.bfloat16:
        # bf16 bitcast is rejected by this backend, so round to bf16 in the
        # uint32 domain with round-to-nearest-even:
        #   bf16_bits = (f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) & ~0xFFFF
        b0 = o0.to(tl.uint32, bitcast=True)
        b1 = o1.to(tl.uint32, bitcast=True)
        r0 = (b0 + 0x7FFF + ((b0 >> 16) & 1)) & 0xFFFF0000
        r1 = (b1 + 0x7FFF + ((b1 >> 16) & 1)) & 0xFFFF0000
        packed = (r1 & 0xFFFF0000) | (r0 >> 16)
    else:
        h0 = o0.to(tl.float16).to(tl.uint16, bitcast=True)
        h1 = o1.to(tl.float16).to(tl.uint16, bitcast=True)
        packed = h0.to(tl.uint32) | (h1.to(tl.uint32) << 16)

    tl.store(output_ptr + in_base + col, packed, mask=mask)


@triton.jit
def _upsample_linear1d_2x32_kernel(
    input_ptr,
    output_ptr,
    W_in,
    BLOCK: tl.constexpr,
):
    # f32 2x fast path. Same stencil as above; the two f32 results are packed
    # into one uint64 store. Little-endian: low 32 bits = out[2m].
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col < W_in

    in_base = row * W_in
    prev = tl.maximum(col - 1, 0)
    cur = tl.minimum(col, W_in - 1)
    nxt = tl.minimum(col + 1, W_in - 1)

    x_prev = tl.load(input_ptr + in_base + prev)
    x_cur = tl.load(input_ptr + in_base + cur)
    x_nxt = tl.load(input_ptr + in_base + nxt)

    o0 = 0.25 * x_prev + 0.75 * x_cur
    o1 = 0.75 * x_cur + 0.25 * x_nxt

    u0 = o0.to(tl.uint32, bitcast=True)
    u1 = o1.to(tl.uint32, bitcast=True)
    packed = u0.to(tl.uint64) | (u1.to(tl.uint64) << 32)

    tl.store(output_ptr + in_base + col, packed, mask=mask)


@triton.jit
def _upsample_linear1d_row_kernel(
    input_ptr,
    output_ptr,
    W_in,
    W_out,
    scale,
    bias,
    BLOCK: tl.constexpr,
):
    # General path: one program per output row, so `nc` is a scalar and no
    # per-element div/mod is needed. The source coordinate is clamped to
    # [0, W_in-1], which keeps every load address inside the row even for the
    # tail lanes (w >= W_out) -> the loads can be unmasked. The store keeps a
    # pure tail mask and is therefore still an affine block DMA.
    nc = tl.program_id(1)
    w = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = w < W_out

    src = w.to(tl.float32) * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    lower = src.to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)
    w0 = 1.0 - t
    w1 = t

    in_base = nc * W_in
    x0 = tl.load(input_ptr + in_base + lower).to(tl.float32)
    x1 = tl.load(input_ptr + in_base + upper).to(tl.float32)

    out = w0 * x0 + w1 * x1
    out = out.to(input_ptr.dtype.element_ty)
    tl.store(output_ptr + nc * W_out + w, out, mask=mask)


def _is_2x_fast_path(align_corners, W_in, W_out, scales):
    """The 2x stencil needs the effective scale to be exactly 1/2."""
    if align_corners or W_out != 2 * W_in:
        return False
    if scales is None:
        return True
    try:
        return float(scales) == 2.0
    except (TypeError, ValueError):
        return False


def upsample_linear1d(
    self: torch.Tensor,
    output_size,
    align_corners: bool,
    scales: float = None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LINEAR1D")
    assert self.ndim == 3, "Input must be [N, C, W]"

    N, C, W_in = self.shape
    NC = N * C

    if output_size is not None:
        W_out = int(
            output_size[0] if isinstance(output_size, (list, tuple)) else output_size
        )
    else:
        assert (
            scales is not None
        ), "scales must be specified if output_size is not provided."
        W_out = int(math.floor(W_in * scales))

    inp = self.contiguous().view(NC, W_in)
    out = torch.empty((NC, W_out), device=self.device, dtype=self.dtype)

    if _is_2x_fast_path(align_corners, W_in, W_out, scales):
        if self.dtype == torch.float32:
            out_u64 = out.view(torch.uint64)
            with torch_device_fn.device(self.device):
                _upsample_linear1d_2x32_kernel[
                    (triton.cdiv(W_in, _BLOCK_SIZE), NC)
                ](inp, out_u64, W_in, BLOCK=_BLOCK_SIZE)
        else:
            out_u32 = out.view(torch.uint32)
            with torch_device_fn.device(self.device):
                _upsample_linear1d_2x16_kernel[
                    (triton.cdiv(W_in, _BLOCK_SIZE), NC)
                ](inp, out_u32, W_in, BLOCK=_BLOCK_SIZE)
        return out.view(N, C, W_out)

    if align_corners:
        if W_out > 1:
            scale_val = (W_in - 1.0) / (W_out - 1.0)
        else:
            scale_val = 0.0
        bias_val = 0.0
    else:
        if scales is not None:
            real_scale = 1.0 / scales
        else:
            real_scale = W_in / W_out

        scale_val = real_scale
        bias_val = 0.5 * real_scale - 0.5

    with torch_device_fn.device(self.device):
        _upsample_linear1d_row_kernel[
            (triton.cdiv(W_out, _BLOCK_SIZE), NC)
        ](
            inp,
            out,
            W_in,
            W_out,
            scale_val,
            bias_val,
            BLOCK=_BLOCK_SIZE,
        )

    return out.view(N, C, W_out)