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


@triton.jit
def upsample_linear1d_kernel(
    input_ptr,
    output_ptr,
    total,
    W_in,
    W_out,
    scale,
    bias,
    BLOCK_SIZE: tl.constexpr,
):
    # Flat 1D grid over the whole [NC, W_out] output. The store index is the raw
    # `o = pid * BLOCK + arange` (provably stride-1 -> contiguous block DMA). The
    # old kernel wrote `base_out + (offs % W_out)`: the runtime `%` in the store
    # index defeats OffsetAnalysis and collapses the store to discrete access
    # (~3-4x slower). A `min(o, total-1)` clamp has the same problem (measured
    # ~3.5x slower), so we keep the raw `o` plus a single tail bool mask.
    #
    # (nc, w) are recovered via div/mod only to compute the GATHER load address
    # (input is read at data-dependent lower/upper -> discrete regardless), so the
    # div/mod adds no penalty over the already-discrete load. On KunlunXin the
    # masked load/store here is correctly suppressed for out-of-range tail lanes
    # (verified on tiny buffers), so no modulo wrap or clamp is needed.
    pid = tl.program_id(0)
    o = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = o < total

    nc = o // W_out
    w = o - nc * W_out

    src = w.to(tl.float32) * scale + bias

    # Clamp source position to [0, W_in - 1]
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    # For non-negative src, int truncation equals floor
    lower = src.to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)
    w0 = 1.0 - t
    w1 = t

    base_in = nc * W_in
    x0 = tl.load(input_ptr + base_in + lower, mask=mask, other=0.0)
    x1 = tl.load(input_ptr + base_in + upper, mask=mask, other=0.0)

    x0_f = x0.to(tl.float32)
    x1_f = x1.to(tl.float32)

    out = w0 * x0_f + w1 * x1_f

    out = out.to(x0.dtype)
    tl.store(output_ptr + o, out, mask=mask)


def upsample_linear1d(
    self: torch.Tensor,
    output_size,
    align_corners: bool,
    scales: float = None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPL_LINEAR1D")
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

    BLOCK_SIZE = 8192
    total = NC * W_out
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    with torch_device_fn.device(self.device):
        upsample_linear1d_kernel[grid](
            inp,
            out,
            total,
            W_in,
            W_out,
            scale_val,
            bias_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out.view(N, C, W_out)
