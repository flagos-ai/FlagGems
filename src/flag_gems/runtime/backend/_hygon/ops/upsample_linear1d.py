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

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def upsample_linear1d_kernel(
    input_ptr,
    output_ptr,
    W_in,
    W_out,
    n_wblocks,
    scale,
    bias,
    BLOCK_SIZE: tl.constexpr,
):
    # 1-D flat grid over (row, w-block): row/wblk are derived from the scalar
    # pid once per program, so consecutive CTAs write consecutive memory --
    # the previous 2-D (NC, W) grid scattered stores across rows and cost ~2x
    # DRAM bandwidth on hygon.
    pid = tl.program_id(0)
    row = pid // n_wblocks
    wblk = pid % n_wblocks

    offs_w = wblk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_w < W_out

    src = offs_w.to(tl.float32) * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    lower = tl.floor(src).to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)

    base_in = row * W_in
    base_out = row * W_out
    x0 = tl.load(input_ptr + base_in + lower, mask=mask, other=0.0)
    x1 = tl.load(input_ptr + base_in + upper, mask=mask, other=0.0)

    out = (1.0 - t) * x0.to(tl.float32) + t * x1.to(tl.float32)
    tl.store(output_ptr + base_out + offs_w, out.to(x0.dtype), mask=mask)


@triton.jit
def upsample_linear1d_kernel_nomask(
    input_ptr,
    output_ptr,
    W_in,
    W_out,
    n_wblocks,
    scale,
    bias,
    BLOCK_SIZE: tl.constexpr,
):
    # Same as above but without bounds masks; requires W_out % BLOCK_SIZE == 0.
    pid = tl.program_id(0)
    row = pid // n_wblocks
    wblk = pid % n_wblocks

    offs_w = wblk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    src = offs_w.to(tl.float32) * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    lower = tl.floor(src).to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)

    base_in = row * W_in
    base_out = row * W_out
    x0 = tl.load(input_ptr + base_in + lower)
    x1 = tl.load(input_ptr + base_in + upper)

    out = (1.0 - t) * x0.to(tl.float32) + t * x1.to(tl.float32)
    tl.store(output_ptr + base_out + offs_w, out.to(x0.dtype))


def upsample_linear1d(
    self: torch.Tensor,
    output_size,
    align_corners: bool,
    scales: float = None,
):
    logger.debug("GEMS UPSAMPLE LINEAR1D OPTIMIZED")
    assert self.ndim == 3, "Input must be [N, C, W]"
    assert self.device.type == flag_gems.device

    N, C, W_in = self.shape
    NC = N * C

    if output_size is not None:
        W_out = int(
            output_size[0] if isinstance(output_size, (list, tuple)) else output_size
        )
    else:
        assert scales is not None
        W_out = int(math.floor(W_in * scales))

    inp = self.contiguous().view(NC, W_in)
    out = torch.empty((NC, W_out), device=self.device, dtype=self.dtype)

    if W_out == 0 or NC == 0:
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

    BLOCK_SIZE = 1024
    n_wblocks = triton.cdiv(W_out, BLOCK_SIZE)
    grid = (NC * n_wblocks,)

    if W_out % BLOCK_SIZE == 0:
        upsample_linear1d_kernel_nomask[grid](
            inp,
            out,
            W_in,
            W_out,
            n_wblocks,
            scale_val,
            bias_val,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
    else:
        upsample_linear1d_kernel[grid](
            inp,
            out,
            W_in,
            W_out,
            n_wblocks,
            scale_val,
            bias_val,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return out.view(N, C, W_out)
