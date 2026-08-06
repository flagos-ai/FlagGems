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
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# XPU wide tl.sum can either lose lanes or miscompile particular fp32 inputs.
# Keep every reduction tile at or below the empirically reliable 512-lane
# boundary and recursively reduce partial sums for large vectors.
SINGLE_BLOCK = 512
SPLIT_BLOCK = 512


@libentry()
@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(x * y))


@libentry()
@triton.jit
def dot_kernel_1(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


@libentry()
@triton.jit
def dot_sum_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(in_ptr + offsets, mask=offsets < N, other=0.0)
    tl.store(out_ptr + pid, tl.sum(values))


@libentry()
@triton.jit
def dot_kernel_2(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < M
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(mid_val))


def dot(x, y):
    logger.debug("GEMS_KUNLUNXIN DOT")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]

    if N <= SINGLE_BLOCK:
        # One program reduces the whole vector in a single reliable tile.
        block_size = triton.next_power_of_2(N)
        out = torch.empty([], dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            dot_kernel[(1,)](x, y, out, N, block_size)
            out = out.to(x.dtype)
        return out

    # Reduce in a tree whose every tile stays within XPU's reliable 512-lane
    # tl.sum limit. The first level multiplies the inputs; later levels sum
    # the partials until the final scalar reduction.
    block_size = SPLIT_BLOCK
    mid_size = triton.cdiv(N, block_size)
    mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
    out = torch.empty([], dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        dot_kernel_1[(mid_size,)](x, y, mid, N, block_size)
        while mid_size > SINGLE_BLOCK:
            next_size = triton.cdiv(mid_size, block_size)
            next_mid = torch.empty((next_size,), dtype=torch.float32, device=x.device)
            dot_sum_kernel[(next_size,)](mid, next_mid, mid_size, block_size)
            mid = next_mid
            mid_size = next_size
        dot_kernel_2[(1,)](
            mid,
            out,
            mid_size,
            triton.next_power_of_2(mid_size),
        )

    return out
