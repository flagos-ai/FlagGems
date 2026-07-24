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

# XPU tl.sum correctness ceilings (verified in isolation, see solution doc):
#   * WITHOUT buffer_size_limit a 1D tile reduction is complete only for
#     BLOCK <= 8192; beyond that tl.sum silently drops the tail lanes.
#   * WITH buffer_size_limit=2048 BLOCK == 32768 is complete, but an
#     intermediate single-program BLOCK such as 16384 still miscompiles
#     (~1e-3 relative error).
# Therefore the single-program path is restricted to BLOCK <= 8192 (no buffer,
# fastest for small N) and everything larger goes through a two-stage split
# with a fixed BLOCK == 32768 tile launched with buffer_size_limit=2048.
SINGLE_BLOCK = 8192
SPLIT_BLOCK = 32768


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
        # One program reduces the whole vector in a single tl.sum. No
        # buffer_size_limit (block <= 8192 is already complete) which is the
        # fastest option for small N.
        block_size = triton.next_power_of_2(N)
        out = torch.empty([], dtype=torch.float32, device=x.device)
        with torch_device_fn.device(x.device):
            dot_kernel[(1,)](x, y, out, N, block_size)
            out = out.to(x.dtype)
        return out

    # Two-stage split reduction. Fixed block=32768 (fastest large-N tile in
    # isolation, and the max block that keeps tl.sum correct with
    # buffer_size_limit=2048). mid is sized to EXACTLY the grid so
    # dot_kernel_2 never sums uninitialized entries. For N up to ~1e9,
    # mid_size <= 32768, so dot_kernel_2's tile also stays correct.
    block_size = SPLIT_BLOCK
    mid_size = triton.cdiv(N, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    grid_1 = (mid_size,)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
    out = torch.empty([], dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        dot_kernel_1[grid_1](x, y, mid, N, block_size, buffer_size_limit=2048)
        dot_kernel_2[(1,)](mid, out, mid_size, block_mid, buffer_size_limit=2048)

    return out
