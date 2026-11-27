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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 has 48 compute units and a hard grid.x limit of 65535 (P19). The partial kernel uses a
# persistent grid-stride scheme (one program striding across multiple tiles), so any N stays within the limit; 48-way +
# large
# tiles also give significantly better bandwidth than the one-tile-one-program linear layout (2^24/2^30 fp32 measured:
# 2110/2406 GB/s vs 826 GB/s for the linear layout).
MAX_GRID = 48

# Not exceeding this length, a single program reduces all the way in one launch, saving the second launch
# (fix-round measurements: N=4096 across three dtypes 4.04-4.06µs, old two-stage version 7.0µs).
SINGLE_MAX = 8192

# Tile size for the partial kernel of 4-byte types (empirically bandwidth-optimal for fp32).
BLOCK_4B = 32768

# Tile size for the partial kernel of 2-byte types (empirically optimal for fp16/bf16, with vector accumulation).
BLOCK_2B = 8192


@libentry()
@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum = tl.sum(x * y)
    tl.store(out_ptr, sum)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dot_kernel_1(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 4-byte types: scalar fp32 accumulation, one reduction per tile.
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    acc = tl.zeros([], dtype=tl.float32)
    for tile_id in tl.range(pid, num_tile, step):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum(x * y)

    tl.store(mid_ptr + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dot_kernel_1v(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 2-byte types: vector fp32 accumulation of BLOCK size, with a single cross-lane reduction only at the end of the
    # loop.
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for tile_id in tl.range(pid, num_tile, step):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        acc += x * y

    tl.store(mid_ptr + pid, tl.sum(acc, axis=0))


@libentry()
@triton.jit
def dot_kernel_2(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid = mid_ptr + offset
    mask = offset < M
    mid_val = tl.load(mid, mask=mask, other=0.0)
    out_val = tl.sum(mid_val)
    tl.store(out_ptr, out_val)


def dot(x, y):
    logger.debug("GEMS_ENFLAME DOT")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]

    # Small scale: a single launch reduces all the way (2^30 fix-round measurements at N=4096 across all dtypes:
    # 4.04us vs 7.0us for the two-stage version).
    if N <= SINGLE_MAX:
        block_size = triton.next_power_of_2(N)

        out = torch.empty([], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel[(1, 1, 1)](x, y, out, N, block_size)

        return out

    # Large scale: two-stage persistent partial + single-program merge. grid is always <= MAX_GRID,
    # so N >= 2^29 no longer triggers a grid.x overflow (P19: before the fix, the 2^30 case required 131072).
    if x.element_size() == 4:
        block_size = BLOCK_4B
        kernel_1 = dot_kernel_1
    else:
        block_size = BLOCK_2B
        kernel_1 = dot_kernel_1v

    mid_size = min(MAX_GRID, triton.cdiv(N, block_size))
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
    out = torch.empty([], dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        kernel_1[(mid_size, 1, 1)](x, y, mid, N, block_size, num_warps=4)
        dot_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, num_warps=1)

    return out
