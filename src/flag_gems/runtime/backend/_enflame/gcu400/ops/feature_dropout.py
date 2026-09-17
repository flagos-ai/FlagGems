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
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def generate_feature_mask_kernel(
    MASK,
    N,
    C,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    n_mask = n_offset < N
    c_mask = c_offset < C

    flat_idx = n_offset[:, None] * C + c_offset[None, :]

    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = flat_idx.to(tl.uint32)
    c0 = c0 + i4
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand_vals = uint_to_uniform_float(r0)

    mask_vals = tl.where(rand_vals > p, scale, 0.0)

    mask_offsets = n_offset[:, None] * C + c_offset[None, :]
    mask_mask = n_mask[:, None] & c_mask[None, :]
    tl.store(MASK + mask_offsets, mask_vals, mask=mask_mask)


@libentry()
@triton.jit(do_not_specialize=["numel"])
def apply_feature_mask_kernel(
    X,
    Y,
    MASK,
    numel,
    N,
    C,
    spatial_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    channel_spatial_size = C * spatial_size
    for block_id in tl.range(pid, (numel + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < numel

        n_idx = off // channel_spatial_size
        c_idx = (off % channel_spatial_size) // spatial_size
        mask_idx = n_idx * C + c_idx

        x = tl.load(X + off, mask=mask, other=0.0)
        m = tl.load(MASK + mask_idx, mask=mask, other=0.0)
        y = x * m
        tl.store(Y + off, y, mask=mask)


def feature_dropout(input, p, train=True):
    logger.debug("GEMS_ENFLAME FEATURE_DROPOUT")

    if not train or p == 0:
        return input.clone()

    if p == 1:
        return torch.zeros_like(input)

    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)

    batch_size = input.shape[0]
    num_channels = input.shape[1]
    spatial_size = 1
    for i in range(2, input.ndim):
        spatial_size *= input.shape[i]

    N = batch_size
    C = num_channels
    numel = input.numel()
    scale = 1.0 / (1.0 - p)

    mask = torch.empty(N, C, device=device, dtype=torch.float32)

    BLOCK_N = min(triton.next_power_of_2(N), 64)
    BLOCK_C = min(triton.next_power_of_2(C), 64)
    grid_mask = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))

    increment = triton.cdiv(N * C, 4) * 4
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        generate_feature_mask_kernel[grid_mask](
            mask, N, C, p, scale, philox_seed, philox_offset, BLOCK_N, BLOCK_C
        )

        BLOCK = 8192
        grid_size = min((numel + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
        apply_feature_mask_kernel[(grid_size,)](
            input, out, mask, numel, N, C, spatial_size, BLOCK=BLOCK, num_warps=4
        )

    return out


def feature_dropout_(input, p, train=True):
    logger.debug("GEMS_ENFLAME FEATURE_DROPOUT_")
    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    out = feature_dropout(input, p, train)
    input.copy_(out)
    return input
