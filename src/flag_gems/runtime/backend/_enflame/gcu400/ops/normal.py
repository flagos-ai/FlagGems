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
from flag_gems.utils.libentry import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import broadcast_shapes, volume

logger = logging.getLogger(__name__)
UNROLL = 4
NUM_SIPS = 24


@triton.jit
def _pair_uniform_to_normal(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    theta = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    return r * tl.cos(theta), r * tl.sin(theta)


@libentry()
@triton.jit(
    do_not_specialize=["philox_seed", "philox_offset", "N", "std_val", "mean_val"]
)
def normal_fused_kernel(
    out_ptr,
    N,
    std_val,
    mean_val,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    base_c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        c0 = base_c0 + block_id * BLOCK + arange
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        n0, n1 = _pair_uniform_to_normal(r0, r1)
        n2, n3 = _pair_uniform_to_normal(r2, r3)

        n0 = n0 * std_val + mean_val
        n1 = n1 * std_val + mean_val
        n2 = n2 * std_val + mean_val
        n3 = n3 * std_val + mean_val

        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def normal_tt_fused_kernel(
    out_ptr,
    mean_ptr,
    std_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    base_c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        c0 = base_c0 + block_id * BLOCK + arange
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        n0, n1 = _pair_uniform_to_normal(r0, r1)
        n2, n3 = _pair_uniform_to_normal(r2, r3)

        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        m0 = tl.load(mean_ptr + off_0, mask=off_0 < N).to(tl.float32)
        m1 = tl.load(mean_ptr + off_1, mask=off_1 < N).to(tl.float32)
        m2 = tl.load(mean_ptr + off_2, mask=off_2 < N).to(tl.float32)
        m3 = tl.load(mean_ptr + off_3, mask=off_3 < N).to(tl.float32)

        s0 = tl.load(std_ptr + off_0, mask=off_0 < N).to(tl.float32)
        s1 = tl.load(std_ptr + off_1, mask=off_1 < N).to(tl.float32)
        s2 = tl.load(std_ptr + off_2, mask=off_2 < N).to(tl.float32)
        s3 = tl.load(std_ptr + off_3, mask=off_3 < N).to(tl.float32)

        n0 = n0 * s0 + m0
        n1 = n1 * s1 + m1
        n2 = n2 * s2 + m2
        n3 = n3 * s3 + m3

        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def normal_tf_fused_kernel(
    out_ptr,
    mean_ptr,
    N,
    std_val,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    base_c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        c0 = base_c0 + block_id * BLOCK + arange
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        n0, n1 = _pair_uniform_to_normal(r0, r1)
        n2, n3 = _pair_uniform_to_normal(r2, r3)

        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        m0 = tl.load(mean_ptr + off_0, mask=off_0 < N).to(tl.float32)
        m1 = tl.load(mean_ptr + off_1, mask=off_1 < N).to(tl.float32)
        m2 = tl.load(mean_ptr + off_2, mask=off_2 < N).to(tl.float32)
        m3 = tl.load(mean_ptr + off_3, mask=off_3 < N).to(tl.float32)

        n0 = n0 * std_val + m0
        n1 = n1 * std_val + m1
        n2 = n2 * std_val + m2
        n3 = n3 * std_val + m3

        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def normal_ft_fused_kernel(
    out_ptr,
    std_ptr,
    N,
    mean_val,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    base_c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        c0 = base_c0 + block_id * BLOCK + arange
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        n0, n1 = _pair_uniform_to_normal(r0, r1)
        n2, n3 = _pair_uniform_to_normal(r2, r3)

        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        s0 = tl.load(std_ptr + off_0, mask=off_0 < N).to(tl.float32)
        s1 = tl.load(std_ptr + off_1, mask=off_1 < N).to(tl.float32)
        s2 = tl.load(std_ptr + off_2, mask=off_2 < N).to(tl.float32)
        s3 = tl.load(std_ptr + off_3, mask=off_3 < N).to(tl.float32)

        n0 = n0 * s0 + mean_val
        n1 = n1 * s1 + mean_val
        n2 = n2 * s2 + mean_val
        n3 = n3 * s3 + mean_val

        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


def _get_block_and_grid(
    N, has_tensor_io=False, use_grid_stride=False, prefer_small_grid=False
):
    if N <= 4096:
        BLOCK = 1024
    elif has_tensor_io:
        BLOCK = 1024
    else:
        BLOCK = 1024
    num_blocks = triton.cdiv(N, BLOCK * UNROLL)
    if use_grid_stride:
        max_grid = NUM_SIPS * 4 if prefer_small_grid else NUM_SIPS * 8
        grid_size = min(num_blocks, max_grid)
    else:
        grid_size = num_blocks
    return BLOCK, (grid_size,)


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_")
    shape = self.shape
    device = self.device
    N = volume(shape)
    is_fp32 = self.dtype == torch.float32
    BLOCK, grid = _get_block_and_grid(N, use_grid_stride=True, prefer_small_grid=True)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    use_temp = (not is_fp32) and (N > 1048576)
    if use_temp:
        temp = torch.empty(N, device=device, dtype=torch.float32)
        with torch_device_fn.device(device):
            normal_fused_kernel[grid](
                temp,
                N,
                float(std),
                float(mean),
                philox_seed,
                philox_offset,
                BLOCK=BLOCK,
                num_warps=4,
            )
        self.view(-1).copy_(temp)
    else:
        with torch_device_fn.device(device):
            normal_fused_kernel[grid](
                self,
                N,
                float(std),
                float(mean),
                philox_seed,
                philox_offset,
                BLOCK=BLOCK,
                num_warps=4,
            )
    return self


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    N = volume(shape)

    mean_expanded = mean.expand(shape).contiguous().view(-1)
    std_expanded = std.expand(shape).contiguous().view(-1)

    out = torch.empty(N, device=device, dtype=torch.float32)

    BLOCK, grid = _get_block_and_grid(N, has_tensor_io=True, use_grid_stride=True)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        normal_tt_fused_kernel[grid](
            out,
            mean_expanded,
            std_expanded,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out.view(shape)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    N = volume(shape)

    mean_flat = mean.contiguous().view(-1)
    out = torch.empty(N, device=device, dtype=torch.float32)

    BLOCK, grid = _get_block_and_grid(N, has_tensor_io=True, use_grid_stride=True)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        normal_tf_fused_kernel[grid](
            out,
            mean_flat,
            N,
            float(std),
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out.view(shape)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    N = volume(shape)

    std_flat = std.contiguous().view(-1)
    out = torch.empty(N, device=device, dtype=torch.float32)

    BLOCK, grid = _get_block_and_grid(N, has_tensor_io=True, use_grid_stride=True)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        normal_ft_fused_kernel[grid](
            out,
            std_flat,
            N,
            float(mean),
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out.view(shape)
