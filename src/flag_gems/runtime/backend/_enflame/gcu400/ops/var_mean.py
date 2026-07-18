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
from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid_m = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid_m, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _sum2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
            _sum += x
            _sum2 += x * x

        total_sum = tl.sum(_sum, axis=1)
        total_sum2 = tl.sum(_sum2, axis=1)
        mean = total_sum / N
        var = (total_sum2 - total_sum * total_sum / N) / (N - correction)
        var = tl.maximum(var, 0.0)

        tl.store(Mean + m_offset, mean, row_mask)
        tl.store(Var + m_offset, var, row_mask)


@libentry()
@triton.jit
def var_mean_global_kernel_1(
    X,
    Mid_sum,
    Mid_sum2,
    M,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        sum_val = tl.sum(x, axis=0)
        sum2_val = tl.sum(x * x, axis=0)
        tl.store(Mid_sum + tile_id, sum_val)
        tl.store(Mid_sum2 + tile_id, sum2_val)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_mean_global_kernel_2(
    Mid_sum,
    Mid_sum2,
    Var,
    Mean,
    N,
    correction,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    partial_sum = tl.load(Mid_sum + offset, mask=mask, other=0.0).to(tl.float32)
    partial_sum2 = tl.load(Mid_sum2 + offset, mask=mask, other=0.0).to(tl.float32)

    total_sum = tl.sum(partial_sum)
    total_sum2 = tl.sum(partial_sum2)

    mean = total_sum / N
    var = (total_sum2 - total_sum * total_sum / N) / (N - correction)
    var = tl.maximum(var, 0.0)

    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ENFLAME VAR_MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        block_size = 32 * 64
        if N < 24 * 16 * 1024:
            block_size = 16 * 1024
        elif N >= 24 * 32 * 1024 and N < 24 * 64 * 1024:
            block_size = 32 * 1024
        elif N >= 24 * 64 * 1024:
            block_size = 64 * 1024

        mid_size = triton.cdiv(N, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        num_stages = 1
        if mid_size > 4 * 24:
            num_stages = 3

        mid_sum = torch.empty([mid_size], dtype=torch.float32, device=x.device)
        mid_sum2 = torch.empty([mid_size], dtype=torch.float32, device=x.device)

        grid_size = min(mid_size, 24)

        with torch_device_fn.device(x.device):
            var_mean_global_kernel_1[(grid_size, 1, 1)](
                x,
                mid_sum,
                mid_sum2,
                N,
                BLOCK_SIZE=block_size,
                num_stages=num_stages,
                num_warps=1,
            )
            var_mean_global_kernel_2[(1, 1, 1)](
                mid_sum,
                mid_sum2,
                var,
                mean,
                N,
                correction,
                mid_size,
                BLOCK_MID=block_mid,
                num_warps=1,
            )
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        BLOCK_N = min(triton.next_power_of_2(N), 2048)
        BLOCK_M = max(1, min(128, 32768 // BLOCK_N))

        num_stages = 1

        grid_m = min(triton.cdiv(M, BLOCK_M), MAX_GRID_DIM)

        with torch_device_fn.device(x.device):
            var_mean_kernel[(grid_m,)](
                x,
                var,
                mean,
                M,
                N,
                correction,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=1,
            )

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
