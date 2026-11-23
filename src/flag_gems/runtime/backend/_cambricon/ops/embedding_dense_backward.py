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

import flag_gems

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@triton.jit
def _embedding_dense_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    total_blocks: tl.constexpr,
    grid_dim_d: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    for flat_pid in range(pid, total_blocks, num_jobs):
        pid_n = flat_pid // grid_dim_d
        pid_d = flat_pid % grid_dim_d

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < EMBED_DIM

        idx = tl.load(indices_ptr + pid_n)
        valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

        go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
        go = tl.load(go_ptrs, mask=mask_d, other=0).to(tl.float32)

        gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
        mask = mask_d & valid
        tl.atomic_add(gw_ptrs, go, mask=mask)


@triton.jit
def _embedding_dense_backward_count_kernel(
    indices_ptr,
    counts_ptr,
    N,
    num_weights,
    padding_idx,
    BLOCK_N: tl.constexpr,
    total_blocks: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    for block_id in range(pid, total_blocks, num_jobs):
        offs = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        idx = tl.load(indices_ptr + offs, mask=mask, other=0).to(tl.int32)
        valid = mask & (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
        tl.atomic_add(counts_ptr + idx, 1, mask=valid)


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq(
    grad_output_ptr,
    indices_ptr,
    counts_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    total_blocks: tl.constexpr,
    grid_dim_d: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    for flat_pid in range(pid, total_blocks, num_jobs):
        pid_n = flat_pid // grid_dim_d
        pid_d = flat_pid % grid_dim_d

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < EMBED_DIM

        idx = tl.load(indices_ptr + pid_n).to(tl.int32)
        valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

        go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
        go = tl.load(go_ptrs, mask=mask_d, other=0.0)

        cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
        go = go / cnt

        gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
        mask = mask_d & valid
        tl.atomic_add(gw_ptrs, go, mask=mask)


def embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    logger.debug("GEMS_CAMBRICON: embedding_dense_backward")
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    if (
        grad_output.device.type != flag_gems.device
        or indices.device.type != flag_gems.device
        or grad_output.device != indices.device
    ):
        raise ValueError(
            f"Inputs must be {flag_gems.device} tensors on the same device."
        )

    device = grad_output.device
    assert (
        grad_output.dim() >= 2
    ), "grad_output must have embedding dimension as the last dim."

    D = grad_output.shape[-1]
    go = grad_output.contiguous().view(-1, D)  # (N, D)
    idx = indices.contiguous().view(-1)
    N = idx.numel()

    assert go.shape[0] == N, "indices number must match grad_output rows."
    grad_weight_fp32 = torch.zeros((num_weights, D), device=device, dtype=torch.float32)

    num_warps = 1
    MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps

    BLOCK_D = 128
    grid_dim_d = (D + BLOCK_D - 1) // BLOCK_D
    total_blocks = N * grid_dim_d
    grid = lambda meta: (min(total_blocks, MAX_GRID_SIZE),)

    if scale_grad_by_freq:
        counts = torch.zeros((num_weights,), device=device, dtype=torch.int32)
        BLOCK_N = 512
        total_blocks_count = (N + BLOCK_N - 1) // BLOCK_N
        grid_count = lambda meta: (min(total_blocks_count, MAX_GRID_SIZE),)
        _embedding_dense_backward_count_kernel[grid_count](
            idx,
            counts,
            N,
            num_weights,
            padding_idx if padding_idx is not None else -1,
            BLOCK_N=BLOCK_N,
            total_blocks=total_blocks_count,
            num_warps=num_warps,
        )

        _embedding_dense_backward_kernel_scale_by_freq[grid](
            go,
            idx,
            counts,
            grad_weight_fp32,
            num_weights,
            padding_idx if padding_idx is not None else -1,
            BLOCK_D=BLOCK_D,
            EMBED_DIM=D,
            total_blocks=total_blocks,
            grid_dim_d=grid_dim_d,
            num_warps=num_warps,
        )
    else:
        _embedding_dense_backward_kernel[grid](
            go,
            idx,
            grad_weight_fp32,
            num_weights,
            padding_idx if padding_idx is not None else -1,
            BLOCK_D=BLOCK_D,
            EMBED_DIM=D,
            total_blocks=total_blocks,
            grid_dim_d=grid_dim_d,
            num_warps=num_warps,
        )

    if grad_output.dtype != torch.float32:
        return grad_weight_fp32.to(grad_output.dtype)
    return grad_weight_fp32
