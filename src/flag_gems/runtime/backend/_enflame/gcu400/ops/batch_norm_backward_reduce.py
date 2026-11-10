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
#
# GCU400 specialization migrated from flag_gems.ops.batch_norm_backward_reduce

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def batch_norm_backward_reduce_kernel(
    grad_out_ptr,
    input_ptr,
    mean_ptr,
    invstd_ptr,
    sum_dy_ptr,
    sum_dy_xmu_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    C,
    spatial_size,
    elements_per_channel,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_INPUT_G: tl.constexpr,
    COMPUTE_GRAD_WEIGHT: tl.constexpr,
    COMPUTE_GRAD_BIAS: tl.constexpr,
):
    """
    Fused batch norm backward reduce kernel.

    Computes per-channel reduction for batch normalization backward:
    - sum_dy = sum(grad_out) per channel
    - sum_dy_xmu = sum(grad_out * (input - mean)) per channel
    - grad_weight = sum(grad_out * normalized_input) per channel (if requested)
    - grad_bias = sum(grad_out) per channel (if requested)

    Grid: (C,) - one program per channel
    Each program reduces all elements belonging to its channel.

    Memory layout: (N, C, *spatial) contiguous
    For channel c, elements are at flat indices:
    - Batch 0: c * spatial_size + [0, spatial_size)
    - Batch 1: C * spatial_size + c * spatial_size + [0, spatial_size)
    - Batch n: n * C * spatial_size + c * spatial_size + [0, spatial_size)
    """
    # Get channel ID
    channel_id = tl.program_id(0)

    # Initialize accumulators in fp32 for precision
    acc_sum_dy = 0.0
    acc_sum_dy_xmu = 0.0
    acc_grad_weight = 0.0
    acc_grad_bias = 0.0

    # Load per-channel statistics
    mean = tl.load(mean_ptr + channel_id).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel_id).to(tl.float32)

    # Loop over all elements in this channel with tiling
    # elements_per_channel = N * spatial_size
    for offset in range(0, elements_per_channel, BLOCK_SIZE):
        # Element indices within this channel (0 to N*spatial_size)
        local_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = local_indices < elements_per_channel

        # Map local indices to global flat indices in the tensor
        # local_index i corresponds to:
        #   batch_idx = i // spatial_size
        #   spatial_idx = i % spatial_size
        #   flat_idx = batch_idx * (C * spatial_size) + channel_id * spatial_size + spatial_idx
        batch_indices = local_indices // spatial_size
        spatial_indices = local_indices % spatial_size
        flat_indices = (
            batch_indices * (C * spatial_size)
            + channel_id * spatial_size
            + spatial_indices
        )

        # Load grad_out and input with proper indexing
        grad_out = tl.load(grad_out_ptr + flat_indices, mask=mask, other=0.0).to(
            tl.float32
        )
        inp = tl.load(input_ptr + flat_indices, mask=mask, other=0.0).to(tl.float32)

        # Compute intermediate values
        xmu = inp - mean  # input - mean
        normalized = xmu * invstd  # normalized input

        # Accumulate reductions
        if COMPUTE_INPUT_G:
            acc_sum_dy += tl.sum(tl.where(mask, grad_out, 0.0))
            acc_sum_dy_xmu += tl.sum(tl.where(mask, grad_out * xmu, 0.0))

        if COMPUTE_GRAD_WEIGHT:
            acc_grad_weight += tl.sum(tl.where(mask, grad_out * normalized, 0.0))
        if COMPUTE_GRAD_BIAS:
            acc_grad_bias += tl.sum(tl.where(mask, grad_out, 0.0))

    # Store results
    if COMPUTE_INPUT_G:
        tl.store(sum_dy_ptr + channel_id, acc_sum_dy)
        tl.store(sum_dy_xmu_ptr + channel_id, acc_sum_dy_xmu)

    if COMPUTE_GRAD_WEIGHT:
        tl.store(grad_weight_ptr + channel_id, acc_grad_weight)
    if COMPUTE_GRAD_BIAS:
        tl.store(grad_bias_ptr + channel_id, acc_grad_bias)


@triton.jit
def _bn_reduce_partial_kernel(
    grad_out_ptr,
    input_ptr,
    mean_ptr,
    invstd_ptr,
    partials_ptr,
    C,
    spatial_size,
    elements_per_channel,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_INPUT_G: tl.constexpr,
    COMPUTE_GRAD_WEIGHT: tl.constexpr,
    COMPUTE_GRAD_BIAS: tl.constexpr,
):
    # 第一阶段：每个 program 负责一个 (channel, chunk)，求 4 个部分和
    channel_id = tl.program_id(0)
    chunk_id = tl.program_id(1)
    chunk_size = tl.cdiv(elements_per_channel, NUM_CHUNKS)
    chunk_start = chunk_id * chunk_size
    chunk_end = chunk_start + chunk_size

    mean = tl.load(mean_ptr + channel_id).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel_id).to(tl.float32)

    acc_sum_dy = 0.0
    acc_sum_dy_xmu = 0.0
    acc_grad_weight = 0.0
    acc_grad_bias = 0.0

    for offset in range(chunk_start, chunk_end, BLOCK_SIZE):
        local_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = (local_indices < elements_per_channel) & (local_indices < chunk_end)
        batch_indices = local_indices // spatial_size
        spatial_indices = local_indices % spatial_size
        flat_indices = (
            batch_indices * (C * spatial_size)
            + channel_id * spatial_size
            + spatial_indices
        )
        grad_out = tl.load(grad_out_ptr + flat_indices, mask=mask, other=0.0).to(
            tl.float32
        )
        inp = tl.load(input_ptr + flat_indices, mask=mask, other=0.0).to(tl.float32)
        xmu = inp - mean
        normalized = xmu * invstd
        if COMPUTE_INPUT_G:
            acc_sum_dy += tl.sum(tl.where(mask, grad_out, 0.0))
            acc_sum_dy_xmu += tl.sum(tl.where(mask, grad_out * xmu, 0.0))
        if COMPUTE_GRAD_WEIGHT:
            acc_grad_weight += tl.sum(tl.where(mask, grad_out * normalized, 0.0))
        if COMPUTE_GRAD_BIAS:
            acc_grad_bias += tl.sum(tl.where(mask, grad_out, 0.0))

    partial_base = channel_id * NUM_CHUNKS + chunk_id
    tl.store(partials_ptr + 0 * C * NUM_CHUNKS + partial_base, acc_sum_dy)
    tl.store(partials_ptr + 1 * C * NUM_CHUNKS + partial_base, acc_sum_dy_xmu)
    tl.store(partials_ptr + 2 * C * NUM_CHUNKS + partial_base, acc_grad_weight)
    tl.store(partials_ptr + 3 * C * NUM_CHUNKS + partial_base, acc_grad_bias)


@triton.jit
def _bn_reduce_combine_kernel(
    partials_ptr,
    sum_dy_ptr,
    sum_dy_xmu_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    C,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_BLOCK: tl.constexpr,
    COMPUTE_INPUT_G: tl.constexpr,
    COMPUTE_GRAD_WEIGHT: tl.constexpr,
    COMPUTE_GRAD_BIAS: tl.constexpr,
):
    # 第二阶段：沿 chunk 维合并部分和，得到每 channel 的最终 4 个输出
    channel_id = tl.program_id(0)
    chunk_offs = tl.arange(0, CHUNK_BLOCK)
    mask = chunk_offs < NUM_CHUNKS
    partial_base = channel_id * NUM_CHUNKS + chunk_offs
    if COMPUTE_INPUT_G:
        partial_sum_dy = tl.load(
            partials_ptr + 0 * C * NUM_CHUNKS + partial_base, mask=mask, other=0.0
        )
        partial_sum_dy_xmu = tl.load(
            partials_ptr + 1 * C * NUM_CHUNKS + partial_base, mask=mask, other=0.0
        )
        tl.store(sum_dy_ptr + channel_id, tl.sum(partial_sum_dy))
        tl.store(sum_dy_xmu_ptr + channel_id, tl.sum(partial_sum_dy_xmu))
    if COMPUTE_GRAD_WEIGHT:
        partial_grad_weight = tl.load(
            partials_ptr + 2 * C * NUM_CHUNKS + partial_base, mask=mask, other=0.0
        )
        tl.store(grad_weight_ptr + channel_id, tl.sum(partial_grad_weight))
    if COMPUTE_GRAD_BIAS:
        partial_grad_bias = tl.load(
            partials_ptr + 3 * C * NUM_CHUNKS + partial_base, mask=mask, other=0.0
        )
        tl.store(grad_bias_ptr + channel_id, tl.sum(partial_grad_bias))


def batch_norm_backward_reduce(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    weight: Tensor,
    input_g: bool,
    weight_g: bool,
    bias_g: bool,
) -> tuple:
    """
    Compute batch normalization backward reduce.

    Args:
        grad_out: Gradient of output (N, C, *spatial)
        input: Input tensor (N, C, *spatial)
        mean: Mean per channel (C,)
        invstd: Inverse standard deviation per channel (C,)
        weight: Weight per channel (C,) or None
        input_g: Whether to compute grad_input (controls sum_dy/sum_dy_xmu computation)
        weight_g: Whether to compute grad_weight
        bias_g: Whether to compute grad_bias

    Returns:
        (sum_dy, sum_dy_xmu, grad_weight, grad_bias) tuple
        Each is (C,) or None depending on the corresponding flag
    """
    logger.debug("GEMS_ENFLAME_GCU400 BATCH_NORM_BACKWARD_REDUCE")

    # Ensure contiguous memory layout
    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    invstd = invstd.contiguous()

    # Get dimensions
    C = input.shape[1]
    N = input.shape[0]
    spatial_size = input.numel() // (N * C)
    elements_per_channel = N * spatial_size

    # Allocate outputs based on flags
    # Note: sum_dy and sum_dy_xmu are always float32, grad_weight and grad_bias match input dtype
    sum_dy = (
        torch.empty(C, dtype=torch.float32, device=grad_out.device) if input_g else None
    )
    sum_dy_xmu = (
        torch.empty(C, dtype=torch.float32, device=grad_out.device) if input_g else None
    )
    grad_weight = (
        torch.empty(C, dtype=grad_out.dtype, device=grad_out.device)
        if weight_g
        else None
    )
    grad_bias = (
        torch.empty(C, dtype=grad_out.dtype, device=grad_out.device) if bias_g else None
    )

    # Handle dummy pointers for unused outputs
    if sum_dy is None:
        sum_dy = torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    if sum_dy_xmu is None:
        sum_dy_xmu = torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    if grad_weight is None:
        grad_weight = torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)
    if grad_bias is None:
        grad_bias = torch.empty(0, dtype=grad_out.dtype, device=grad_out.device)

    # 确定性分档：按每通道元素数选 tile 与 warp 数，替代 libtuner 运行时 autotune
    if elements_per_channel <= 1024:
        block_size, num_warps, num_stages = 512, 4, 2
    elif elements_per_channel <= 8192:
        block_size, num_warps, num_stages = 1024, 4, 2
    else:
        block_size, num_warps, num_stages = 4096, 16, 1

    total_elements = C * elements_per_channel
    with torch_device_fn.device(input.device):
        if total_elements >= 1_000_000:
            # 大 workload：两阶段归约把 6 个 program 扩成 C×48 个，打满 SIP
            num_chunks = 48
            # 2 字节 dtype 用中等 tile、4 字节 dtype 用大 tile，匹配各自的访存粒度
            partial_block_size = 2048 if grad_out.element_size() == 2 else 4096
            partials = torch.empty(
                (4, C, num_chunks), dtype=torch.float32, device=grad_out.device
            )
            _bn_reduce_partial_kernel[(C, num_chunks)](
                grad_out,
                input,
                mean,
                invstd,
                partials,
                C,
                spatial_size,
                elements_per_channel,
                NUM_CHUNKS=num_chunks,
                BLOCK_SIZE=partial_block_size,
                COMPUTE_INPUT_G=input_g,
                COMPUTE_GRAD_WEIGHT=weight_g,
                COMPUTE_GRAD_BIAS=bias_g,
                num_warps=4,
                num_stages=1,
            )
            _bn_reduce_combine_kernel[(C,)](
                partials,
                sum_dy,
                sum_dy_xmu,
                grad_weight,
                grad_bias,
                C,
                NUM_CHUNKS=num_chunks,
                CHUNK_BLOCK=triton.next_power_of_2(num_chunks),
                COMPUTE_INPUT_G=input_g,
                COMPUTE_GRAD_WEIGHT=weight_g,
                COMPUTE_GRAD_BIAS=bias_g,
                num_warps=4,
            )
        else:
            # 小 workload 单 kernel，避免额外 launch 与 partials 分配
            batch_norm_backward_reduce_kernel[(C,)](
                grad_out,
                input,
                mean,
                invstd,
                sum_dy,
                sum_dy_xmu,
                grad_weight,
                grad_bias,
                C,
                spatial_size,
                elements_per_channel,
                BLOCK_SIZE=block_size,
                COMPUTE_INPUT_G=input_g,
                COMPUTE_GRAD_WEIGHT=weight_g,
                COMPUTE_GRAD_BIAS=bias_g,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    # Return None for outputs that weren't computed
    return (
        sum_dy if input_g else None,
        sum_dy_xmu if input_g else None,
        grad_weight if weight_g else None,
        grad_bias if bias_g else None,
    )
