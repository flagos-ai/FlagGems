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

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["hidden_size", "topk"],
)
@triton.jit
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token

    for expert_idx in range(topk):
        expert_ptr = input_base + expert_idx * input_stride_topk
        expert_data = tl.load(expert_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
        acc += expert_data
    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets

    tl.store(
        output_ptr_pos,
        acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc,
        mask=hidden_mask,
    )


@triton.jit
def _moe_sum_ep_kernel(
    input_ptr,
    output_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_tokens,
    topk,
    hidden_size,
    num_global_experts,
    local_num_experts,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    topk_ids_stride_token,
    topk_ids_stride_topk,
    BLOCK_SIZE: tl.constexpr,
):
    """Combine only routes owned by the current expert-parallel rank."""
    token_idx = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token
    ids_base = topk_ids_ptr + token_idx * topk_ids_stride_token
    for route_idx in range(topk):
        global_expert_raw = tl.load(ids_base + route_idx * topk_ids_stride_topk)
        valid_global_expert = (global_expert_raw >= 0) & (
            global_expert_raw < num_global_experts
        )
        safe_global_expert = tl.where(valid_global_expert, global_expert_raw, 0).to(
            tl.int64
        )
        local_expert_raw = tl.load(
            expert_map_ptr + safe_global_expert,
            mask=valid_global_expert,
            other=-1,
        )
        local_route = (
            valid_global_expert
            & (local_expert_raw >= 0)
            & (local_expert_raw < local_num_experts)
        )
        route_ptr = input_base + route_idx * input_stride_topk
        route_data = tl.load(
            route_ptr + hidden_offsets,
            mask=hidden_mask & local_route,
            other=0.0,
        )
        acc += route_data

    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets
    tl.store(output_ptr_pos, acc, mask=hidden_mask)


# Keep the generic API autotuned, while retaining direct access to the shared
# JIT body for shape-specific deterministic launches.
moe_sum_ep_kernel = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["hidden_size", "topk", "local_num_experts"],
)(_moe_sum_ep_kernel)

_SUPPORTED_FIXED_EP_SUM_CONFIGS = frozenset(
    {
        (256, 2),
        (512, 2),
        (1024, 4),
    }
)


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    logger.debug("GEMS MOE SUM")
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()
    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    moe_sum_kernel[grid](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
    )


def moe_sum_ep(
    input: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    local_num_experts: int,
    *,
    fixed_block_size: int | None = None,
    fixed_num_warps: int | None = None,
):
    """Sum local EP routes without requiring remote cache rows to be zeroed."""
    logger.debug("GEMS MOE SUM EP")
    if (fixed_block_size is None) != (fixed_num_warps is None):
        raise ValueError(
            "fixed_block_size and fixed_num_warps must be provided together"
        )
    fixed_config = None
    if fixed_block_size is not None and fixed_num_warps is not None:
        fixed_config = (fixed_block_size, fixed_num_warps)
        if fixed_config not in _SUPPORTED_FIXED_EP_SUM_CONFIGS:
            raise ValueError(
                "unsupported fixed moe_sum_ep config "
                f"{fixed_config}; expected one of "
                f"{sorted(_SUPPORTED_FIXED_EP_SUM_CONFIGS)}"
            )
    num_tokens, topk, hidden_size = input.shape
    if topk_ids.shape != (num_tokens, topk):
        raise ValueError(
            f"topk_ids must have shape {(num_tokens, topk)}, "
            f"got {tuple(topk_ids.shape)}"
        )
    if expert_map.ndim != 1:
        raise ValueError("expert_map must be one-dimensional")
    input_strides = input.stride()
    output_strides = output.stride()
    topk_ids_strides = topk_ids.stride()
    kernel_args = (
        input,
        output,
        topk_ids,
        expert_map,
        num_tokens,
        topk,
        hidden_size,
        expert_map.numel(),
        local_num_experts,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
        topk_ids_strides[0],
        topk_ids_strides[1],
    )
    if fixed_config is None:
        grid = lambda meta: (
            num_tokens,
            triton.cdiv(hidden_size, meta["BLOCK_SIZE"]),
        )
        moe_sum_ep_kernel[grid](*kernel_args)
    else:
        block_size, num_warps = fixed_config
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))
        _moe_sum_ep_kernel[grid](
            *kernel_args,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
