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

_CONTIGUOUS_BLOCK_SIZE_N = 2048
_CONTIGUOUS_NUM_WARPS = 8
_STRIDED_BLOCK_SIZE_N = 1024


@libentry()
@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_contiguous_kernel(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    blocks_per_row: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    row = pid // blocks_per_row
    block = pid - row * blocks_per_row
    col_start = block * BLOCK_SIZE_N

    input_block = tl.make_block_ptr(
        input_ptr + row * N,
        shape=(N,),
        strides=(1,),
        offsets=(col_start,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    vec2_block = tl.make_block_ptr(
        vec2_ptr,
        shape=(N,),
        strides=(1,),
        offsets=(col_start,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    output_block = tl.make_block_ptr(
        output_ptr + row * N,
        shape=(N,),
        strides=(1,),
        offsets=(col_start,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    input_val = tl.load(input_block, boundary_check=(0,)).to(tl.float32)
    vec1 = tl.load(vec1_ptr + row).to(tl.float32)
    vec2 = tl.load(vec2_block, boundary_check=(0,)).to(tl.float32)
    result = beta * input_val + alpha * vec1 * vec2
    tl.store(
        output_block,
        result.to(output_block.type.element_ty),
        boundary_check=(0,),
    )


@libentry()
@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_strided_kernel(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    N: tl.constexpr,
    stride_input_m: tl.constexpr,
    stride_input_n: tl.constexpr,
    stride_vec1: tl.constexpr,
    stride_vec2: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    input_offsets = row * stride_input_m + cols * stride_input_n
    input_val = tl.load(input_ptr + input_offsets, mask=mask).to(tl.float32)
    vec1 = tl.load(vec1_ptr + row * stride_vec1).to(tl.float32)
    vec2 = tl.load(vec2_ptr + cols * stride_vec2, mask=mask).to(tl.float32)

    result = beta * input_val + alpha * vec1 * vec2
    tl.store(output_ptr + row * N + cols, result, mask=mask)


def addr(input, vec1, vec2, *, beta=1, alpha=1):
    logger.debug("GEMS ADDR")
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("addr: expected 1-D vectors")

    M = vec1.shape[0]
    N = vec2.shape[0]
    output_shape = (M, N)

    try:
        input_broadcasted = torch.broadcast_to(input, output_shape)
    except RuntimeError:
        raise ValueError(
            f"addr: input tensor of shape {input.shape} cannot be broadcast to output shape {output_shape}"
        )
    out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    if M == 0 or N == 0:
        return out

    with torch_device_fn.device(input.device):
        if (
            input_broadcasted.is_contiguous()
            and vec1.is_contiguous()
            and vec2.is_contiguous()
        ):
            grid = (M * triton.cdiv(N, _CONTIGUOUS_BLOCK_SIZE_N),)
            addr_contiguous_kernel[grid](
                input_broadcasted,
                vec1,
                vec2,
                out,
                beta,
                alpha,
                N,
                BLOCK_SIZE_N=_CONTIGUOUS_BLOCK_SIZE_N,
                num_warps=_CONTIGUOUS_NUM_WARPS,
            )
        else:
            grid = lambda META: (M, triton.cdiv(N, META["BLOCK_SIZE_N"]))
            addr_strided_kernel[grid](
                input_broadcasted,
                vec1,
                vec2,
                out,
                beta,
                alpha,
                N,
                input_broadcasted.stride(0),
                input_broadcasted.stride(1),
                vec1.stride(0),
                vec2.stride(0),
                BLOCK_SIZE_N=_STRIDED_BLOCK_SIZE_N,
            )
    return out
