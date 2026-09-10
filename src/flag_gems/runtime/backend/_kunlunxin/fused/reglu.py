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
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def heur_tile_m(args):
    return triton.cdiv(args["M"], 12)  # cluster_num


def heru_tile_n(args):
    import builtins

    return builtins.min(args["N"], 8192)


@libentry()
@triton.jit(do_not_specialize=["num_tasks"])
def dreglu_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    num_tasks,
    N: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_CTA: tl.constexpr,
    ONE_TILE: tl.constexpr,
):
    # XPU-specialized dreglu: 1D flattened "pair" kernel.
    #
    # The 2D (BLOCK_M x BLOCK_N) tiling of the generic kernel is pathological on
    # this backend: the XPU CoreTiling pass collapses the block to a single row
    # and serializes the BLOCK_M rows one by one, and the grad_output pointer
    # arithmetic is inferred as a discrete gather (offsetState=-1 / stride=-1),
    # which at large shapes drives latency from ~0.4ms (TE) to 8.5ms.
    #
    # Instead we iterate over the M*N "pairs" (one pair per grad_output element,
    # each producing the a-half and b-half of one grad_input row). Resolving the
    # row with tid // N keeps every load/store on wide contiguous ranges:
    #   * grad_output is contiguous (N per row),
    #   * input a-half / b-half are contiguous N-elements per row,
    # so the backend emits full-width block DMA instead of row-serialized tiles.
    # grid = (12,) with the fixed-tile / grid-stride pattern used by the other
    # XPU pointwise kernels (copysign_, special_erfinv, native_dropout_backward).
    # Masked lanes are clamped to index 0 so out-of-range tail-tile addresses are
    # never dereferenced (the masked store discards them anyway).
    pid = tl.program_id(0)
    if ONE_TILE:
        tid = pid * TILE + tl.arange(0, TILE)
        mask = tid < num_tasks
        a_off = (tid // N) * N
        grad_out = tl.load(grad_output_ptr + tid, mask=mask).to(tl.float32)
        block_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
        block_b = tl.load(input_ptr + tid + a_off + N, mask=mask).to(tl.float32)
        relu_a = tl.maximum(block_a, 0.0)
        d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
        grad_a = grad_out * d_relu_a * block_b
        grad_b = grad_out * relu_a
        tl.store(
            grad_input_ptr + tid + a_off,
            grad_a.to(input_ptr.type.element_ty),
            mask=mask,
        )
        tl.store(
            grad_input_ptr + tid + a_off + N,
            grad_b.to(input_ptr.type.element_ty),
            mask=mask,
        )
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, TILES_PER_CTA):
            tile_id = pid + j * num_ctas
            tid = tile_id * TILE + tl.arange(0, TILE)
            mask = tid < num_tasks
            a_off = (tid // N) * N
            grad_out = tl.load(grad_output_ptr + tid, mask=mask).to(tl.float32)
            block_a = tl.load(input_ptr + tid + a_off, mask=mask).to(tl.float32)
            block_b = tl.load(input_ptr + tid + a_off + N, mask=mask).to(tl.float32)
            relu_a = tl.maximum(block_a, 0.0)
            d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
            grad_a = grad_out * d_relu_a * block_b
            grad_b = grad_out * relu_a
            tl.store(
                grad_input_ptr + tid + a_off,
                grad_a.to(input_ptr.type.element_ty),
                mask=mask,
            )
            tl.store(
                grad_input_ptr + tid + a_off + N,
                grad_b.to(input_ptr.type.element_ty),
                mask=mask,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gated_activation"),
    key=["M", "N_OUT"],
)
@triton.jit
def reglu_kernel(
    x_ptr,
    y_ptr,
    M,
    N_OUT,
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptr_a = x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x_ptr_b = (
        x_ptr + offs_m[:, None] * stride_x_m + (offs_n[None, :] + N_OUT) * stride_x_n
    )
    y_ptr = y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    block_a = tl.load(x_ptr_a, mask=mask, other=0.0)
    block_b = tl.load(x_ptr_b, mask=mask, other=0.0)
    gate = tl.where(block_a > 0, block_a, 0.0)
    output = gate * block_b
    tl.store(y_ptr, output, mask=mask)


def reglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    N_OUT = last_dim // 2
    M = input_tensor.numel() // last_dim
    if input_tensor.numel() == 0:
        output_shape = (*shape[:-1], N_OUT)
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(
        (M, N_OUT), device=input_tensor.device, dtype=input_tensor.dtype
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N_OUT, META["BLOCK_N"]),
    )
    reglu_kernel[grid](
        input_2d,
        output_2d,
        M,
        N_OUT,
        input_2d.stride(0),
        input_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


def dreglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS DREGLU")
    shape = input_tensor.shape
    if shape[:-1] != grad_output.shape[:-1] or shape[-1] != 2 * grad_output.shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {shape} vs grad_output {grad_output.shape}"
        )
    M = grad_output.numel() // grad_output.shape[-1]
    N = grad_output.shape[-1]
    grad_output_2d = grad_output.contiguous().view(M, N)
    input_2d = input_tensor.contiguous().view(M, 2 * N)
    grad_input = torch.empty_like(input_2d)
    num_tasks = grad_output_2d.numel()
    if num_tasks == 0:
        return grad_input.view(shape)
    num_ctas = 12
    num_tiles = num_ctas
    tile = triton.next_power_of_2(triton.cdiv(num_tasks, num_tiles))
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    dreglu_kernel[(num_ctas, 1, 1)](
        grad_output_2d,
        input_2d,
        grad_input,
        num_tasks,
        N=N,
        TILE=tile,
        TILES_PER_CTA=tiles_per_cta,
        ONE_TILE=tiles_per_cta == 1,
    )
    return grad_input.view(shape)
