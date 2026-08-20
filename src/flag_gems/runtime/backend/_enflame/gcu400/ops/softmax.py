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

GRID_DIM_X = 24


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ONE_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    if ONE_TILE:
        for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
            out_block = tl.make_block_ptr(
                base=out_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            grad_block = tl.make_block_ptr(
                base=out_grad_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            out_tile = tl.load(
                out_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            grad_tile = tl.load(
                grad_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            scale = tl.sum(out_tile * grad_tile, axis=1)
            in_grad_tile = out_tile * (grad_tile - scale[:, None])
            in_grad_block = tl.make_block_ptr(
                base=in_grad_ptr,
                shape=(M, N),
                strides=(N, 1),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            tl.store(
                in_grad_block,
                in_grad_tile.to(in_grad_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )
    else:
        for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
            scale = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for n_start in tl.range(0, N, BLOCK_N):
                out_block = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                grad_block = tl.make_block_ptr(
                    base=out_grad_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                out_tile = tl.load(
                    out_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                grad_tile = tl.load(
                    grad_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                scale += out_tile * grad_tile
            scale = tl.sum(scale, axis=1)

            for n_start in tl.range(0, N, BLOCK_N):
                out_block = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                grad_block = tl.make_block_ptr(
                    base=out_grad_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                in_grad_block = tl.make_block_ptr(
                    base=in_grad_ptr,
                    shape=(M, N),
                    strides=(N, 1),
                    offsets=(m_start, n_start),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                out_tile = tl.load(
                    out_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                grad_tile = tl.load(
                    grad_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                in_grad_tile = out_tile * (grad_tile - scale[:, None])
                tl.store(
                    in_grad_block,
                    in_grad_tile.to(in_grad_ptr.dtype.element_ty),
                    boundary_check=(0, 1),
                )


@libentry()
@triton.jit(do_not_specialize=["N"])
def softmax_backward_scale_kernel(
    out_ptr,
    out_grad_ptr,
    partial_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Stage 1 for M=1: compute partial scale sums across programs."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for n_start in tl.range(pid * BLOCK_N, N, num_programs * BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        out_val = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_val = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += out_val * grad_val
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N", "num_partials"])
def softmax_backward_grad_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    partial_ptr,
    N,
    num_partials,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Stage 2 for M=1: read partial sums, compute total scale, compute in_grad."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    p_offsets = tl.arange(0, BLOCK_P)
    p_mask = p_offsets < num_partials
    partials = tl.load(partial_ptr + p_offsets, mask=p_mask, other=0.0)
    scale = tl.sum(partials, axis=0)

    for n_start in tl.range(pid * BLOCK_N, N, num_programs * BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        out_val = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_val = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        in_grad_val = out_val * (grad_val - scale)
        tl.store(
            in_grad_ptr + n_offsets,
            in_grad_val.to(in_grad_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_k_programs = tl.num_programs(1)

    for k_start in tl.range(pid_k * BLOCK_K, K, num_k_programs * BLOCK_K):
        scale = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        for n_start in tl.range(0, N, BLOCK_N):
            out_block = tl.make_block_ptr(
                base=out_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            grad_block = tl.make_block_ptr(
                base=out_grad_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            out_tile = tl.load(
                out_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            grad_tile = tl.load(
                grad_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            scale += out_tile * grad_tile
        scale = tl.sum(scale, axis=0)

        for n_start in tl.range(0, N, BLOCK_N):
            out_block = tl.make_block_ptr(
                base=out_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            grad_block = tl.make_block_ptr(
                base=out_grad_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            in_grad_block = tl.make_block_ptr(
                base=in_grad_ptr + pid_m * N * K,
                shape=(N, K),
                strides=(K, 1),
                offsets=(n_start, k_start),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            out_tile = tl.load(
                out_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            grad_tile = tl.load(
                grad_block, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            in_grad_tile = out_tile * (grad_tile - scale[None, :])
            tl.store(
                in_grad_block,
                in_grad_tile.to(in_grad_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_ENFLAME SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            BLOCK_K = min(triton.next_power_of_2(K), 1024)
            BLOCK_N = max(1, min(64, 32768 // BLOCK_K))
            num_k_programs = min(GRID_DIM_X, triton.cdiv(K, BLOCK_K))
            grid = (M, num_k_programs)
            softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                K,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                num_warps=1,
            )
        elif M == 1 and N > 2048:
            BLOCK_N = 4096
            num_programs = min(GRID_DIM_X, triton.cdiv(N, BLOCK_N))
            partial = torch.empty(
                num_programs, dtype=torch.float32, device=in_grad.device
            )
            softmax_backward_scale_kernel[(num_programs,)](
                output,
                grad_output,
                partial,
                N,
                BLOCK_N=BLOCK_N,
                num_warps=1,
            )
            BLOCK_P = triton.next_power_of_2(num_programs)
            softmax_backward_grad_kernel[(num_programs,)](
                output,
                grad_output,
                in_grad,
                partial,
                N,
                num_programs,
                BLOCK_N=BLOCK_N,
                BLOCK_P=BLOCK_P,
                num_warps=1,
            )
        else:
            BLOCK_N = min(triton.next_power_of_2(N), 2048)
            BLOCK_M = max(1, min(128, 32768 // BLOCK_N))
            ONE_TILE = N <= BLOCK_N
            num_programs = min(GRID_DIM_X, triton.cdiv(M, BLOCK_M))
            grid = (num_programs,)
            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ONE_TILE=ONE_TILE,
                num_warps=1,
            )
    return in_grad
