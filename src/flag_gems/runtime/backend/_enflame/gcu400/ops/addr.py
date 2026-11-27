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


@libentry()
@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_kernel(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    M,
    N,
    stride_input_m,
    stride_input_n,
    stride_vec1,
    stride_vec2,
    stride_output_m,
    stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    vec1_ptrs = vec1_ptr + offs_m * stride_vec1
    vec2_ptrs = vec2_ptr + offs_n * stride_vec2

    mask_m = offs_m < M
    mask_n = offs_n < N

    vec1 = tl.load(vec1_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    vec2 = tl.load(vec2_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    input_ptrs = (
        input_ptr + offs_m[:, None] * stride_input_m + offs_n[None, :] * stride_input_n
    )

    mask_2d = mask_m[:, None] & mask_n[None, :]
    input_val = tl.load(input_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    result = beta * input_val + alpha * (vec1[:, None] * vec2[None, :])

    output_ptrs = (
        output_ptr
        + offs_m[:, None] * stride_output_m
        + offs_n[None, :] * stride_output_n
    )
    tl.store(output_ptrs, result, mask=mask_2d)


@libentry()
@triton.jit(do_not_specialize=["beta", "alpha"])
def addr_flat_kernel(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    N,
    stride_input_m,
    stride_vec1,
    num_tiles,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(pid, num_tiles, NPROG):
        base = tile * BLOCK
        row = base // N
        col0 = base - row * N
        offs = tl.arange(0, BLOCK)
        vec1 = tl.load(vec1_ptr + row * stride_vec1).to(tl.float32)
        vec2 = tl.load(vec2_ptr + col0 + offs).to(tl.float32)
        input_val = tl.load(input_ptr + row * stride_input_m + col0 + offs).to(
            tl.float32
        )
        result = beta * input_val + alpha * vec1 * vec2
        tl.store(output_ptr + base + offs, result.to(output_ptr.dtype.element_ty))


_FLAT_BLOCKS = (4096, 2048, 1024, 512, 256, 128, 64, 32)
_NUM_SPS = 48


def _pick_flat_block(N):
    for block in _FLAT_BLOCKS:
        if N % block == 0:
            return block
    return None


def addr(input, vec1, vec2, *, beta=1, alpha=1):
    logger.debug("GEMS_ENFLAME ADDR")
    if vec1.dim() != 1 or vec2.dim() != 1:
        raise ValueError("addr: expected 1-D vectors")

    # ATen types alpha/beta as Scalar: a 0-dim tensor argument must behave as a
    # python number. Passing the tensor straight through also trips a Triton
    # 3.5 compile error (fp16 pointer vs fp32 scalar operand).
    if torch.is_tensor(beta) and beta.numel() == 1:
        beta = beta.item()
    if torch.is_tensor(alpha) and alpha.numel() == 1:
        alpha = alpha.item()

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

    total = M * N
    if total == 0:
        return out

    if input_broadcasted.stride(1) == 1 and vec2.stride(0) == 1:
        block = _pick_flat_block(N)
        if block is not None:
            tiles = total // block
            if tiles <= _NUM_SPS:
                nprog = tiles
                num_warps = 1
            else:
                nprog = _NUM_SPS
                num_warps = 4
            with torch_device_fn.device(input.device):
                addr_flat_kernel[(nprog,)](
                    input_broadcasted,
                    vec1,
                    vec2,
                    out,
                    beta,
                    alpha,
                    N,
                    input_broadcasted.stride(0),
                    vec1.stride(0),
                    tiles,
                    BLOCK=block,
                    NPROG=nprog,
                    num_warps=num_warps,
                )
            return out

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    with torch_device_fn.device(input.device):
        addr_kernel[grid](
            input_broadcasted,
            vec1,
            vec2,
            out,
            beta,
            alpha,
            M,
            N,
            input_broadcasted.stride(0),
            input_broadcasted.stride(1),
            vec1.stride(0),
            vec2.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=1,
        )
    return out
