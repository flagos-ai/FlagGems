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

_GROUP_SIZE_M = 8


def _tile(n: int, cap: int) -> int:
    block = 16
    while block < n and block < cap:
        block *= 2
    return block


def _choose_blocks(m: int, n: int, k: int) -> tuple[int, int, int]:
    """Tiles measured on the profile shapes (M=16, K=2048).

    BLOCK_M follows M so a decode row is not padded to 256.
    BLOCK_K=1024 cuts the K loop; BLOCK_N=128 is faster than 256/512
    until N is large (the 31040-wide gate).
    """
    block_m = 16 if m <= 16 else _tile(m, 64)
    if n >= 8192:
        block_n = 512
    elif n >= 128:
        block_n = 128
    else:
        block_n = _tile(max(n, 1), 64)
    if k >= 1024:
        block_k = 1024
    elif k >= 512:
        block_k = 512
    else:
        block_k = _tile(max(k, 1), 64)
    return block_m, block_n, block_k


@triton.jit
def _linear_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BIAS: tl.constexpr,
):
    """C[M,N] = A[M,K] @ W[N,K]^T. W is addressed as (K, N) via its strides."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    # W is stored (N, K). View it as (K, N) so the dot matches matmul_bf16.
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")
        accumulator = tl.dot(a, b, accumulator)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_SIZE_K, 0))

    if BIAS:
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def linear(input, weight, bias=None):
    logger.debug("GEMS_TSINGMICRO LINEAR")
    single_1d = input.dim() == 1
    if single_1d:
        input = input.unsqueeze(0)

    batch_dims = input.shape[:-1]
    m = 1
    for dim in batch_dims:
        m *= dim
    k = input.shape[-1]
    n = weight.shape[0]
    a = input.reshape(m, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    block_m, block_n, block_k = _choose_blocks(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _linear_kernel[grid](
        a,
        weight,
        bias if bias is not None else out,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        weight.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=_GROUP_SIZE_M,
        BIAS=bias is not None,
        num_warps=1,
    )
    out = out.view(*batch_dims, n)
    if single_1d:
        out = out.squeeze(0)
    return out
