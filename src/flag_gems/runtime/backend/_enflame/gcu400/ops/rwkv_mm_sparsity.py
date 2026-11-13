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

"""gcu400 rwkv_mm_sparsity: 用 tl.dot(M=16) 复现 aten GEMM 的数值路径。"""

import torch
import triton
import triton.language as tl


@triton.jit
def rwkv_dot_kernel(
    k_ptr,
    v_ptr,
    output_ptr,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        k_tile = tl.load(
            k_ptr + kk[None, :] + offs_m[:, None] * 0,
            mask=kk[None, :] < K,
            other=0.0,
        ).to(tl.float32)
        v_tile = tl.load(
            v_ptr + kk[:, None] * N + offs_n[None, :],
            mask=(kk[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(k_tile, v_tile, acc)
    row0 = tl.sum(tl.where(offs_m[:, None] == 0, acc, 0.0), axis=0)
    tl.store(output_ptr + offs_n, row0.to(output_ptr.dtype.element_ty), mask=offs_n < N)


def rwkv_mm_sparsity(k: torch.Tensor, v: torch.Tensor):
    assert k.dim() == 1 and v.dim() == 2
    assert k.size(0) == v.size(0)
    N, K = v.size(1), k.size(0)
    output = torch.empty(N, device=k.device, dtype=k.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 64
    grid = (triton.cdiv(N, BLOCK_N),)
    rwkv_dot_kernel[grid](
        k,
        v,
        output,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return output
