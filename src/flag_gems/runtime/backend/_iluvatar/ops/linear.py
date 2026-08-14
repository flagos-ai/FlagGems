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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("linear"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("linear"))
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bn,
    BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A = input_ptr + (ram[:, None] * stride_im + rk[None, :] * stride_ik)
    W = weight_ptr + (rk[:, None] * stride_wk + rbn[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for _ in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(W)
            acc += tl.dot(a, b, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ik
            W += BLOCK_K * SPLIT_K * stride_wk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for _ in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(W)
            acc += tl.dot(a, b, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ik
            W += BLOCK_K * SPLIT_K * stride_wk

        _0 = tl.zeros((1, 1), dtype=output_ptr.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(W, mask=rk[:, None] < k_remaining, other=_0)
        acc += tl.dot(a, b, allow_tf32=False)

    if BIAS:
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
        acc = acc + bias[None, :]

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc.to(output_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(C, acc.to(output_ptr.dtype.element_ty), mask=mask)


def linear(input, weight, bias=None):
    logger.debug("GEMS_ILUVATAR LINEAR")

    input_dim = input.dim()
    if input_dim == 1:
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    M = batch_size
    K = input.shape[-1]
    N = weight.shape[0]

    input_flat = input.view(M, K)
    weight = weight.contiguous()
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META.get("SPLIT_K", 1),
    )

    with torch_device_fn.device(input.device):
        linear_kernel[grid](
            input_flat,
            weight,
            bias if bias is not None else weight,
            output,
            M,
            N,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            bias.stride(0) if bias is not None else 0,
            BIAS=bias is not None,
            GROUP_M=8,
            SPLIT_K=1,
        )

    output = output.view(*batch_dims, N)
    if single_1d:
        output = output.squeeze(0)
    return output
