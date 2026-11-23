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

logger = logging.getLogger(__name__)


@triton.jit
def bilinear_kernel(
    input1_ptr,
    input2_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K1,
    K2,
    stride_i1m,
    stride_i1k,
    stride_i2m,
    stride_i2k,
    stride_wn,
    stride_om,
    BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Bilinear kernel: y[m, o] = sum_{i, j} x1[m, i] * W[o, i, j] * x2[m, j] + b[o]
    - input1: (M, K1) where M is the flattened batch size, K1 is in1_features
    - input2: (M, K2) where K2 is in2_features
    - weight: treated as (N, K1 * K2), i.e. W[o, i, j] flattened over (i, j)
    - bias: (N,) optional
    - output: (M, N)

    The 2D (in1_features, in2_features) reduction is merged into a single GEMM
    reduction of length K1 * K2: the outer product Z[m, i * K2 + j] =
    x1[m, i] * x2[m, j] is the left operand and the flattened weight
    W_flat[o, i * K2 + j] the right one, so y = Z @ W_flat^T is fully handed to
    tl.dot. Each program owns one (BLOCK_M, BLOCK_N) output tile: the current j
    slice of x2 is loaded once and reused across the i loop, and every load
    stays contiguous along the last axis.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    num_j_chunks = tl.cdiv(K2, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for j_chunk in range(0, num_j_chunks):
        offs_j = j_chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_j = offs_j < K2

        # The current j slice of x2 is reused by the whole i loop, so load it once.
        # The reduction length can reach K1 * K2 (thousands), so the outer product
        # must stay in fp32: rounding z to fp16/bf16 accumulates error past the
        # tolerance enforced by tests/test_bilinear.py.
        x2_slice = tl.load(
            input2_ptr + offs_m[:, None] * stride_i2m + offs_j[None, :] * stride_i2k,
            mask=mask_m[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)

        for i in range(0, K1):
            x1_col = tl.load(
                input1_ptr + offs_m * stride_i1m + i * stride_i1k,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            w_rows = tl.load(
                weight_ptr
                + offs_n[:, None] * stride_wn
                + (i * K2 + offs_j)[None, :],
                mask=mask_n[:, None] & mask_j[None, :],
                other=0.0,
            ).to(tl.float32)
            z_tile = x1_col[:, None] * x2_slice
            acc += tl.dot(z_tile, tl.trans(w_rows), allow_tf32=False)

    if BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]

    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        acc.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def bilinear(input1, input2, weight, bias=None):
    """
    Applies a bilinear transformation to the incoming data:
        y = x1^T A x2 + b

    Args:
        input1: Tensor of shape (N, *, in1_features)
        input2: Tensor of shape (N, *, in2_features)
        weight: Tensor of shape (out_features, in1_features, in2_features)
        bias: Tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (N, *, out_features)
    """
    logger.debug("GEMS_ENFLAME BILINEAR")

    # All but the last dimension of input1 and input2 must match.
    batch_dims = input1.shape[:-1]
    M = 1
    for dim in batch_dims:
        M *= dim
    K1 = input1.shape[-1]  # in1_features
    K2 = input2.shape[-1]  # in2_features
    N = weight.shape[0]  # out_features

    input1_flat = input1.reshape(M, K1)
    input2_flat = input2.reshape(M, K2)
    # Flatten the weight to (N, K1 * K2) to feed the 1D GEMM reduction in the kernel.
    weight_flat = weight.contiguous().view(N, K1 * K2)

    output = torch.empty((M, N), device=input1.device, dtype=input1.dtype)

    # tl.dot needs every tile dim >= 16; shrink blocks to 16 for small shapes so no
    # program is spent on padding lanes only.
    BLOCK_M = max(16, min(32, triton.next_power_of_2(M)))
    BLOCK_N = max(16, min(32, triton.next_power_of_2(N)))
    BLOCK_K = max(16, min(128, triton.next_power_of_2(K2)))

    # Shrinking the tiles further (16x16) to raise occupancy when the program count
    # is below one SIPS wave (24) was measured and rejected: shape (128,256,256,128)
    # dropped from 2.37x to 1.14x native. With K1 * K2 = 65536 the per-program work
    # already hides the low parallelism, and smaller tiles only cost tl.dot
    # efficiency, so 32x32 is kept.
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (min(num_pid_m * num_pid_n, 65535),)

    with torch_device_fn.device(input1.device):
        bilinear_kernel[grid](
            input1_flat,
            input2_flat,
            weight_flat,
            bias if bias is not None else weight_flat,
            output,
            M,
            N,
            K1,
            K2,
            input1_flat.stride(0),
            input1_flat.stride(1),
            input2_flat.stride(0),
            input2_flat.stride(1),
            weight_flat.stride(0),
            output.stride(0),
            BIAS=bias is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    output = output.view(*batch_dims, N)
    return output
