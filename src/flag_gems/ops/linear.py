# Copyright 2026, The FlagOS Contributors.
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


_GEMV_MAX_M = 5

_GEMV_BLOCK_N = 64
_GEMV_BLOCK_K = 128
_GEMV_SPLIT_K = 16
_GEMV_NUM_WARPS = 4
_GEMV_NUM_STAGES = 3

# Score threshold on M*(N*K)**1.5 for materialising an fp32 weight as (K, N).
_TRANSPOSE_MIN_SCORE = 1.5e11


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("linear"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
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
    # Bias is present or not
    BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear kernel: y = x @ W^T + b
    - input: (M, K) where M is batch size (flattened), K is in_features
    - weight: (N, K) where N is out_features
    - bias: (N,) optional
    - output: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)

    weight_ptrs = weight_ptr + (
        offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # The m and n side of both masks is loop-invariant; only the k side moves.
    input_mask_m = offs_m < M
    weight_mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # The k side of both masks moves with the loop; the m and n sides do not.
        mask_k = offs_k < (K - k * BLOCK_SIZE_K)

        # Load input and weight blocks
        a = tl.load(input_ptrs, mask=input_mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(
            weight_ptrs, mask=mask_k[:, None] & weight_mask_n[None, :], other=0.0
        )

        # Compute dot product
        accumulator += tl.dot(a, b, allow_tf32=False)

        # Move to next block
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    output_ptrs = output_ptr + (
        offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )

    output_mask = input_mask_m[:, None] & weight_mask_n[None, :]

    # Add bias if present
    if BIAS:
        bias_ptrs = bias_ptr + offs_n * stride_bn
        bias = tl.load(bias_ptrs, mask=weight_mask_n, other=0.0)
        accumulator = accumulator + bias

    # Store result
    output = accumulator.to(output_ptr.dtype.element_ty)
    tl.store(output_ptrs, output, mask=output_mask)


@triton.jit
def linear_gemv_kernel(
    input_ptr,
    weight_ptr,
    partial_ptr,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_pm,
    stride_pn,
    stride_pk,
    K_CHUNK,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """partial[m, sk, n] = sum over this split's k range of x[m, k] * W[n, k]; the
    caller sets BLOCK_SIZE_M == M, so the work is exactly M*N*K multiply-adds."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    k_start = pid_k * K_CHUNK
    k_end = tl.minimum(k_start + K_CHUNK, K)

    for m in range(0, BLOCK_SIZE_M):
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for k in range(0, tl.cdiv(K_CHUNK, BLOCK_SIZE_K)):
            offs_k = k_start + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < k_end
            x = tl.load(
                input_ptr + m * stride_im + offs_k * stride_ik,
                mask=mask_k,
                other=0.0,
            )
            w = tl.load(
                weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=mask_k[None, :] & mask_n[:, None],
                other=0.0,
            )
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
        tl.store(
            partial_ptr + m * stride_pm + pid_k * stride_pk + offs_n * stride_pn,
            acc,
            mask=mask_n,
        )


@triton.jit
def linear_gemv_reduce_kernel(
    partial_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    stride_pm,
    stride_pn,
    stride_pk,
    stride_om,
    stride_on,
    stride_bn,
    SPLIT_K,
    BIAS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Sum the split-k partials, add bias, cast to the output dtype."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    for m in range(0, M):
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for sk in range(0, SPLIT_K):
            acc += tl.load(
                partial_ptr + m * stride_pm + sk * stride_pk + offs_n * stride_pn,
                mask=mask_n,
                other=0.0,
            )
        if BIAS:
            acc += tl.load(bias_ptr + offs_n * stride_bn, mask=mask_n, other=0.0)
        tl.store(
            output_ptr + m * stride_om + offs_n * stride_on,
            acc.to(output_ptr.dtype.element_ty),
            mask=mask_n,
        )


@triton.jit
def linear_transpose_kernel(
    src_ptr,
    dst_ptr,
    N,
    K,
    stride_sn,
    stride_sk,
    stride_dk,
    stride_dn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """dst (K, N) = src (N, K)^T, both read through explicit strides. Stands in for
    weight.t().contiguous(), which faults in the registered copy_ path."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    src_ptrs = src_ptr + offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    src_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    value = tl.load(src_ptrs, mask=src_mask, other=0.0)

    dst_ptrs = dst_ptr + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn
    dst_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(dst_ptrs, tl.trans(value), mask=dst_mask)


def _transpose_weight(weight):
    """Materialise a (K, N) contiguous copy of an (N, K) weight."""
    N, K = weight.shape
    out = torch.empty((K, N), device=weight.device, dtype=weight.dtype)
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    with torch_device_fn.device(weight.device):
        linear_transpose_kernel[grid](
            weight,
            out,
            N,
            K,
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            num_warps=4,
        )
    return out


def _bias_stride(bias):
    return bias.stride(-1) if bias is not None else 0


def _may_transpose(weight, M):
    """Whether this shape is worth a (K, N) copy."""
    N, K = weight.shape
    return M * (N * K) ** 1.5 >= _TRANSPOSE_MIN_SCORE


def _linear_gemv(input_flat, weight, bias, M, K, N):
    """y = x @ W^T + b for small M, without any BLOCK_SIZE_M padding."""
    split_k = max(1, min(_GEMV_SPLIT_K, triton.cdiv(K, _GEMV_BLOCK_K)))
    k_chunk = triton.cdiv(K, split_k)

    partial = torch.empty(
        (M, split_k, N), device=input_flat.device, dtype=torch.float32
    )
    output = torch.empty((M, N), device=input_flat.device, dtype=input_flat.dtype)

    grid_partial = (triton.cdiv(N, _GEMV_BLOCK_N), split_k)
    grid_reduce = (triton.cdiv(N, _GEMV_BLOCK_N),)

    with torch_device_fn.device(input_flat.device):
        linear_gemv_kernel[grid_partial](
            input_flat,
            weight,
            partial,
            N,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            # partial is (M, split_k, N); the kernel wants (pm, pn, pk)
            partial.stride(0),
            partial.stride(2),
            partial.stride(1),
            k_chunk,
            BLOCK_SIZE_M=M,
            BLOCK_SIZE_N=_GEMV_BLOCK_N,
            BLOCK_SIZE_K=_GEMV_BLOCK_K,
            num_warps=_GEMV_NUM_WARPS,
            num_stages=_GEMV_NUM_STAGES,
        )
        linear_gemv_reduce_kernel[grid_reduce](
            partial,
            bias if bias is not None else weight,
            output,
            M,
            N,
            partial.stride(0),
            partial.stride(2),
            partial.stride(1),
            output.stride(0),
            output.stride(1),
            _bias_stride(bias),
            split_k,
            BIAS=bias is not None,
            BLOCK_SIZE_N=_GEMV_BLOCK_N,
        )
    return output


def _linear_dot(input_flat, weight, bias, M, N, K, device):
    if input_flat.dtype == torch.float32 and _may_transpose(weight, M):
        weight = _transpose_weight(weight)
        stride_wk, stride_wn = weight.stride(0), weight.stride(1)
    else:
        weight = weight.contiguous()
        stride_wk, stride_wn = weight.stride(1), weight.stride(0)

    output = torch.empty((M, N), device=device, dtype=input_flat.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    with torch_device_fn.device(device):
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
            stride_wn,
            stride_wk,
            output.stride(0),
            output.stride(1),
            _bias_stride(bias),
            BIAS=bias is not None,
        )
    return output


def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Args:
        input: Input tensor of shape (*, in_features) where * means any number of
               additional dimensions, including none.
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional

    Returns:
        Output tensor of shape (*, out_features)
    """
    logger.debug("GEMS LINEAR")

    if input.dim() == 1:
        input = input.unsqueeze(0)
        single_1d = True
    else:
        single_1d = False

    batch_dims = input.shape[:-1]
    K = input.shape[-1]
    if weight.shape[1] != K:
        raise RuntimeError(f"weight {tuple(weight.shape)} is not an (N, {K}) matrix")
    N = weight.shape[0]
    input_flat = input.view(-1, K)
    M = input_flat.shape[0]

    if input.dtype == torch.float32 and M <= _GEMV_MAX_M:
        output = _linear_gemv(input_flat, weight, bias, M, K, N)
    else:
        output = _linear_dot(input_flat, weight, bias, M, N, K, input.device)

    output = output.view(*batch_dims, N)

    if single_1d:
        output = output.squeeze(0)

    return output
