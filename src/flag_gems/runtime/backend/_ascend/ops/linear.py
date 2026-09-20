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
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# ============================================================
# M == 1 GEMV specialization
#
# Generic tiled GEMM is extremely inefficient for autoregressive
# decode where M == 1. Use a vector reduction kernel instead.
# ============================================================


@libentry()
@triton.jit
def linear_gemv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    K,
    stride_xk,
    stride_wn,
    stride_wk,
    BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros(
        (BLOCK_N,),
        dtype=tl.float32,
    )

    for kb in range(
        0,
        tl.cdiv(K, BLOCK_K),
    ):
        k = kb * BLOCK_K + offs_k

        x = tl.load(
            input_ptr + k * stride_xk,
            mask=k < K,
            other=0.0,
        )

        w = tl.load(
            weight_ptr + offs_n[:, None] * stride_wn + k[None, :] * stride_wk,
            mask=((offs_n[:, None] < N) & (k[None, :] < K)),
            other=0.0,
        )

        acc += tl.sum(
            w.to(tl.float32) * x[None, :].to(tl.float32),
            axis=1,
        )

    if BIAS:
        b = tl.load(
            bias_ptr + offs_n,
            mask=offs_n < N,
            other=0.0,
        )

        acc += b.to(tl.float32)

    tl.store(
        output_ptr + offs_n,
        acc.to(output_ptr.dtype.element_ty),
        mask=offs_n < N,
    )


# ============================================================
# M > 1 bias kernel
#
# Do NOT use:
#
#     acc += bias[None, :]
#
# in the GEMM kernel on current Triton-Ascend.
#
# That broadcast path produced incorrect FP16/FP32 results.
#
# Instead:
# - one program per output row
# - walk across N in vector chunks
# - output and bias remain 1-D vectors
# ============================================================


@libentry()
@triton.jit
def linear_add_bias_kernel(
    output_ptr,
    bias_ptr,
    M,
    N,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(
        0,
        BLOCK_N,
    )

    for nb in range(
        0,
        tl.cdiv(N, BLOCK_N),
    ):
        cols = nb * BLOCK_N + offs

        mask = (row < M) & (cols < N)

        out_ptrs = output_ptr + row * stride_om + cols * stride_on

        out = tl.load(
            out_ptrs,
            mask=mask,
            other=0.0,
        )

        bias = tl.load(
            bias_ptr + cols,
            mask=cols < N,
            other=0.0,
        )

        out += bias

        tl.store(
            out_ptrs,
            out,
            mask=mask,
        )


# ============================================================
# General M > 1 GEMM
#
# input  : [M, K]
# weight : [N, K]
# output : [M, N]
#
# Computes:
#
#     output = input @ weight.T
#
# Layout / scheduling follows Ascend mm.
# ============================================================


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
)
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["mm"])
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(
        M,
        BLOCK_M,
    )
    grid_n = tl.cdiv(
        N,
        BLOCK_N,
    )

    width = GROUP_M * grid_n

    group_id = pid // width

    group_size = min(
        grid_m - group_id * GROUP_M,
        GROUP_M,
    )

    pid_m = group_id * GROUP_M + pid % group_size

    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(
        0,
        BLOCK_M,
    )

    offs_n = pid_n * BLOCK_N + tl.arange(
        0,
        BLOCK_N,
    )

    offs_k = pid_z * BLOCK_K + tl.arange(
        0,
        BLOCK_K,
    )

    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik

    # weight is physically [N, K].
    #
    # Construct logical [K, N]
    # tiles directly through strides.
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros(
        (
            BLOCK_M,
            BLOCK_N,
        ),
        dtype=dot_out_dtype,
    )

    for k in range(
        0,
        tl.cdiv(
            K,
            BLOCK_K * SPLIT_K,
        ),
    ):
        if EVEN_K:
            a = tl.load(
                input_ptrs,
                mask=(offs_m < M)[:, None],
                other=0.0,
            )

            b = tl.load(
                weight_ptrs,
                mask=(offs_n < N)[None, :],
                other=0.0,
            )

        else:
            k_remaining = K - k * BLOCK_K * SPLIT_K

            a = tl.load(
                input_ptrs,
                mask=((offs_m < M)[:, None] & (offs_k[None, :] < k_remaining)),
                other=0.0,
            )

            b = tl.load(
                weight_ptrs,
                mask=((offs_k[:, None] < k_remaining) & (offs_n < N)[None, :]),
                other=0.0,
            )

        if a.dtype != b.dtype:
            a = a.to(output_ptr.dtype.element_ty)

            b = b.to(output_ptr.dtype.element_ty)

        acc += tl.dot(
            a,
            b,
            out_dtype=dot_out_dtype,
            allow_tf32=False,
        )

        input_ptrs += BLOCK_K * SPLIT_K * stride_ik

        weight_ptrs += BLOCK_K * SPLIT_K * stride_wk

    acc = acc.to(output_ptr.dtype.element_ty)

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    output_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    if SPLIT_K == 1:
        tl.store(
            output_ptrs,
            acc,
            mask=output_mask,
        )

    else:
        tl.atomic_add(
            output_ptrs,
            acc,
            mask=output_mask,
        )


def linear(
    input,
    weight,
    bias=None,
):
    logger.debug("GEMS_ASCEND LINEAR")

    original_shape = input.shape

    K = original_shape[-1]
    N = weight.shape[0]

    assert weight.shape[1] == K, "incompatible dimensions"

    M = input.numel() // K

    input_flat = input.reshape(
        M,
        K,
    )

    # Match Ascend mm handling of
    # unsupported non-contiguous layouts.
    if input_flat.stride(0) > 1 and input_flat.stride(1) > 1:
        input_flat = input_flat.contiguous()

    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()

    output = torch.empty(
        (M, N),
        device=input.device,
        dtype=input.dtype,
    )

    # ========================================================
    # M == 1
    #
    # Autoregressive decode / GEMV specialization.
    # ========================================================

    if M == 1:
        BLOCK_N = 64
        BLOCK_K = 256

        grid = (
            triton.cdiv(
                N,
                BLOCK_N,
            ),
        )

        with torch_device_fn.device(input.device):
            linear_gemv_kernel[grid](
                input_flat,
                weight,
                (bias if bias is not None else weight),
                output,
                N,
                K,
                input_flat.stride(1),
                weight.stride(0),
                weight.stride(1),
                BIAS=bias is not None,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=3,
            )

    # ========================================================
    # M > 1
    #
    # Ascend Cube-style GEMM.
    # ========================================================

    else:
        dot_out_dtype = tl.float32

        grid = lambda META: (
            triton.cdiv(
                M,
                META["BLOCK_M"],
            )
            * triton.cdiv(
                N,
                META["BLOCK_N"],
            ),
            META.get(
                "SPLIT_K",
                1,
            ),
        )

        with torch_device_fn.device(input.device):
            linear_kernel[grid](
                input_flat,
                weight,
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
                dot_out_dtype=dot_out_dtype,
                GROUP_M=8,
            )

        # Current Triton-Ascend broadcast lowering for
        # acc += bias[None, :] produced incorrect FP16/FP32
        # results. Add bias in a separate row-wise kernel.
        if bias is not None:
            BLOCK_N_BIAS = 1024

            bias_grid = (M,)

            with torch_device_fn.device(input.device):
                linear_add_bias_kernel[bias_grid](
                    output,
                    bias,
                    M,
                    N,
                    output.stride(0),
                    output.stride(1),
                    BLOCK_N=BLOCK_N_BIAS,
                    num_warps=2,
                    num_stages=1,
                )

    return output.reshape(
        *original_shape[:-1],
        N,
    )
