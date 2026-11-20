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

"""Stride-aware direct Triton fallback and gradient helpers for BAddBMM."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

_GENERIC_BLOCK_M = 32
_GENERIC_BLOCK_N = 32
_GENERIC_BLOCK_K = 32


@triton.jit(do_not_specialize=["alpha", "beta"])
def _generic_fallback_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_bias_b,
    stride_bias_m,
    stride_bias_n,
    stride_out_b,
    stride_out_m,
    stride_out_n,
    BETA_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    pid_b = ext.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr
        + pid_b * stride_ab
        + rows[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + pid_b * stride_bb
        + k_offsets[:, None] * stride_bk
        + cols[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        current_k = k_iter * BLOCK_K + k_offsets
        a = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & (current_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(current_k[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # FlagGems validates against an FP64 CPU reference and casts only the final
    # value to the requested output dtype.  Keep the complete epilogue fused in
    # FP32; in particular, do not emulate the double-rounding behavior of the
    # non-aliasing torch_musa functional path for matrix/batched bias.
    scaled_product = alpha * accumulator
    if BETA_ZERO:
        result = scaled_product
    else:
        out_mask = (rows[:, None] < M) & (cols[None, :] < N)
        bias_ptrs = (
            bias_ptr
            + pid_b * stride_bias_b
            + rows[:, None] * stride_bias_m
            + cols[None, :] * stride_bias_n
        )
        bias = tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        result = scaled_product + beta * bias
    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + rows[:, None] * stride_out_m
        + cols[None, :] * stride_out_n
    )
    tl.store(
        out_ptrs,
        result.to(out_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def _broadcast_strides(input, output_shape):
    """Return right-aligned broadcast strides without dispatching an ATen op."""
    pad = len(output_shape) - input.ndim
    input_shape = (1,) * pad + tuple(input.shape)
    input_strides = (0,) * pad + tuple(input.stride())
    return tuple(
        0 if size == 1 and size != output_size else stride
        for size, output_size, stride in zip(input_shape, output_shape, input_strides)
    )


def generic_out(input, batch1, batch2, out, *, beta, alpha):
    batch, m, k = batch1.shape
    n = batch2.shape[2]
    if out.numel() == 0:
        return out
    block_m = _GENERIC_BLOCK_M
    block_n = _GENERIC_BLOCK_N
    block_k = _GENERIC_BLOCK_K
    num_warps = 4
    # MThreads' direct tl.dot lowering strongly favors its 32x32 output tile,
    # but the reduction tile is shape-sensitive.  These routes are the winners
    # of a fresh-cache sweep over every active FlagGems forward/backward shape.
    if m <= 32 and n <= 256 and k >= 512:
        block_k = 64
    elif k <= 128 and m >= 128 and n >= 512:
        block_k = 16
    elif m >= 128 and n <= 128 and k >= 1024:
        block_m, block_n = 16, 16
    elif m <= 128 and n >= 512 and k >= 128:
        block_k = 16
    grid = (
        triton.cdiv(m, block_m),
        triton.cdiv(n, block_n),
        batch,
    )
    alpha_value, beta_value = float(alpha), float(beta)
    bias_strides = _broadcast_strides(input, out.shape)
    with torch_device_fn.device(batch1.device):
        _generic_fallback_kernel[grid](
            batch1,
            batch2,
            input,
            out,
            alpha_value,
            beta_value,
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            bias_strides[0],
            bias_strides[1],
            bias_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BETA_ZERO=beta_value == 0.0,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


@triton.jit(do_not_specialize=["beta"])
def _bias_grad_exact_kernel(
    grad_output_ptr,
    grad_input_ptr,
    beta,
    stride_gb,
    stride_gm,
    stride_gn,
    M: tl.constexpr,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scale a non-broadcasted bias gradient without a PyTorch fallback."""
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_indices = offsets // (M * N)
    within_batch = offsets % (M * N)
    row_indices = within_batch // N
    col_indices = within_batch % N
    mask = offsets < NUMEL
    values = tl.load(
        grad_output_ptr
        + batch_indices * stride_gb
        + row_indices * stride_gm
        + col_indices * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_input_ptr + offsets,
        (beta * values).to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["beta"])
def _bias_grad_broadcast_kernel(
    grad_output_ptr,
    grad_input_ptr,
    beta,
    stride_gb,
    stride_gm,
    stride_gn,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    INPUT_B: tl.constexpr,
    INPUT_M: tl.constexpr,
    INPUT_N: tl.constexpr,
    REDUCE_B: tl.constexpr,
    REDUCE_M: tl.constexpr,
    REDUCE_N: tl.constexpr,
    INPUT_NUMEL: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    """Reduce a right-aligned broadcast bias entirely in Triton."""
    output_offsets = ext.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    valid_output = output_offsets < INPUT_NUMEL
    input_b = output_offsets // (INPUT_M * INPUT_N)
    within_input_b = output_offsets % (INPUT_M * INPUT_N)
    input_m = within_input_b // INPUT_N
    input_n = within_input_b % INPUT_N
    accumulator = tl.zeros((BLOCK_REDUCE, BLOCK_OUT), dtype=tl.float32)
    for reduction_block in range(tl.cdiv(REDUCE_SIZE, BLOCK_REDUCE)):
        reduction_offsets = reduction_block * BLOCK_REDUCE + tl.arange(0, BLOCK_REDUCE)
        reduce_b = reduction_offsets // (REDUCE_M * REDUCE_N)
        within_reduce_b = reduction_offsets % (REDUCE_M * REDUCE_N)
        reduce_m = within_reduce_b // REDUCE_N
        reduce_n = within_reduce_b % REDUCE_N
        if INPUT_B == 1:
            batch_indices = reduce_b[:, None] + tl.zeros((1, BLOCK_OUT), dtype=tl.int32)
        else:
            batch_indices = input_b[None, :] + tl.zeros(
                (BLOCK_REDUCE, 1), dtype=tl.int32
            )
        if INPUT_M == 1:
            row_indices = reduce_m[:, None] + tl.zeros((1, BLOCK_OUT), dtype=tl.int32)
        else:
            row_indices = input_m[None, :] + tl.zeros((BLOCK_REDUCE, 1), dtype=tl.int32)
        if INPUT_N == 1:
            col_indices = reduce_n[:, None] + tl.zeros((1, BLOCK_OUT), dtype=tl.int32)
        else:
            col_indices = input_n[None, :] + tl.zeros((BLOCK_REDUCE, 1), dtype=tl.int32)
        load_mask = (
            (reduction_offsets[:, None] < REDUCE_SIZE)
            & valid_output[None, :]
            & (batch_indices < B)
            & (row_indices < M)
            & (col_indices < N)
        )
        values = tl.load(
            grad_output_ptr
            + batch_indices * stride_gb
            + row_indices * stride_gm
            + col_indices * stride_gn,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += values
    totals = tl.sum(accumulator, axis=0)
    tl.store(
        grad_input_ptr + output_offsets,
        (beta * totals).to(grad_input_ptr.dtype.element_ty),
        mask=valid_output,
    )


def bias_gradient(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    *,
    beta,
) -> torch.Tensor:
    """Return beta * sum_to_size(grad_output, input.shape) via Triton."""
    batch, m, n = grad_output.shape
    padded_shape = (1,) * (3 - input.ndim) + tuple(input.shape)
    if len(padded_shape) != 3:
        raise RuntimeError(
            "MThreads baddbmm autograd requires a bias with at most three dimensions"
        )
    input_b, input_m, input_n = padded_shape
    for input_dim, output_dim in zip(padded_shape, (batch, m, n)):
        if input_dim not in (1, output_dim):
            raise RuntimeError(
                f"cannot reduce baddbmm gradient {(batch, m, n)} "
                f"to bias shape {tuple(input.shape)}"
            )
    grad_input = torch.empty_like(input, memory_format=torch.contiguous_format)
    if grad_input.numel() == 0:
        return grad_input
    beta_value = float(beta)
    with torch_device_fn.device(grad_output.device):
        if padded_shape == (batch, m, n):
            block = 256
            _bias_grad_exact_kernel[(triton.cdiv(input.numel(), block),)](
                grad_output,
                grad_input,
                beta_value,
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                M=m,
                N=n,
                NUMEL=input.numel(),
                BLOCK=block,
                num_warps=4,
            )
        else:
            reduce_b = batch if input_b == 1 else 1
            reduce_m = m if input_m == 1 else 1
            reduce_n = n if input_n == 1 else 1
            reduce_size = reduce_b * reduce_m * reduce_n
            # Common vector and matrix broadcasts need at most B*M elements.
            # Keep each program bounded while accumulating additional chunks.
            block_out = 32
            block_reduce = min(32, triton.next_power_of_2(reduce_size))
            _bias_grad_broadcast_kernel[(triton.cdiv(input.numel(), block_out),)](
                grad_output,
                grad_input,
                beta_value,
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                B=batch,
                M=m,
                N=n,
                INPUT_B=input_b,
                INPUT_M=input_m,
                INPUT_N=input_n,
                REDUCE_B=reduce_b,
                REDUCE_M=reduce_m,
                REDUCE_N=reduce_n,
                INPUT_NUMEL=input.numel(),
                REDUCE_SIZE=reduce_size,
                BLOCK_OUT=block_out,
                BLOCK_REDUCE=block_reduce,
                num_warps=4,
            )
    return grad_input


def matmul_gradient(
    left: torch.Tensor,
    right: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    alpha,
) -> torch.Tensor:
    """Allocate and compute an alpha-scaled batched matmul gradient."""
    out = torch.empty(shape, dtype=left.dtype, device=left.device)
    # beta=0 makes the generic kernel skip the dummy bias read.  Reusing out as
    # that dummy does not create an aliasing hazard.
    return generic_out(
        out,
        left,
        right,
        out,
        beta=0.0,
        alpha=alpha,
    )


__all__ = ["bias_gradient", "generic_out", "matmul_gradient"]
