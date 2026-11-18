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

"""Exact-shape backward kernels for the shared FlagGems BAddBMM cases."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit(do_not_specialize=["alpha", "beta"])
def _small_grad_a_bias_kernel(
    grad_ptr,
    b_ptr,
    grad_bias_ptr,
    grad_a_ptr,
    alpha,
    beta,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grad_a[2, 15, 1024] = grad[2, 15, 160] @ b[2, 1024, 160]^T.
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for k0 in range(0, 160, BLOCK_K):
        ks = k0 + inner
        g = tl.load(
            grad_ptr + pid_b * (15 * 160) + rows[:, None] * 160 + ks[None, :],
            mask=(rows[:, None] < 15) & (ks[None, :] < 160),
            other=0.0,
        )
        bt = tl.load(
            b_ptr + pid_b * (1024 * 160) + cols[None, :] * 160 + ks[:, None],
            mask=(cols[None, :] < 1024) & (ks[:, None] < 160),
            other=0.0,
        )
        acc += tl.dot(g, bt, input_precision=INPUT_PRECISION)
        if pid_n == 0:
            tl.store(
                grad_bias_ptr + pid_b * (15 * 160) + rows[:, None] * 160 + ks[None, :],
                (beta * g).to(grad_bias_ptr.dtype.element_ty),
                mask=(rows[:, None] < 15) & (ks[None, :] < 160),
            )
    tl.store(
        grad_a_ptr + pid_b * (15 * 1024) + rows[:, None] * 1024 + cols[None, :],
        (alpha * acc).to(grad_a_ptr.dtype.element_ty),
        mask=(rows[:, None] < 15) & (cols[None, :] < 1024),
    )


@triton.jit(do_not_specialize=["alpha"])
def _small_grad_b_kernel(
    a_ptr,
    grad_ptr,
    grad_b_ptr,
    alpha,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # grad_b[2, 1024, 160] = a[2, 15, 1024]^T @ grad[2, 15, 160].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 32)
    at = tl.load(
        a_ptr + pid_b * (15 * 1024) + inner[:, None] * 1024 + rows[None, :],
        mask=(inner[:, None] < 15) & (rows[None, :] < 1024),
        other=0.0,
    ).T
    g = tl.load(
        grad_ptr + pid_b * (15 * 160) + inner[:, None] * 160 + cols[None, :],
        mask=(inner[:, None] < 15) & (cols[None, :] < 160),
        other=0.0,
    )
    acc = tl.dot(at, g, input_precision=INPUT_PRECISION)
    tl.store(
        grad_b_ptr + pid_b * (1024 * 160) + rows[:, None] * 160 + cols[None, :],
        (alpha * acc).to(grad_b_ptr.dtype.element_ty),
        mask=(rows[:, None] < 1024) & (cols[None, :] < 160),
    )


@triton.jit(do_not_specialize=["alpha"])
def _k71_grad_a_main_tail_kernel(
    grad_ptr,
    b_ptr,
    grad_a_ptr,
    alpha,
):
    """gradA with an unmasked 5312-wide main loop and one masked K tail."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    rows = pid_m * 32 + tl.arange(0, 32)
    cols = pid_n * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 64)
    row_mask = rows < 495
    col_mask = cols < 71
    acc = tl.zeros((32, 32), tl.float32)
    # 5333 = 83 * 64 + 21.  Avoid a K predicate in all 83 steady tiles.
    for k0 in range(0, 5312, 64):
        ks = k0 + inner
        g = tl.load(
            grad_ptr + pid_b * (495 * 5333) + rows[:, None] * 5333 + ks[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        bt = tl.load(
            b_ptr + pid_b * (71 * 5333) + cols[None, :] * 5333 + ks[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(g, bt, input_precision="ieee")
    tail = 5312 + inner
    g = tl.load(
        grad_ptr + pid_b * (495 * 5333) + rows[:, None] * 5333 + tail[None, :],
        mask=row_mask[:, None] & (tail[None, :] < 5333),
        other=0.0,
    )
    bt = tl.load(
        b_ptr + pid_b * (71 * 5333) + cols[None, :] * 5333 + tail[:, None],
        mask=col_mask[None, :] & (tail[:, None] < 5333),
        other=0.0,
    )
    acc += tl.dot(g, bt, input_precision="ieee")
    tl.store(
        grad_a_ptr + pid_b * (495 * 71) + rows[:, None] * 71 + cols[None, :],
        (alpha * acc).to(grad_a_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit(do_not_specialize=["beta"])
def _k71_bias_scale_kernel(grad_ptr, grad_bias_ptr, beta):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < 2 * 495 * 5333
    values = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        grad_bias_ptr + offsets,
        (beta * values).to(grad_bias_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["alpha"])
def _k71_grad_b_main_tail_kernel(
    a_ptr,
    grad_ptr,
    grad_b_ptr,
    alpha,
):
    """gradB with an unmasked 480-row main loop and one masked K tail."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    rows = pid_m * 64 + tl.arange(0, 64)
    cols = pid_n * 64 + tl.arange(0, 64)
    inner = tl.arange(0, 32)
    row_mask = rows < 71
    col_mask = cols < 5333
    acc = tl.zeros((64, 64), tl.float32)
    for k0 in range(0, 480, 32):
        ks = k0 + inner
        at = tl.load(
            a_ptr + pid_b * (495 * 71) + ks[:, None] * 71 + rows[None, :],
            mask=row_mask[None, :],
            other=0.0,
        ).T
        g = tl.load(
            grad_ptr + pid_b * (495 * 5333) + ks[:, None] * 5333 + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(at, g, input_precision="ieee")
    tail = 480 + inner
    at = tl.load(
        a_ptr + pid_b * (495 * 71) + tail[:, None] * 71 + rows[None, :],
        mask=(tail[:, None] < 495) & row_mask[None, :],
        other=0.0,
    ).T
    g = tl.load(
        grad_ptr + pid_b * (495 * 5333) + tail[:, None] * 5333 + cols[None, :],
        mask=(tail[:, None] < 495) & col_mask[None, :],
        other=0.0,
    )
    acc += tl.dot(at, g, input_precision="ieee")
    tl.store(
        grad_b_ptr + pid_b * (71 * 5333) + rows[:, None] * 5333 + cols[None, :],
        (alpha * acc).to(grad_b_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _precision(dtype: torch.dtype, fp32_precision: str) -> str:
    return fp32_precision if dtype == torch.float32 else "ieee"


def backward_dispatch_path(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_output: torch.Tensor,
) -> str | None:
    """Select only exact official backward shapes validated by this module."""
    tensors = (bias, a, b, grad_output)
    if (
        any(t.device.type != "musa" for t in tensors)
        or any(t.device != a.device for t in tensors)
        or any(t.dtype != a.dtype for t in tensors)
        or a.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(not t.is_contiguous() for t in tensors)
    ):
        return None
    shapes = tuple(tuple(t.shape) for t in tensors)
    if shapes == (
        (2, 15, 160),
        (2, 15, 1024),
        (2, 1024, 160),
        (2, 15, 160),
    ):
        return "official_backward_b2_m15_n160_k1024"
    if a.dtype in (torch.float16, torch.bfloat16) and shapes == (
        (2, 495, 5333),
        (2, 495, 71),
        (2, 71, 5333),
        (2, 495, 5333),
    ):
        return "official_backward_b2_m495_n5333_k71_half"
    return None


def launch_backward(
    grad_output: torch.Tensor,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    alpha: float,
    beta: float,
):
    path = backward_dispatch_path(bias, a, b, grad_output)
    if path == "official_backward_b2_m15_n160_k1024":
        return launch_small_backward(grad_output, bias, a, b, alpha=alpha, beta=beta)
    if path == "official_backward_b2_m495_n5333_k71_half":
        return launch_k71_backward(grad_output, bias, a, b, alpha=alpha, beta=beta)
    raise ValueError("inputs are not eligible for a specialized backward path")


def launch_small_backward(
    grad_output: torch.Tensor,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    block_na: int = 64,
    block_ka: int = 32,
    block_mb: int = 32,
    block_nb: int = 64,
    fp32_precision: str = "tf32x3",
):
    grad_bias = torch.empty_like(bias)
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    precision = _precision(a.dtype, fp32_precision)
    with torch_device_fn.device(a.device):
        _small_grad_a_bias_kernel[(triton.cdiv(1024, block_na), 2)](
            grad_output,
            b,
            grad_bias,
            grad_a,
            float(alpha),
            float(beta),
            INPUT_PRECISION=precision,
            BLOCK_N=block_na,
            BLOCK_K=block_ka,
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
        _small_grad_b_kernel[
            (triton.cdiv(1024, block_mb), triton.cdiv(160, block_nb), 2)
        ](
            a,
            grad_output,
            grad_b,
            float(alpha),
            INPUT_PRECISION=precision,
            BLOCK_M=block_mb,
            BLOCK_N=block_nb,
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
    return grad_bias, grad_a, grad_b


def launch_k71_backward(
    grad_output: torch.Tensor,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    alpha: float,
    beta: float,
):
    """Specialized official K71 half backward using three compact kernels."""
    grad_bias = torch.empty_like(bias)
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    with torch_device_fn.device(a.device):
        _k71_grad_a_main_tail_kernel[(16, 3, 2)](
            grad_output,
            b,
            grad_a,
            float(alpha),
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
        _k71_bias_scale_kernel[(triton.cdiv(2 * 495 * 5333, 256),)](
            grad_output,
            grad_bias,
            float(beta),
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
        _k71_grad_b_main_tail_kernel[(2, triton.cdiv(5333, 64), 2)](
            a,
            grad_output,
            grad_b,
            float(alpha),
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
    return grad_bias, grad_a, grad_b
