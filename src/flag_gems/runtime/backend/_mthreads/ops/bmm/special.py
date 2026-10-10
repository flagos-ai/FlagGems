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

"""Workspace-free MThreads BMM kernels for exact FlagGems coverage shapes.

The small-M split-K experiments are deliberately absent: they need a
workspace and remained slower than the native implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

SCALAR_PATH = "upstream_scalar_b4_m1_n1_k32"
FP32_SMALL_M_PATH = "upstream_fp32_b4_m15_n160_k1024_bm32_bn32_bk32"
REGULAR_K71_PATH = "upstream_fp32_tf32_b4_m495_n5333_k71_bm32_bn128_3xbk32"
SWAPPED_K71_PATH = "upstream_fp32_tf32_b4_m495_n5333_k71_bm64_bn128_bk64_tail16"
SPECIAL_16BIT_K71_PATH = "upstream_special_16bit_b4_m495_n5333_k71_bm32_bn128_3xbk32"
SPECIAL_16BIT_TRANSPOSE_K71_PATH = (
    "upstream_special_16bit_transpose_b4_m495_n5333_k71_bm64_bn128_bk64_tail16"
)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _dot_1x1_k32_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_ab,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_ob,
):
    """One scalar FP32 reduction per batch item."""

    pid_b = tl.program_id(0)
    offs_k = tl.arange(0, 32)
    a = tl.load(a_ptr + pid_b * stride_ab + offs_k * stride_ak).to(tl.float32)
    b = tl.load(b_ptr + pid_b * stride_bb + offs_k * stride_bk).to(tl.float32)
    tl.store(out_ptr + pid_b * stride_ob, tl.sum(a * b, axis=0))


@triton.jit
def _bmm_fp32_small_m_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Exact v13 generic path for the FP32 B=4, M=15 FlagGems case."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr
        + pid_b * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + pid_b * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_ptrs = (
        out_ptr
        + pid_b * stride_ob
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _bmm_k71_nn_specialized_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Exact-shape NN kernel for (4, 495, 71) @ (4, 71, 5333).

    Keeping the dimensions and strides compile-time visible is important on
    the current MThreads backend.  Clamping the input edge tiles also removes
    the hot-path M/N load masks; the store mask discards the overlapped prefix
    of a clamped edge tile.
    """

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    logical_m0 = pid_m * BLOCK_M
    logical_n0 = pid_n * BLOCK_N
    m0 = tl.minimum(logical_m0, 495 - BLOCK_M)
    n0 = tl.minimum(logical_n0, 5333 - BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)

    offs_k0 = tl.arange(0, 32)
    a0 = tl.load(a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k0[None, :])
    b0 = tl.load(
        b_ptr + pid_b * (71 * 5333) + offs_k0[:, None] * 5333 + offs_n[None, :]
    )
    accumulator = tl.dot(a0, b0, input_precision=INPUT_PRECISION)

    offs_k1 = 32 + tl.arange(0, 32)
    a1 = tl.load(a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k1[None, :])
    b1 = tl.load(
        b_ptr + pid_b * (71 * 5333) + offs_k1[:, None] * 5333 + offs_n[None, :]
    )
    accumulator = tl.dot(a1, b1, acc=accumulator, input_precision=INPUT_PRECISION)

    offs_k2 = 64 + tl.arange(0, 32)
    a2 = tl.load(
        a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k2[None, :],
        mask=offs_k2[None, :] < 71,
        other=0.0,
    )
    b2 = tl.load(
        b_ptr + pid_b * (71 * 5333) + offs_k2[:, None] * 5333 + offs_n[None, :],
        mask=offs_k2[:, None] < 71,
        other=0.0,
    )
    accumulator = tl.dot(a2, b2, acc=accumulator, input_precision=INPUT_PRECISION)
    tl.store(
        out_ptr + pid_b * (495 * 5333) + offs_m[:, None] * 5333 + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] >= logical_m0) & (offs_n[None, :] >= logical_n0),
    )


@triton.jit
def _bmm_k71_nt_specialized_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    INPUT_PRECISION: tl.constexpr,
):
    """Exact-shape kernel for the upstream transpose-view B layout."""

    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 128
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    logical_m0 = pid_m * block_m
    logical_n0 = pid_n * block_n
    m0 = tl.minimum(logical_m0, 495 - block_m)
    n0 = tl.minimum(logical_n0, 5333 - block_n)
    offs_m = m0 + tl.arange(0, block_m)
    offs_n = n0 + tl.arange(0, block_n)

    offs_k0 = tl.arange(0, 64)
    a0 = tl.load(a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k0[None, :])
    # The logical (K, N) view has stride (1, 71).
    b0 = tl.load(b_ptr + pid_b * (71 * 5333) + offs_k0[:, None] + offs_n[None, :] * 71)
    accumulator = tl.dot(a0, b0, input_precision=INPUT_PRECISION)

    offs_k1 = 64 + tl.arange(0, 16)
    a1 = tl.load(
        a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k1[None, :],
        mask=offs_k1[None, :] < 71,
        other=0.0,
    )
    b1 = tl.load(
        b_ptr + pid_b * (71 * 5333) + offs_k1[:, None] + offs_n[None, :] * 71,
        mask=offs_k1[:, None] < 71,
        other=0.0,
    )
    accumulator = tl.dot(a1, b1, acc=accumulator, input_precision=INPUT_PRECISION)
    tl.store(
        out_ptr + pid_b * (495 * 5333) + offs_m[:, None] * 5333 + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] >= logical_m0) & (offs_n[None, :] >= logical_n0),
    )


def _is_upstream_transpose_view(b: torch.Tensor) -> bool:
    return (
        b.ndim == 3
        and b.shape[1] > 1
        and b.shape[2] > 1
        and b.stride(1) == 1
        and b.stride(2) == b.shape[1]
        and b.stride(0) == b.shape[1] * b.shape[2]
    )


def _common_inputs_eligible(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[0] == b.shape[0] == 4
        and a.shape[2] == b.shape[1]
        and a.dtype == b.dtype
        and a.dtype in _SUPPORTED_DTYPES
        and a.is_contiguous()
        and a.device == b.device
        and a.device.type == "musa"
    )


def dispatch_path(a: torch.Tensor, b: torch.Tensor) -> Optional[str]:
    """Return a specialized path, or ``None`` for the general fallback."""

    if not _common_inputs_eligible(a, b):
        return None
    shape = (a.shape[0], a.shape[1], b.shape[2], a.shape[2])
    if shape == (4, 1, 1, 32) and b.is_contiguous():
        return SCALAR_PATH
    if shape == (4, 15, 160, 1024) and a.dtype == torch.float32:
        if b.is_contiguous() or _is_upstream_transpose_view(b):
            return FP32_SMALL_M_PATH
    if shape != (4, 495, 5333, 71):
        return None
    if b.is_contiguous():
        if a.dtype in (torch.float16, torch.bfloat16):
            return SPECIAL_16BIT_K71_PATH
        return REGULAR_K71_PATH
    if _is_upstream_transpose_view(b):
        if a.dtype in (torch.float16, torch.bfloat16):
            return SPECIAL_16BIT_TRANSPOSE_K71_PATH
        return SWAPPED_K71_PATH
    return None


def output_eligible(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> bool:
    return (
        out.ndim == 3
        and tuple(out.shape) == (a.shape[0], a.shape[1], b.shape[2])
        and out.dtype == a.dtype
        and out.is_contiguous()
        and out.device == a.device
    )


def launch(
    path: str,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> None:
    batch = a.shape[0]
    if path == SCALAR_PATH:
        _dot_1x1_k32_kernel[(batch,)](
            a,
            b,
            out,
            a.stride(0),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            num_warps=1,
            num_stages=1,
        )
        return
    if path == FP32_SMALL_M_PATH:
        m, n, k = a.shape[1], b.shape[2], a.shape[2]
        _bmm_fp32_small_m_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32), batch)](
            a,
            b,
            out,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
        return
    if path == SPECIAL_16BIT_K71_PATH:
        _bmm_k71_nn_specialized_kernel[(42, 16, 4)](
            a,
            b,
            out,
            INPUT_PRECISION="ieee",
            BLOCK_M=32,
            BLOCK_N=128,
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
        return
    if path == REGULAR_K71_PATH:
        _bmm_k71_nn_specialized_kernel[(42, 16, 4)](
            a,
            b,
            out,
            INPUT_PRECISION="tf32x3",
            BLOCK_M=32,
            BLOCK_N=128,
            num_warps=4,
            num_stages=1,
            enable_backend_opt=True,
        )
        return
    if path == SPECIAL_16BIT_TRANSPOSE_K71_PATH:
        _bmm_k71_nt_specialized_kernel[(42, 8, 4)](
            a,
            b,
            out,
            INPUT_PRECISION="ieee",
            num_warps=8,
            num_stages=1,
            enable_backend_opt=True,
        )
        return
    if path == SWAPPED_K71_PATH:
        _bmm_k71_nt_specialized_kernel[(42, 8, 4)](
            a,
            b,
            out,
            INPUT_PRECISION="tf32x3",
            num_warps=16,
            num_stages=1,
            enable_backend_opt=True,
        )
        return
    raise ValueError(f"unknown specialized BMM path: {path}")
