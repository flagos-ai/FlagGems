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

"""Scaled INT8 matrix multiplication on THead/PPU.

A[M,K] and B[K,N] are already quantized. Caller-provided FP32 scales
are per tensor, per A row, or per B column. Bias is added in FP32 before
output conversion. AIU requires FlagTree #1026 with correct INT8 .b8 lowering.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle_async

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_FLOAT = {torch.bfloat16, torch.float16, torch.float32}


@triton.jit
def _grouped_pids(
    pid,
    m: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    group_m: tl.constexpr,
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    if grid_m <= group_m:
        return pid % grid_m, pid // grid_m
    width = group_m * grid_n
    group_id = pid // width
    if grid_m % group_m == 0:
        return group_id * group_m + pid % group_m, (pid % width) // group_m
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@libentry()
@triton.jit(do_not_specialize_on_alignment=["A_Q", "B_Q", "OUT"])
def _mm_w8a8_aiu_kernel(
    A_Q,
    B_Q,
    A_SCALE,
    B_SCALE,
    OUT,
    BIAS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    OUT_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BOUNDARY: tl.constexpr,
    STORE_MASK: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TRANSPOSE_OUT: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _grouped_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    a_block_ptr = tl.make_block_ptr(
        A_Q,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # B is physically contiguous [N, K], logically column-major [K, N].
    # N may be padded to BLOCK_N so skinny GEMM can skip load boundary checks.
    b_block_ptr = tl.make_block_ptr(
        B_Q,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    if BOUNDARY:
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    else:
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tle_async.load(a_block_ptr, is_async=True)
            b = tle_async.load(b_block_ptr, is_async=True)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if STORE_MASK:
        a_scale = tl.load(
            A_SCALE + tl.where(A_SCALAR, 0, offs_m), mask=offs_m < M, other=0.0
        ).to(tl.float32)
        b_scale = tl.load(
            B_SCALE + tl.where(B_SCALAR, 0, offs_n), mask=offs_n < N, other=0.0
        ).to(tl.float32)
    else:
        a_scale = tl.load(A_SCALE + tl.where(A_SCALAR, 0, offs_m)).to(tl.float32)
        b_scale = tl.load(B_SCALE + tl.where(B_SCALAR, 0, offs_n)).to(tl.float32)
    if TRANSPOSE_OUT:
        # The swapped GEMM is W @ A.T. Keep activation scaling first and
        # write its transpose directly into the caller's contiguous output.
        out = acc.to(tl.float32) * b_scale[None, :] * a_scale[:, None]
        if HAS_BIAS:
            out += tl.load(BIAS + offs_m, offs_m < M, other=0).to(tl.float32)[:, None]
        offsets = offs_m[:, None] + offs_n[None, :] * M
    else:
        out = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
        if HAS_BIAS:
            out += tl.load(BIAS + offs_n, offs_n < OUT_N, other=0).to(tl.float32)[
                None, :
            ]
        offsets = offs_m[:, None] * OUT_N + offs_n[None, :]
    if STORE_MASK:
        tl.store(
            OUT + offsets,
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_N),
        )
    else:
        tl.store(OUT + offsets, out)


def _pick_tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, GROUP_M.

    INT8 AIU v1 uses channels of 32/64/128 bytes. Wide decode projections
    load two 128-byte K segments per tile to amortize async load overhead.
    Never pad BLOCK_N far past N: a 64-wide MMA on N=1 is ~64x wasted work.
    """
    if 1 < m <= 8 and n >= 32768 and 2048 <= k <= 4096:
        return 16, 128, 256, 4, 3, 8
    if 32 < m <= 128 and n <= 512 and k >= 2048:
        return 16, 16, 128, 1, 2, 8
    if 128 < m <= 256 and n <= 512 and k >= 2048:
        return 32, 32, 128, 4, 2, 8
    if m >= 1024 and n <= 1024 and k >= 2048:
        return 64, 128, 128, 4, 3, 8
    if 32 < m <= 256 and n >= 1024 and 256 <= k <= 512:
        return 32, 128, 128, 4, 3, 8
    if 16 < m <= 64 and 512 <= n < 2048 and k >= 1024:
        return 32, 64, 128, 4, 3, 8
    if 32 < m <= 512 and n >= 2048 and k >= 2048:
        return 64, 128, 128, 4 if n >= 16384 else 8, 3, 8
    if k >= 256:
        block_k = 128
    elif k >= 64:
        block_k = 64
    else:
        block_k = 32
    if k >= 2048:
        stages = 3 if m >= 1024 else 4
    elif k >= 512:
        stages = 3
    else:
        stages = 2
    group_m = 8
    # Keep BLOCK_N close to N so skinny GEMM does not compute unused columns.
    if n <= 16:
        block_n = 16
    elif n <= 32:
        block_n = 32
    elif n < 512:
        block_n = 64
    else:
        block_n = 128
    if m <= 16:
        # Tiny wide GEMM is launch-bound; one warp + BLOCK_K=128 beats gems BF16.
        if n <= 256 and k <= 256 and k >= 128:
            return 16, min(block_n, 64), 128, 1, 2, group_m
        warps = 1 if block_n <= 32 else 4
        return 16, block_n, block_k, warps, stages, group_m
    if m <= 32:
        warps = 2 if block_n <= 16 else 4
        return 32, block_n, block_k, warps, stages, group_m
    return 64, block_n, block_k, 4, stages, group_m


@libentry()
@triton.jit
def _mm_w8a8_vector_kernel(
    A,
    B,
    SCALE_A,
    SCALE_B,
    OUT,
    BIAS,
    LENGTH: tl.constexpr,
    K: tl.constexpr,
    M_ONE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # A GEMV has no second matrix dimension to amortize padded MMA work.
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    if M_ONE:
        a = tl.load(A + k[None, :], k[None, :] < K, other=0).to(tl.int32)
        b = tl.load(
            B + r[:, None] * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        sa = tl.load(SCALE_A)
        sb = tl.load(SCALE_B + tl.where(B_SCALAR, 0, r), r < LENGTH, other=0)
    else:
        a = tl.load(
            A + r[:, None] * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        b = tl.load(B + k[None, :], k[None, :] < K, other=0).to(tl.int32)
        sa = tl.load(SCALE_A + tl.where(A_SCALAR, 0, r), r < LENGTH, other=0)
        sb = tl.load(SCALE_B)
    value = tl.sum(a * b, 1).to(tl.float32) * sa * sb
    if HAS_BIAS:
        value += tl.load(BIAS + tl.where(M_ONE, r, 0), r < LENGTH, other=0).to(
            tl.float32
        )
    tl.store(OUT + r, value, r < LENGTH)


@libentry()
@triton.jit
def _zero_output_kernel(
    OUT,
    BIAS,
    SIZE: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    if HAS_BIAS:
        value = tl.load(BIAS + offsets % N, offsets < SIZE, other=0).to(tl.float32)
    tl.store(OUT + offsets, value, offsets < SIZE)


def _validate_mm_inputs(a, b, scale_a, scale_b):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b)):
        raise TypeError("mm_w8a8_int8 expects Tensor inputs and scales")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized torch.int8 tensors")
    if a.device.type != "cuda" or any(
        x.device != a.device for x in (b, scale_a, scale_b)
    ):
        raise ValueError("inputs and scales must be on the same PPU device")
    m, k = a.shape
    n = b.shape[1]
    if (scale_a.numel() != 1 and scale_a.shape not in ((m,), (m, 1))) or (
        scale_b.numel() != 1 and scale_b.shape not in ((n,), (1, n))
    ):
        raise ValueError("expected scalar scales or scale_a[M]/[M,1], scale_b[N]/[1,N]")
    if any(
        x.dtype != torch.float32 or not x.is_contiguous() for x in (scale_a, scale_b)
    ):
        raise ValueError("scales must be contiguous FP32 tensors")
    return m, n, k


def _launch(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    bias=None,
) -> torch.Tensor:
    transpose_out = (m == 256 and n >= 16384 and 2048 <= k <= 4096) or (
        m >= 1024 and n <= 1024 and 2048 <= k <= 4096
    )
    if transpose_out:
        # Both operands are already in the required physical layout.
        # Swapping them lets one tile reuse more activation rows.
        a_q, b_q = b_q, a_q
        a_scale, b_scale = b_scale, a_scale
        m, n = n, m
        block_m, block_n, block_k, num_warps, num_stages, group_m = (
            64,
            256,
            128,
            8,
            3,
            8,
        )
    else:
        (
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            group_m,
        ) = _pick_tiles(m, n, k)
    n_b = n
    boundary = (m % block_m) != 0 or (n_b % block_n) != 0 or (k % block_k) != 0
    store_mask = boundary or (n != n_b)
    logger.debug(
        "GEMS_THEAD MM_W8A8_AIU m=%s n=%s k=%s n_b=%s "
        "tiles=(%s,%s,%s) warps=%s stages=%s boundary=%s store_mask=%s",
        m,
        n,
        k,
        n_b,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        boundary,
        store_mask,
    )
    with torch_device_fn.device(a_q.device):
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n_b, block_n),)
        _mm_w8a8_aiu_kernel[grid](
            a_q,
            b_q,
            a_scale,
            b_scale,
            out,
            bias,
            m,
            n_b,
            k,
            n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            BOUNDARY=boundary,
            STORE_MASK=store_mask,
            NUM_WARPS=num_warps,
            TRANSPOSE_OUT=transpose_out,
            A_SCALAR=a_scale.numel() == 1,
            B_SCALAR=b_scale.numel() == 1,
            HAS_BIAS=bias is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _run_mm(a, b, scale_a, scale_b, out, m, n, k, bias=None):
    if m == 0 or n == 0:
        return out
    if k == 0:
        with torch_device_fn.device(a.device):
            _zero_output_kernel[(triton.cdiv(m * n, 1024),)](
                out, bias, m * n, n, bias is not None, BLOCK=1024
            )
        return out
    if (
        1 < m <= 8
        and n <= 512
        and k <= 4096
        and k % 4 == 0
        and a.is_contiguous()
        and b.stride() == (1, k)
        and a.data_ptr() % 4 == 0
        and b.data_ptr() % 4 == 0
    ):
        r, warps = (1, 1) if m > 4 else (2, 4)
        with torch_device_fn.device(a.device):
            _small_rows[(triton.cdiv(n, r), m)](
                a,
                b,
                scale_a,
                scale_b,
                out,
                bias,
                n,
                k,
                R=r,
                BK=triton.next_power_of_2(k),
                A_SCALAR=scale_a.numel() == 1,
                B_SCALAR=scale_b.numel() == 1,
                HAS_BIAS=bias is not None,
                num_warps=warps,
            )
        return out
    if (
        min(m, n) == 1
        and k <= 4096
        and n <= 16384
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
    ):
        length = max(m, n)
        with torch_device_fn.device(a.device):
            _mm_w8a8_vector_kernel[(triton.cdiv(length, 2),)](
                a,
                b,
                scale_a,
                scale_b,
                out,
                bias,
                length,
                k,
                m == 1,
                BLOCK_R=2,
                BLOCK_K=triton.next_power_of_2(k),
                A_SCALAR=scale_a.numel() == 1,
                B_SCALAR=scale_b.numel() == 1,
                HAS_BIAS=bias is not None,
                num_warps=1 if k <= 512 else 4,
            )
        return out
    if (
        m == 1
        and n >= 32768
        and 1024 <= k <= 4096
        and k % 4 == 0
        and a.is_contiguous()
        and b.stride() == (1, k)
        and a.data_ptr() % 4 == 0
        and b.data_ptr() % 4 == 0
    ):
        with torch_device_fn.device(a.device):
            _small_rows[(triton.cdiv(n, 4), 1)](
                a,
                b,
                scale_a,
                scale_b,
                out,
                bias,
                n,
                k,
                R=4,
                BK=triton.next_power_of_2(k),
                A_SCALAR=scale_a.numel() == 1,
                B_SCALAR=scale_b.numel() == 1,
                HAS_BIAS=bias is not None,
                num_warps=4,
            )
        return out
    # Normalize inputs for the required async AIU loader.
    a = a.contiguous()
    b = b.t().contiguous().t()
    return _launch(a, scale_a, b, scale_b, out, m, n, k, bias)


@triton.jit
def _small_rows(
    A,
    B,
    SA,
    SB,
    OUT,
    BIAS,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    m = tl.program_id(1)
    r = tl.program_id(0) * R + tl.arange(0, R)
    k = tl.arange(0, BK // 4)
    a = tl.load(A.to(tl.pointer_type(tl.int32)) + m * (K // 4) + k, k < K // 4, other=0)
    b = tl.load(
        B.to(tl.pointer_type(tl.int32)) + r[:, None] * (K // 4) + k[None, :],
        (r[:, None] < N) & (k[None, :] < K // 4),
        other=0,
    )
    partial = tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, 0;",
        constraints="=r,r,r",
        args=[a[None, :], b],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    acc = tl.sum(partial, 1)
    sa = tl.load(SA + tl.where(A_SCALAR, 0, m))
    sb = tl.load(SB + tl.where(B_SCALAR, 0, r), r < N, other=0)
    value = acc.to(tl.float32) * sa * sb
    if HAS_BIAS:
        value += tl.load(BIAS + r, r < N, other=0).to(tl.float32)
    tl.store(OUT + m * N + r, value, r < N)


def _validate_output_dtype_and_bias(a, n, out_dtype, bias):
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a Tensor")
        if bias.shape != (n,) or bias.device != a.device or not bias.is_contiguous():
            raise ValueError("bias must be contiguous [N] on the input device")
        if bias.dtype != out_dtype:
            raise TypeError("bias dtype must match the output dtype")


def mm_w8a8_int8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype=torch.bfloat16,
    bias=None,
) -> torch.Tensor:
    """Return (A_int8 @ B_int8) * scale_a * scale_b + bias.

    Inputs have shapes [M,K] and [K,N]. Contiguous FP32 scales may contain
    one value, M activation values ([M] or [M,1]), or N weight values
    ([N] or [1,N]). Optional bias is [N] with the output dtype.
    Quantization is the caller's responsibility. No input values are cached.
    """
    logger.debug("GEMS MM_W8A8_INT8")
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    _validate_output_dtype_and_bias(a, n, out_dtype, bias)
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k, bias)


def mm_w8a8_int8_out(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out: torch.Tensor,
    bias=None,
) -> torch.Tensor:
    """Write scaled INT8 GEMM into a contiguous caller-owned [M,N] output."""
    logger.debug("GEMS MM_W8A8_INT8_OUT")
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    _validate_output_dtype_and_bias(a, n, out.dtype, bias)
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k, bias)
