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

"""W8A8 INT8 matrix multiplication on THead/PPU.

A[M,K] contains floating activations; B[K,N] contains prequantized INT8
weights with caller-provided per-column FP32 dequantization scales.
Only activations are quantized per row on each call and CUDA Graph replay.
AIU requires FlagTree #1026 with correct INT8 .b8 lowering.
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
        a_scale = tl.load(A_SCALE + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        b_scale = tl.load(B_SCALE + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    else:
        a_scale = tl.load(A_SCALE + offs_m).to(tl.float32)
        b_scale = tl.load(B_SCALE + offs_n).to(tl.float32)
    if TRANSPOSE_OUT:
        # The swapped GEMM is W @ A.T. Keep activation scaling first and
        # write its transpose directly into the caller's contiguous output.
        out = (acc.to(tl.float32) * b_scale[None, :] * a_scale[:, None]).to(
            OUT.dtype.element_ty
        )
        offsets = offs_m[:, None] + offs_n[None, :] * M
    else:
        out = (acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(
            OUT.dtype.element_ty
        )
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
    LENGTH: tl.constexpr,
    K: tl.constexpr,
    M_ONE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        sb = tl.load(SCALE_B + r, r < LENGTH, other=0)
    else:
        a = tl.load(
            A + r[:, None] * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        b = tl.load(B + k[None, :], k[None, :] < K, other=0).to(tl.int32)
        sa = tl.load(SCALE_A + r, r < LENGTH, other=0)
        sb = tl.load(SCALE_B)
    value = tl.sum(a * b, 1).to(tl.float32) * sa * sb
    tl.store(OUT + r, value, r < LENGTH)


@libentry()
@triton.jit
def _zero_output_kernel(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + offsets, 0, offsets < SIZE)


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
    if scale_a.shape not in ((m,), (m, 1)) or scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected scale_a[M] or [M,1], scale_b[N] or [1,N]")
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
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _run_mm(a, b, scale_a, scale_b, out, m, n, k):
    if m == 0 or n == 0:
        return out
    if k == 0:
        with torch_device_fn.device(a.device):
            _zero_output_kernel[(triton.cdiv(m * n, 1024),)](out, m * n, BLOCK=1024)
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
                n,
                k,
                R=r,
                BK=triton.next_power_of_2(k),
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
                length,
                k,
                m == 1,
                BLOCK_R=2,
                BLOCK_K=triton.next_power_of_2(k),
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
                n,
                k,
                R=4,
                BK=triton.next_power_of_2(k),
                num_warps=4,
            )
        return out
    # Normalize inputs for the required async AIU loader.
    a = a.contiguous()
    b = b.t().contiguous().t()
    return _launch(a, scale_a, b, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, *, out):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out must be BF16, FP16 or FP32")
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


@triton.jit
def _quantize_values(value, peak, LOW_PRECISION: tl.constexpr = False):
    if LOW_PRECISION and peak > 1.0e-10:
        # FP16/BF16 inputs have at most 11 significant bits. Non-ties are
        # separated from half-integers by more than 2**-20 in relative value.
        # Rounding away three low FP32 mantissa bits restores exact ties
        # without moving a non-tie across an integer-rounding boundary.
        normalized = value * tl.div_rn(127.0, peak)
        normalized = ((normalized.to(tl.uint32, bitcast=True) + 4) & 0xFFFFFFF8).to(
            tl.float32, bitcast=True
        )
    else:
        normalized = tl.div_rn(value, peak) * 127.0
    rounded = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[normalized],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return tl.minimum(tl.maximum(rounded, -127), 127).to(tl.int8)


@libentry()
@triton.jit
def _quantize_activation_rows(
    X,
    Q,
    SCALE,
    K: tl.constexpr,
    STRIDE_R: tl.constexpr,
    STRIDE_C: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_WARPS: tl.constexpr = 4,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    value = tl.load(X + row * STRIDE_R + k * STRIDE_C, k < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-10)
    quantized = _quantize_values(value, peak, X.dtype.element_ty != tl.float32)
    tl.store(Q + row * K + k, quantized, k < K)
    tl.store(SCALE + row, peak * (1.0 / 127.0))


@libentry()
@triton.jit
def _activation_partial_peaks(
    X,
    PEAK,
    K: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    k = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * SR + k * SC, k < K, other=0).to(tl.float32)
    tl.store(PEAK + row * PARTS + part, tl.max(tl.abs(x), 0))


@libentry()
@triton.jit
def _quantize_activation_chunks(
    X,
    PEAK,
    Q,
    SCALE,
    K: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    PARTS: tl.constexpr,
    BP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    p = tl.arange(0, BP)
    peaks = tl.load(PEAK + row * PARTS + p, p < PARTS, other=0)
    peak = tl.maximum(tl.max(peaks, 0), 1.0e-10)
    k = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * SR + k * SC, k < K, other=0).to(tl.float32)
    tl.store(
        Q + row * K + k,
        _quantize_values(value, peak, X.dtype.element_ty != tl.float32),
        k < K,
    )
    if part == 0:
        tl.store(SCALE + row, peak * (1.0 / 127.0))


def _validate_activation_inputs(a, b, scale_b):
    """Validate the public floating-activation / INT8-weight contract."""
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_b)):
        raise TypeError("mm_w8a8_int8 expects Tensor inputs and weight scales")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("A must be an FP16, BF16 or FP32 tensor")
    if b.dtype != torch.int8:
        raise TypeError("B must be a prequantized torch.int8 tensor")
    if a.device.type != "cuda" or any(x.device != a.device for x in (b, scale_b)):
        raise ValueError("inputs and scales must be on the same PPU device")
    m, k = a.shape
    n = b.shape[1]
    if scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected scale_b[N] or [1,N]")
    if scale_b.dtype != torch.float32 or not scale_b.is_contiguous():
        raise ValueError("scale_b must be a contiguous FP32 tensor")
    return m, n, k


def _prepare_mm_w8a8_int8_inputs(a, b, scale_b):
    m, n, k = _validate_activation_inputs(a, b, scale_b)
    a_q = torch.empty((m, k), device=a.device, dtype=torch.int8)
    # Empty outputs and reductions must not call amax on an empty axis.
    if m == 0 or n == 0 or k == 0:
        return a_q, b, torch.ones(m, device=a.device, dtype=torch.float32), scale_b
    scale_a = torch.empty(m, device=a.device, dtype=torch.float32)
    with torch_device_fn.device(a.device):
        if k <= 32768:
            # LibEntry keys include constexpr arguments, not launch metadata.
            # Keep small-row and throughput quantizers in separate cache entries.
            quant_warps = 32 if m <= 32 else (8 if k <= 512 else 4)
            _quantize_activation_rows[(m,)](
                a,
                a_q,
                scale_a,
                k,
                *a.stride(),
                BLOCK=triton.next_power_of_2(k),
                NUM_WARPS=quant_warps,
                num_warps=quant_warps,
            )
        else:
            parts = triton.cdiv(k, 4096)
            peaks = torch.empty((m, parts), device=a.device, dtype=torch.float32)
            _activation_partial_peaks[(m, parts)](
                a,
                peaks,
                k,
                *a.stride(),
                parts,
                BLOCK=4096,
                num_warps=4,
            )
            _quantize_activation_chunks[(m, triton.cdiv(k, 1024))](
                a,
                peaks,
                a_q,
                scale_a,
                k,
                *a.stride(),
                parts,
                BP=triton.next_power_of_2(parts),
                BLOCK=1024,
                num_warps=4,
            )
    return a_q, b.t().contiguous().t(), scale_a, scale_b


@libentry()
@triton.jit
def _long_mv_partials(
    A,
    B,
    PEAK,
    PARTIAL,
    SCALE,
    K: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    BS: tl.constexpr,
    PARTS: tl.constexpr,
    BP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    ps = tl.arange(0, BP)
    peaks = tl.load(PEAK + row * PARTS + ps, ps < PARTS, other=0)
    peak = tl.maximum(tl.max(peaks, 0), 1.0e-10)
    k = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + row * SR + k * SC, k < K, other=0).to(tl.float32)
    aq = _quantize_values(a, peak, A.dtype.element_ty != tl.float32).to(tl.int32)
    b = tl.load(B + k * BS, k < K, other=0).to(tl.int32)
    tl.store(PARTIAL + row * PARTS + part, tl.sum(aq * b, 0))
    if part == 0:
        tl.store(SCALE + row, peak * (1.0 / 127.0))


@libentry()
@triton.jit
def _long_mv_reduce(PARTIAL, SCALE, SB, OUT, PARTS: tl.constexpr, BP: tl.constexpr):
    row = tl.program_id(0)
    ps = tl.arange(0, BP)
    p = tl.load(PARTIAL + row * PARTS + ps, ps < PARTS, other=0)
    acc = tl.sum(p, 0).to(tl.float32)
    sa, sb = tl.load(SCALE + row), tl.load(SB)
    tl.store(OUT + row, acc * sa * sb)


@libentry()
@triton.jit
def _fused_single_row(
    A, B, S, OUT, N: tl.constexpr, K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr
):
    m = tl.program_id(1)
    r = tl.program_id(0) * R + tl.arange(0, R)
    k = tl.arange(0, BK)
    a = tl.load(A + m * K + k, k < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(a), 0), 1.0e-10)
    q = _quantize_values(a, peak, A.dtype.element_ty != tl.float32).to(tl.int32)
    q = tl.reshape(q, (BK // 4, 4)).to(tl.uint32) & 255
    shift = tl.arange(0, 4) * 8
    packed = tl.sum(q << shift[None, :], 1).to(tl.int32)
    kk = tl.arange(0, BK // 4)
    b = tl.load(
        B.to(tl.pointer_type(tl.int32)) + r[:, None] * (K // 4) + kk[None, :],
        (r[:, None] < N) & (kk[None, :] < K // 4),
        other=0,
    )
    partial = tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, 0;",
        constraints="=r,r,r",
        args=[packed[None, :], b],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    acc = tl.sum(partial, 1)
    sb = tl.load(S + r, r < N, other=0)
    tl.store(OUT + m * N + r, acc.to(tl.float32) * (peak * (1.0 / 127.0)) * sb, r < N)


@libentry()
@triton.jit
def _small_rows(
    A,
    B,
    SA,
    SB,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
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
    sa = tl.load(SA + m)
    sb = tl.load(SB + r, r < N, other=0)
    tl.store(OUT + m * N + r, acc.to(tl.float32) * sa * sb, r < N)


@libentry()
@triton.jit
def _persistent_long_mv(
    A,
    B,
    S,
    OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    ks = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), M, PROGRAMS):
        vmax = tl.zeros((BLOCK,), tl.float32)
        for part in range(triton.cdiv(K, BLOCK)):
            k = part * BLOCK + ks
            a = tl.load(A + row * K + k, k < K, other=0, cache_modifier=".ca").to(
                tl.float32
            )
            vmax = tl.maximum(vmax, tl.abs(a))
        peak = tl.maximum(tl.max(vmax, 0), 1.0e-10)
        inv = tl.div_rn(127.0, peak)
        acc = tl.zeros((BLOCK,), tl.int32)
        for part in range(triton.cdiv(K, BLOCK)):
            k = part * BLOCK + ks
            a = tl.load(A + row * K + k, k < K, other=0, cache_modifier=".ca").to(
                tl.float32
            )
            if peak > 1.0e-10:
                norm = a * inv
                norm = ((norm.to(tl.uint32, bitcast=True) + 4) & 0xFFFFFFF8).to(
                    tl.float32, bitcast=True
                )
            else:
                norm = tl.div_rn(a, peak) * 127.0
            q = tl.inline_asm_elementwise(
                "cvt.rni.s32.f32 $0, $1;",
                constraints="=r,f",
                args=[norm],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            b = tl.load(B + k, k < K, other=0).to(tl.int32)
            acc += q * b
        result = tl.sum(acc, 0).to(tl.float32) * (peak * (1.0 / 127.0)) * tl.load(S)
        tl.store(OUT + row, result)


def _run_activation_mm(a, b, scale_b, out, m, n, k):
    if (
        m == 1
        and 128 <= k <= 4096
        and k % 4 == 0
        and 0 < n <= 1024
        and a.is_contiguous()
        and b.stride() == (1, k)
        and a.dtype != torch.float32
        and b.data_ptr() % 4 == 0
    ):
        with torch_device_fn.device(a.device):
            _fused_single_row[(triton.cdiv(n, 2), 1)](
                a,
                b,
                scale_b,
                out,
                n,
                k,
                R=2,
                BK=triton.next_power_of_2(k),
                num_warps=4,
            )
        return out
    if (
        m >= 256
        and n == 1
        and k > 32768
        and a.dtype != torch.float32
        and a.is_contiguous()
        and b.stride(0) == 1
    ):
        with torch_device_fn.device(a.device):
            _persistent_long_mv[(128,)](
                a,
                b,
                scale_b,
                out,
                m,
                k,
                BLOCK=8192,
                PROGRAMS=128,
                num_warps=16,
            )
        return out
    if m and n == 1 and k > 32768:
        parts = triton.cdiv(k, 4096)
        peaks = torch.empty((m, parts), device=a.device, dtype=torch.float32)
        partial = torch.empty((m, parts), device=a.device, dtype=torch.int32)
        scale = torch.empty(m, device=a.device, dtype=torch.float32)
        with torch_device_fn.device(a.device):
            _activation_partial_peaks[(m, parts)](
                a,
                peaks,
                k,
                *a.stride(),
                parts,
                BLOCK=4096,
                num_warps=4,
            )
            _long_mv_partials[(m, parts)](
                a,
                b,
                peaks,
                partial,
                scale,
                k,
                *a.stride(),
                b.stride(0),
                parts,
                BP=triton.next_power_of_2(parts),
                BLOCK=4096,
                num_warps=4,
            )
            _long_mv_reduce[(m,)](
                partial,
                scale,
                scale_b,
                out,
                parts,
                BP=triton.next_power_of_2(parts),
                num_warps=4,
            )
        return out
    aq, bq, sa, sb = _prepare_mm_w8a8_int8_inputs(a, b, scale_b)
    return _run_mm(aq, bq, sa, sb, out, m, n, k)


def mm_w8a8_int8(a, b, scale_b, *, out_dtype=None):
    """Multiply floating A[M,K] by prequantized INT8 weights B[K,N].

    A supports FP16, BF16 and FP32. scale_b is a contiguous FP32 tensor of
    shape [N] or [1,N], with dequantized weights B * scale_b. Weights use
    symmetric quantization with zero point 0; no weight quantization occurs.
    Column-major B avoids a layout copy; other strided inputs are supported.
    A is dynamically quantized per row to [-127,127] using round-to-even.
    INT32 products are scaled by the activation and weight scales in FP32.
    Output defaults to BF16, with FP16 and FP32 also supported. All inputs
    must reside on the same PPU. Input and scale updates are read on Graph replay.
    """
    logger.debug("GEMS MM_W8A8_INT8")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    m, n, k = _validate_activation_inputs(a, b, scale_b)
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run_activation_mm(a, b, scale_b, out, m, n, k)


def mm_w8a8_int8_out(a, b, scale_b, *, out):
    """Write mm_w8a8_int8(a, b, scale_b) into contiguous FP16/BF16/FP32 out."""
    logger.debug("GEMS MM_W8A8_INT8_OUT")
    m, n, k = _validate_activation_inputs(a, b, scale_b)
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out must be BF16, FP16 or FP32")
    return _run_activation_mm(a, b, scale_b, out, m, n, k)
