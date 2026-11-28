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

"""GCU400 W8A8 grouped GEMM (INT8 / E4M3, NT layout).

Semantics: A[M,K] (int8 or fp8) is split into E groups described by m_indices/m_rows, each
group uses its own weight B[E,N,K] (row-major, i.e. NT: out = A @ B[g].T), the per-row scale
As[M,1] and per-column scale Bs[E,N,1] are applied, and the result goes to the caller C[M,N].
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_E4M3_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

# GEMM tiling: on GCU400, the 32×64×64 fp16 input / fp32 accumulation combination reliably goes through the matrix unit
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 64


@triton.jit
def _decode_e4m3(bits, FNUZ: tl.constexpr):
    """Hand-decodes 8-bit E4M3 storage to fp16 (the GCU backend has no direct fp8 pointer)."""
    bits = bits.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    if FNUZ:
        power = ((exponent + 119) << 23).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.0009765625
        is_nan = bits == 128
    else:
        power = ((exponent + 120) << 23).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.001953125
        is_nan = (bits & 127) == 127
    value = tl.where(exponent == 0, subnormal, power * (1.0 + mantissa * 0.125))
    value = tl.where((bits & 128) != 0, -value, value)
    return tl.where(is_nan, float("nan"), value).to(tl.float16)


@triton.jit
def _load_tiles(
    a_ptr,
    b_ptr,
    rows_safe,
    cols_safe,
    k_base,
    k_size,
    row_mask,
    col_mask,
    group_id,
    stride_am,
    stride_bn,
    BLOCK_K: tl.constexpr,
):
    """Loads a (BLOCK_M, BLOCK_K) tile of A and a (BLOCK_K, BLOCK_N) tile of B[group].

    Addresses are clamped into range then zeroed with where, because on GCU400 a masked load's
    ``other=0`` is unreliable when the whole tile is masked (it leaks uninitialized data).
    """
    k_offsets = k_base + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k_size
    k_safe = tl.minimum(k_offsets, k_size - 1)

    a_ptrs = a_ptr + rows_safe[:, None] * stride_am + k_safe[None, :]
    a_tile = tl.load(a_ptrs)
    a_tile = tl.where(row_mask[:, None] & k_mask[None, :], a_tile, 0)

    # B[group] is (N, K) row-major: address = group*N*K + n*K + k
    b_ptrs = (
        b_ptr + group_id * stride_bn + cols_safe[None, :] * k_size + k_safe[:, None]
    )
    b_tile = tl.load(b_ptrs)
    b_tile = tl.where(col_mask[None, :] & k_mask[:, None], b_tile, 0)
    return a_tile, b_tile


@triton.jit
def _grouped_mm_w8a8_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    group_start_ptr,
    group_count_ptr,
    k_size,
    n_size,
    num_rows,
    stride_am,
    stride_bn,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8: tl.constexpr,
    FNUZ: tl.constexpr,
):
    """Dim 0 is the expert, dim 1 the row block inside that expert, dim 2 the block along N."""
    group_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    group_start = tl.load(group_start_ptr + group_id)
    group_count = tl.load(group_count_ptr + group_id)

    # The grid uses max(count) as the upper bound; empty groups / out-of-bounds blocks exit immediately:
    # this both skips useless tiles and avoids the dirty data introduced by the degraded path on GCU when a whole block
    # is masked out.
    if group_count == 0:
        return
    if m_block * BLOCK_M >= group_count:
        return
    if n_block * BLOCK_N >= n_size:
        return

    row_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < group_count
    rows = group_start + row_offsets
    rows_safe = tl.minimum(rows, num_rows - 1)

    col_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < n_size
    cols_safe = tl.minimum(col_offsets, n_size - 1)

    a_scale = tl.where(row_mask, tl.load(a_scale_ptr + rows_safe), 0.0)
    b_scale = tl.where(
        col_mask,
        tl.load(b_scale_ptr + group_id * n_size + cols_safe),
        0.0,
    )

    if FP8:
        # FP8 accumulates into fp32: values after e4m3 dequantization are exact in fp16, and the matrix unit accumulates
        # internally in fp32
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_base in range(0, k_size, BLOCK_K):
            a_tile, b_tile = _load_tiles(
                a_ptr,
                b_ptr,
                rows_safe,
                cols_safe,
                k_base,
                k_size,
                row_mask,
                col_mask,
                group_id,
                stride_am,
                stride_bn,
                BLOCK_K,
            )
            accumulator += tl.dot(
                _decode_e4m3(a_tile, FNUZ), _decode_e4m3(b_tile, FNUZ)
            )
        result = accumulator * a_scale[:, None] * b_scale[None, :]
    else:
        # INT8 requires exact integer accumulation: this operator's test cases include strong cancellation scenarios
        # (±127 alternating, K=6144),
        # fp32 accumulation cannot simultaneously hold intermediate sums on the order of 5e7 and a final result on the
        # order of -127.
        accumulator_int = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k_base in range(0, k_size, BLOCK_K):
            a_tile, b_tile = _load_tiles(
                a_ptr,
                b_ptr,
                rows_safe,
                cols_safe,
                k_base,
                k_size,
                row_mask,
                col_mask,
                group_id,
                stride_am,
                stride_bn,
                BLOCK_K,
            )
            accumulator_int += tl.dot(a_tile, b_tile)
        result = accumulator_int.to(tl.float32) * a_scale[:, None] * b_scale[None, :]

    # Explicitly zero out-of-bounds positions instead of relying on masked store to discard NaN
    result = tl.where(row_mask[:, None] & col_mask[None, :], result, 0.0)

    c_ptrs = c_ptr + rows[:, None] * stride_cm + col_offsets[None, :]
    tl.store(
        c_ptrs,
        result.to(tl.bfloat16),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def grouped_mm_w8a8(
    mat_a: torch.Tensor,
    mat_a_scale: torch.Tensor,
    mat_b: torch.Tensor,
    mat_b_scale: torch.Tensor,
    mat_c: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    """Writes the W8A8 grouped NT GEMM result into the caller BF16 output."""
    logger.debug("GEMS_ENFLAMEGROUPED_MM_W8A8")
    if mat_a.ndim != 2 or mat_b.ndim != 3:
        raise ValueError("mat_a must be [M,K] and mat_b must be [E,N,K]")

    num_rows, k_size = mat_a.shape
    num_groups, n_size, weight_k = mat_b.shape
    if weight_k != k_size:
        raise ValueError("mat_a and mat_b must agree on K")
    if mat_a.dtype not in (torch.int8, *_E4M3_DTYPES) or mat_b.dtype != mat_a.dtype:
        raise TypeError("mat_a and mat_b must share INT8 or E4M3 dtype")

    if num_rows == 0 or n_size == 0:
        return

    # Starting row of each group: starts[g] = sum(m_rows[:g])
    group_start = torch.cumsum(m_rows, 0, dtype=torch.int32) - m_rows
    group_stride = int(m_rows.max())

    is_fp8 = mat_a.dtype != torch.int8
    group_nz = mat_b.dtype == torch.float8_e4m3fnuz

    # The GCU backend does not support fp8 pointers; load via a byte view and decode inside the kernel
    a_arg = mat_a.view(torch.uint8) if is_fp8 else mat_a
    b_arg = mat_b.view(torch.uint8) if is_fp8 else mat_b

    grid = (
        num_groups,
        triton.cdiv(group_stride, BLOCK_M),
        triton.cdiv(n_size, BLOCK_N),
    )
    _grouped_mm_w8a8_kernel[grid](
        a_arg,
        mat_a_scale[:, 0],
        b_arg,
        mat_b_scale[:, :, 0].reshape(-1),
        mat_c,
        group_start,
        m_rows,
        k_size,
        n_size,
        num_rows,
        mat_a.stride(0),
        n_size * k_size,
        mat_c.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        FP8=is_fp8,
        FNUZ=group_nz,
    )
