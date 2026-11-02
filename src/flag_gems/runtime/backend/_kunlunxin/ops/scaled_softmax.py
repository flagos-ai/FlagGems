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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def scaled_softmax_forward_multirow_kernel(
    output_ptr,
    input_ptr,
    scale_factor,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    # Single-pass contiguous [M, N] path: TILE_M full rows stay resident in the
    # kernel, so the row max / sum-of-exp are 2D axis-1 reductions and the input
    # is read exactly once (same structure as the tuned `softmax_kernel_multirow`
    # in _kunlunxin/ops/softmax.py, with the scale fused into the load).
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]

    s = tl.load(input_ptr + off)
    s = s * scale_factor

    m = tl.max(s, axis=1)
    e = tl.exp(s - m[:, None])
    z = tl.sum(e, axis=1)
    out = e * (1.0 / z)[:, None]

    tl.store(output_ptr + off, out)


@libentry()
@triton.jit
def scaled_softmax_forward_row_kernel(
    output_ptr,
    input_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_K: tl.constexpr,
):
    # One program per (batch, head, query) row: the whole row is a 1D tile so the
    # per-row max/sum reductions stay 1D (2D (1, BLOCK_K) tiles lower badly on XPU:
    # BLOCK_Q>1 fails to compile with `uni_sram`, and even BLOCK_Q=1 is ~2 orders of
    # magnitude slower than a 1D row kernel for the same BLOCK_K). Used for strided
    # inputs and non power-of-two key lengths, where the multirow path does not fit.
    query_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    row_start_ptr = (
        input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_idx * stride_q
    )

    m = -float("inf")
    exp_sum = 0.0

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        col_mask = k_offsets < key_seq_len

        s_block = tl.load(
            row_start_ptr + k_offsets,
            mask=col_mask,
            other=-float("inf"),
            cache_modifier=".ca",
        )
        s_block = s_block * scale_factor

        m_new = tl.max(s_block, axis=0)
        m_old = m
        m = tl.maximum(m_old, m_new)

        s_prev = tl.exp(m_old - m)
        exp_sum = exp_sum * s_prev

        s_curr = tl.exp(s_block - m)
        l_new = tl.sum(tl.where(col_mask, s_curr, 0.0), axis=0)
        exp_sum = exp_sum + l_new

    exp_sum_inv = 1.0 / exp_sum

    out_row_start_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_idx * stride_q
    )

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        col_mask = k_offsets < key_seq_len

        s_block = tl.load(
            row_start_ptr + k_offsets,
            mask=col_mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        )

        s_block = s_block * scale_factor
        s_block = s_block - m
        p_block = tl.exp(s_block)
        p_block = p_block * exp_sum_inv

        tl.store(
            out_row_start_ptr + k_offsets,
            p_block,
            mask=col_mask,
            cache_modifier=".cs",
        )


def _rows_per_program(key_seq_len: int, n_rows: int) -> int:
    """Pick how many whole rows one program should own.

    Tuned on XPU3: the best tile is roughly 8-16K elements, and the optimum
    shifts for very long rows (where register pressure caps the tile).
    Always returns a positive divisor of ``n_rows``.
    """
    if key_seq_len <= 128:
        rows = 64
    elif key_seq_len <= 1024:
        rows = 8192 // key_seq_len
    elif key_seq_len <= 2048:
        rows = 16
    elif key_seq_len <= 4096:
        rows = 4
    else:
        rows = 2
    while rows > 1 and n_rows % rows:
        rows //= 2
    return rows if rows > 0 and n_rows % rows == 0 else 0


def scaled_softmax_forward(input_t: torch.Tensor, scale_factor: float):
    logger.debug("GEMS_KUNLUNXIN SCALED_SOFTMAX_FORWARD")
    assert input_t.dim() == 4, "expected 4D tensor"
    batch_size, attn_heads, query_seq_len, key_seq_len = input_t.shape
    assert input_t.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert key_seq_len <= 16384, "Key sequence length must be 16384 or less"
    assert key_seq_len % 8 == 0, "Key sequence length must be divisible by 8"
    assert query_seq_len > 1, "Query sequence length must be greater than 1"

    output_t = torch.empty_like(input_t)

    # Fast path: contiguous [M, N] with N a power of two -> single-pass multirow
    # kernel (one read + one write per element). Same structure as the tuned
    # `softmax_kernel_multirow` in _kunlunxin/ops/softmax.py; the scale is fused
    # into the load and the normalise uses a per-row reciprocal instead of a
    # per-element divide.
    if input_t.is_contiguous() and key_seq_len >= 2:
        num_rows = batch_size * attn_heads * query_seq_len
        if (key_seq_len & (key_seq_len - 1)) == 0:
            # TILE_M buckets measured on the official matrix (fp16, XPU3):
            # N<=128 -> 64, N=256 -> 32, N<=2048 -> 64, N>2048 -> 256. The tile
            # is then shrunk until it divides the row count (no row mask in the
            # kernel), giving up to TILE_M=1 for odd row counts.
            if key_seq_len <= 128:
                tile_m = 64
            elif key_seq_len <= 256:
                tile_m = 32
            elif key_seq_len <= 2048:
                tile_m = 64
            else:
                tile_m = 256
            while tile_m > 1 and num_rows % tile_m != 0:
                tile_m >>= 1
            grid = (num_rows // tile_m,)
            if tile_m * key_seq_len > 8192:
                scaled_softmax_forward_multirow_kernel[grid](
                    output_t,
                    input_t,
                    scale_factor,
                    N=key_seq_len,
                    TILE_M=tile_m,
                    num_warps=4,
                    buffer_size_limit=2048,
                )
            else:
                scaled_softmax_forward_multirow_kernel[grid](
                    output_t,
                    input_t,
                    scale_factor,
                    N=key_seq_len,
                    TILE_M=tile_m,
                    num_warps=4,
                )
            return output_t

    stride_b = input_t.stride(0)
    stride_h = input_t.stride(1)
    stride_q = input_t.stride(2)

    # Strided / non-power-of-two fallback: one program per query row; constant
    # tuple grid (a callable grid is folded into the XPU compile-cache key, see
    # harness knowledge §2.4-41).
    grid = (query_seq_len, attn_heads, batch_size)

    scaled_softmax_forward_row_kernel[grid](
        output_t,
        input_t,
        scale_factor,
        query_seq_len,
        key_seq_len,
        stride_b,
        stride_h,
        stride_q,
        BLOCK_K=1024,
        num_warps=4,
        num_stages=2,
    )
    return output_t


@libentry()
@triton.jit
def scaled_softmax_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_seq_tile_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    start_query_idx = query_seq_tile_idx * BLOCK_Q
    query_offsets = start_query_idx + tl.arange(0, BLOCK_Q)

    query_mask = query_offsets < query_seq_len

    output_row_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_output_row_ptr = (
        grad_output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_input_row_ptr = (
        grad_input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    D = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, cache_modifier=".ca")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, cache_modifier=".ca")

        dot_block = P_block * dP_block
        D += tl.sum(tl.where(mask, dot_block, 0.0), axis=1)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dS = grad_input_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, eviction_policy="evict_first")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, eviction_policy="evict_first")

        dZ_block = P_block * (dP_block - D[:, None])
        dS_block = scale_factor * dZ_block

        tl.store(ptr_dS, dS_block, mask=mask, cache_modifier=".cs")


def _backward_rows_per_program(key_seq_len: int, n_rows: int) -> int:
    """Pick how many whole rows one backward program should own.

    Measured separately from the forward policy: backward touches three
    streams (P, dP, dGrad) per element, and its optimum sits at a larger tile
    than forward's (roughly 16-32K elements).  Always returns a positive
    divisor of ``n_rows``.
    """
    if key_seq_len <= 256:
        rows = 64
    elif key_seq_len <= 2048:
        rows = 32
    elif key_seq_len <= 4096:
        rows = 8
    else:
        rows = 2
    while rows > 1 and n_rows % rows:
        rows //= 2
    return rows if rows > 0 and n_rows % rows == 0 else 0


@libentry()
@triton.jit
def scaled_softmax_backward_rows_kernel(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    scale_factor,
    KK: tl.constexpr,
    ROWS: tl.constexpr,
    F32_MATH: tl.constexpr,
):
    """Whole-row softmax backward: ROWS complete (query, key) rows per program.

    Same straight-line ``[ROWS, KK]`` structure as the forward fast path: one
    reduction for ``D = sum(P * dP)`` over the row, then the elementwise
    ``P * (dP - D) * scale`` store.  A streaming ``[1, BLOCK_K]`` tile with a
    dynamic two-pass loop over the keys collapses to ~1 GB/s on this backend.

    ``F32_MATH`` is a dtype-gated body choice.  fp16 arithmetic is 13-16%
    faster (the reference computation is fp16 as well, so results stay within
    one ulp); bf16 gets no benefit from the low-precision body, so it keeps the
    fp32 one.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, KK)
    offs = rows[:, None] * KK + cols[None, :]
    if F32_MATH:
        P = tl.load(output_ptr + offs).to(tl.float32)
        dP = tl.load(grad_output_ptr + offs).to(tl.float32)
        D = tl.sum(P * dP, axis=1)
        dS = P * (dP - D[:, None])
    else:
        P = tl.load(output_ptr + offs)
        dP = tl.load(grad_output_ptr + offs)
        D = tl.sum(P * dP, axis=1)
        dS = P * (dP - D[:, None])
    tl.store(grad_input_ptr + offs, dS * scale_factor)


def scaled_softmax_backward(
    grad_output: torch.Tensor, softmax_results: torch.Tensor, scale_factor: float
):
    logger.debug("GEMS_KUNLUNXIN SCALED_SOFTMAX_BACKWARD")
    assert grad_output.dim() == 4, "expected 4D tensor"
    assert softmax_results.dim() == 4, "expected 4D tensor"
    assert grad_output.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"
    assert softmax_results.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"

    batch_size, attn_heads, query_seq_len, key_seq_len = softmax_results.shape
    n_rows = batch_size * attn_heads * query_seq_len

    # Fast path: both operands are contiguous and share one layout, so the
    # tensor is just n_rows independent rows of length key_seq_len.  It needs a
    # power-of-two key length (for tl.arange) and a row count that divides
    # evenly so the tile stays unmasked.  The output buffer is contiguous, so
    # the returned grad_input has the same layout as the (contiguous) input.
    rows_per_program = 0
    if (
        grad_output.shape == softmax_results.shape
        and grad_output.dtype == softmax_results.dtype
        and grad_output.is_contiguous()
        and softmax_results.is_contiguous()
        and (key_seq_len & (key_seq_len - 1)) == 0
    ):
        rows_per_program = _backward_rows_per_program(key_seq_len, n_rows)

    if rows_per_program:
        grad_input = torch.empty_like(grad_output)
        grid = (n_rows // rows_per_program,)
        scaled_softmax_backward_rows_kernel[grid](
            grad_input,
            grad_output,
            softmax_results,
            scale_factor,
            KK=key_seq_len,
            ROWS=rows_per_program,
            F32_MATH=grad_output.dtype != torch.float16,
            num_warps=4,
            num_stages=2,
        )
        return grad_input

    # General path: non-contiguous / odd sizes keep the original masked
    # two-pass kernel.  Both operands are made contiguous first (a plain
    # strided copy, not a CPU/native numerical fallback), which also fixes the
    # strides used below.  The grid is a plain tuple (not a callable): a
    # callable grid participates in the compilation cache key on this backend.
    grad_output = grad_output.contiguous()
    softmax_results = softmax_results.contiguous()

    grad_input = torch.empty_like(grad_output)

    stride_b = softmax_results.stride(0)
    stride_h = softmax_results.stride(1)
    stride_q = softmax_results.stride(2)

    grid = (query_seq_len, attn_heads, batch_size)
    scaled_softmax_backward_kernel[grid](
        grad_input,
        grad_output,
        softmax_results,
        scale_factor,
        query_seq_len,
        key_seq_len,
        stride_b,
        stride_h,
        stride_q,
        BLOCK_Q=1,
        BLOCK_K=128,
        num_warps=4,
        num_stages=2,
    )

    return grad_input
