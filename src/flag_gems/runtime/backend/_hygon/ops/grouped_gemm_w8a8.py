# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib.util import find_spec

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .bmm_w8a8 import MAX_INT32_REDUCTION

logger = logging.getLogger(__name__)

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language import BlockedLayout, SliceLayout
    from triton.experimental.gluon.language.amd import _layouts as amd_layouts
    from triton.experimental.gluon.language.amd.cdna3 import buffer_load
except ImportError:
    _HAS_GEMV = False
else:
    # Some builds expose the Gluon API but lack the layout class used by HCU
    # layout inference. The vendor compiler supports this kernel without it.
    _HAS_GEMV = hasattr(amd_layouts, "HCUMFMALayout")
    if not _HAS_GEMV:
        try:
            _HAS_GEMV = find_spec("triton.backends.amd.compiler_hcu") is not None
        except ModuleNotFoundError:
            pass

    @gluon.jit
    def _dot_packed_int8(a, b, acc):
        return gl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3",
            "=v,v,v,v",
            [b, a, acc],
            dtype=gl.int32,
            is_pure=True,
            pack=1,
        )

    @gluon.jit
    def _store_gemv_row(
        A,
        As,
        C,
        row,
        k,
        cols,
        b0,
        b1,
        b2,
        scale_b,
        N: gl.constexpr,
        layout: gl.constexpr,
    ):
        off = row * 1536 + k
        acc = gl.full((2, 2, 256), 0, gl.int32, layout)
        acc = _dot_packed_int8(buffer_load(A, off), b0, acc)
        acc = _dot_packed_int8(buffer_load(A, off + 256), b1, acc)
        acc = _dot_packed_int8(buffer_load(A, off + 512), b2, acc)
        value = gl.sum(gl.sum(acc, 2), 0).to(gl.float32) * gl.load(As + row) * scale_b
        gl.store(C + row * N + cols, value)

    @gluon.jit
    def _grouped_gemm_w8a8_gemv(
        A, As, B, Bs, C, RowIndices, Rows, M: gl.constexpr, N: gl.constexpr
    ):
        row = gl.program_id(1)
        expert = gl.load(RowIndices + row)
        previous = gl.load(RowIndices + row - 1, row > 0, other=-1)
        if (row % 2 == 1) & (expert == previous):
            return
        next_expert = gl.load(RowIndices + row + 1, row + 1 < M, other=-1)
        has_second = (row % 2 == 0) & (expert == next_expert)
        # Pair two-row experts that start on an odd packed row.
        if (row % 2 == 0) & (expert == previous) & (expert != next_expert):
            second_previous = gl.load(RowIndices + row - 2, row >= 2, other=-1)
            if expert != second_previous:
                return
        if (row % 2 == 1) & (expert != previous) & (expert == next_expert):
            second_next = gl.load(RowIndices + row + 2, row + 2 < M, other=-1)
            has_second = expert != second_next

        # Each wave covers half of K for two output columns.
        K_WORDS: gl.constexpr = 6144 // 4
        BLOCK_WORDS: gl.constexpr = 1024 // 4
        layout: gl.constexpr = BlockedLayout(
            [1, 1, 4], [1, 1, 64], [2, 1, 1], [2, 0, 1]
        )
        col_layout: gl.constexpr = SliceLayout(0, SliceLayout(2, layout))
        part = gl.arange(0, 2, layout=SliceLayout(1, SliceLayout(2, layout)))
        cols = gl.program_id(0) * 2 + gl.arange(0, 2, layout=col_layout)
        words = gl.arange(0, BLOCK_WORDS, layout=SliceLayout(0, SliceLayout(1, layout)))
        k = part[:, None, None] * (K_WORDS // 2) + words[None, None, :]
        a = A.to(gl.pointer_type(gl.int32))
        b = B.to(gl.pointer_type(gl.int32))
        w = (expert * N + cols[None, :, None]) * K_WORDS + k
        scale_a = gl.load(As + row)
        scale_b = gl.load(Bs + expert * N + cols)
        b0 = buffer_load(b, w)
        b1 = buffer_load(b, w + BLOCK_WORDS)
        b2 = buffer_load(b, w + 2 * BLOCK_WORDS)
        # Keep a single row accumulator live for pairs in longer expert runs.
        if has_second and gl.load(Rows + expert) > 2:
            _store_gemv_row(a, As, C, row, k, cols, b0, b1, b2, scale_b, N, layout)
            _store_gemv_row(a, As, C, row + 1, k, cols, b0, b1, b2, scale_b, N, layout)
            return

        acc = gl.full((2, 2, BLOCK_WORDS), 0, gl.int32, layout)
        acc2 = gl.full((2, 2, BLOCK_WORDS), 0, gl.int32, layout)
        for block in gl.static_range(3):
            if block == 0:
                weights = b0
            elif block == 1:
                weights = b1
            else:
                weights = b2
            position = k + block * BLOCK_WORDS
            acc = _dot_packed_int8(
                buffer_load(a, row * K_WORDS + position), weights, acc
            )
            if has_second:
                acc2 = _dot_packed_int8(
                    buffer_load(a, (row + 1) * K_WORDS + position), weights, acc2
                )
        value = gl.sum(gl.sum(acc, 2), 0).to(gl.float32) * scale_a * scale_b
        gl.store(C + row * N + cols, value)
        if has_second:
            scale_a2 = gl.load(As + row + 1)
            value2 = gl.sum(gl.sum(acc2, 2), 0).to(gl.float32) * scale_a2 * scale_b
            gl.store(C + (row + 1) * N + cols, value2)


@libentry()
@triton.jit
def grouped_gemm_w8a8_prefix(
    Rows, RowEnds, TileEnds, E: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr
):
    experts = tl.arange(0, BLOCK)
    rows = tl.load(Rows + experts, experts < E, other=0)
    tl.store(RowEnds + experts, tl.cumsum(rows, 0), experts < E)
    tl.store(TileEnds + experts, tl.cumsum(tl.cdiv(rows, BM), 0), experts < E)


@libentry()
@triton.jit
def grouped_gemm_w8a8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowIndices,
    Rows,
    RowEnds,
    TileEnds,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DIRECT: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    M: tl.constexpr,
):
    tiles_n: tl.constexpr = (N + BN - 1) // BN
    pid = tl.program_id(0)
    if SPLIT_K > 1:
        tl.static_assert(EVEN_N and K % (BK * SPLIT_K) == 0)
        partition = pid % SPLIT_K
        pid = pid // SPLIT_K
    tile_m = pid // tiles_n
    col = (pid % tiles_n) * BN + tl.arange(0, BN)
    if DIRECT:
        # A packed run has one owner even if it spans several M tiles.
        row_start = tile_m
        expert = tl.load(RowIndices + row_start)
        previous = tl.load(RowIndices + row_start - 1, row_start > 0, other=-1)
        if expert == previous:
            return
        row_end = row_start + tl.load(Rows + expert)
        tile_count = tl.cdiv(row_end - row_start, BM)
    else:
        if tile_m >= tl.load(TileEnds + E - 1):
            return
        lo, hi = 0, E - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if tile_m < tl.load(TileEnds + mid):
                hi = mid
            else:
                lo = mid + 1
        expert = lo
        start = tl.load(RowEnds + expert - 1, expert > 0, other=0)
        first_tile = tl.load(TileEnds + expert - 1, expert > 0, other=0)
        row_start = start + (tile_m - first_tile) * BM
        row_end = tl.load(RowEnds + expert)
        tile_count = 1

    weight_col = expert.to(tl.int64) * N + col
    if EVEN_N:
        scale_b = tl.load(Bs + weight_col)
    else:
        scale_b = tl.load(Bs + weight_col, col < N, other=0)
    for tile in range(tile_count):
        row = row_start + tile * BM + tl.arange(0, BM)
        acc = tl.zeros((BM, BN), tl.int32)
        blocks: tl.constexpr = tl.cdiv(K, BK * SPLIT_K)
        for block in range(blocks):
            k_block = block
            if SPLIT_K > 1:
                k_block += partition * blocks
            red = k_block * BK + tl.arange(0, BK)
            a_ptrs = A + row[:, None].to(tl.int64) * K + red[None, :]
            b_ptrs = B + weight_col[None, :] * K + red[:, None]
            if EVEN_K:
                a = tl.load(a_ptrs, row[:, None] < row_end, other=0)
            else:
                a = tl.load(
                    a_ptrs,
                    (row[:, None] < row_end) & (red[None, :] < K),
                    other=0,
                )
            if EVEN_K and EVEN_N:
                b = tl.load(b_ptrs)
            elif EVEN_K:
                b = tl.load(b_ptrs, col[None, :] < N, other=0)
            else:
                b = tl.load(
                    b_ptrs,
                    (red[:, None] < K) & (col[None, :] < N),
                    other=0,
                )
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        if SPLIT_K > 1:
            offset = (partition * M + row[:, None]).to(tl.int64) * N + col[None, :]
            tl.store(C + offset, acc, row[:, None] < row_end)
        else:
            scale_a = tl.load(As + row, row < row_end, other=0)
            value = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
            c_ptrs = C + row[:, None].to(tl.int64) * N + col[None, :]
            if EVEN_N:
                tl.store(c_ptrs, value, row[:, None] < row_end)
            else:
                tl.store(
                    c_ptrs,
                    value,
                    (row[:, None] < row_end) & (col[None, :] < N),
                )


@libentry()
@triton.jit
def _grouped_gemm_w8a8_reduce_split_k(
    Partials,
    As,
    Bs,
    RowIndices,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < M * N
    row = index // N
    col = index % N
    acc = tl.zeros((BLOCK,), tl.int32)
    for partition in tl.static_range(SPLIT_K):
        acc += tl.load(Partials + partition * M * N + index, valid, other=0)
    expert = tl.load(RowIndices + row, valid, other=0)
    scale_a = tl.load(As + row, valid, other=0)
    scale_b = tl.load(Bs + expert.to(tl.int64) * N + col, valid, other=0)
    tl.store(Out + index, acc.to(tl.float32) * scale_a * scale_b, valid)


@libentry()
@triton.jit
def _grouped_gemm_w8a8_direct_k256(
    A, As, B, Bs, C, RowIndices, Rows, N: tl.constexpr, EVEN_N: tl.constexpr
):
    BM: tl.constexpr = 16
    BN: tl.constexpr = 64
    BK: tl.constexpr = 128
    K: tl.constexpr = 256
    pid = tl.program_id(0)
    tiles_n: tl.constexpr = tl.cdiv(N, BN)
    tile_m = pid // tiles_n
    col = (pid % tiles_n) * BN + tl.arange(0, BN)
    # A packed run has one owner even if it spans several M tiles.
    row_start = tile_m
    expert = tl.load(RowIndices + row_start)
    previous = tl.load(RowIndices + row_start - 1, row_start > 0, other=-1)
    if expert == previous:
        return
    row_end = row_start + tl.load(Rows + expert)
    tile_count = tl.cdiv(row_end - row_start, BM)
    weight_col = expert.to(tl.int64) * N + col
    if EVEN_N:
        scale_b = tl.load(Bs + weight_col)
    else:
        scale_b = tl.load(Bs + weight_col, col < N, other=0)
    for tile in range(tile_count):
        row = row_start + tile * BM + tl.arange(0, BM)
        red = tl.arange(0, BK)
        a_base = A + row[:, None].to(tl.int64) * K + red[None, :]
        b_base = B + weight_col[None, :] * K + red[:, None]
        row_mask = row[:, None] < row_end
        col_mask = col[None, :] < N
        a0 = tl.load(a_base, row_mask, other=0)
        b0 = tl.load(b_base, col_mask, other=0)
        a1 = tl.load(a_base + BK, row_mask, other=0)
        b1 = tl.load(b_base + BK, col_mask, other=0)
        # Keep the two K tiles on independent accumulator chains.
        acc = tl.dot(a0, b0, out_dtype=tl.int32) + tl.dot(a1, b1, out_dtype=tl.int32)
        scale_a = tl.load(As + row, row < row_end, other=0)
        value = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
        c_ptrs = C + row[:, None].to(tl.int64) * N + col[None, :]
        if EVEN_N:
            tl.store(c_ptrs, value, row[:, None] < row_end)
        else:
            tl.store(
                c_ptrs,
                value,
                (row[:, None] < row_end) & (col[None, :] < N),
            )


def grouped_gemm_w8a8(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    """Write packed INT8 NT grouped GEMM to the caller's BF16 output.

    The caller supplies sorted INT32 expert indices and matching nonnegative
    row counts summing to M. All operands are contiguous and on one device.
    """
    logger.debug("GEMS_HYGON GROUPED_GEMM_W8A8")
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("grouped_gemm_w8a8 expects tensor operands")
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("expected A[M,K] and B[E,N,K]")
    m, k = A.shape
    e, n, weight_k = B.shape
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("Hygon grouped_gemm_w8a8 requires signed INT8 inputs")
    if (As.dtype, Bs.dtype, C.dtype, m_indices.dtype, m_rows.dtype) != (
        torch.float32,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int32,
    ):
        raise TypeError("expected FP32 scales, BF16 output and INT32 routing")
    if A.device.type != "cuda" or any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same Hygon device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    if weight_k != k or (
        As.shape,
        Bs.shape,
        C.shape,
        m_indices.shape,
        m_rows.shape,
    ) != (
        (m, 1),
        (e, n, 1),
        (m, n),
        (m,),
        (e,),
    ):
        raise ValueError("incompatible scales, output or expert metadata shapes")
    if k > MAX_INT32_REDUCTION:
        raise ValueError(f"INT8 grouped GEMM requires K <= {MAX_INT32_REDUCTION}")
    if m and not e:
        raise ValueError("nonempty input requires at least one expert")
    if not m or not n:
        return
    output_start = C.data_ptr()
    output_end = output_start + C.numel() * C.element_size()
    for tensor in (A, As, B, Bs, m_indices, m_rows):
        start = tensor.data_ptr()
        if (
            output_start < start + tensor.numel() * tensor.element_size()
            and start < output_end
        ):
            raise ValueError("C must not overlap an input")

    direct = m <= e
    large_gate_up = not direct and (n, k) == (512, 6144)
    split_k = 1
    if direct:
        bm, bn = 16, 64
        if (n, k) == (512, 6144):
            # Two K tiles per CTA expose more independent weight loads.
            bk, split_k = 256, 12
        else:
            bk = 512 if k % 512 == 0 else 128
    elif large_gate_up:
        bm, bn, bk = 128, 128, 128
    else:
        bm, bn, bk = 64, 64, 64
    with torch_device_fn.device(A.device):
        if (
            split_k > 1
            and _HAS_GEMV
            and A.data_ptr() % 16 == 0
            and B.data_ptr() % 16 == 0
            and B.numel() < 2**31
            and triton.runtime.driver.active.get_current_target().arch == "gfx936"
        ):
            _grouped_gemm_w8a8_gemv[(n // 2, m)](
                A,
                As,
                B,
                Bs,
                C,
                m_indices,
                m_rows,
                m,
                n,
                num_warps=2,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return
        if direct and k == 256:
            _grouped_gemm_w8a8_direct_k256[(m * triton.cdiv(n, 64),)](
                A,
                As,
                B,
                Bs,
                C,
                m_indices,
                m_rows,
                n,
                n % 64 == 0,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return
        if direct:
            row_ends, tile_ends = None, None
            max_tiles = m
        else:
            metadata = torch.empty((2, e), dtype=torch.int32, device=A.device)
            row_ends, tile_ends = metadata[0], metadata[1]
            grouped_gemm_w8a8_prefix[(1,)](
                m_rows, row_ends, tile_ends, e, bm, triton.next_power_of_2(e)
            )
            max_tiles = min(m, triton.cdiv(m, bm) + e - 1)
        destination = C
        if split_k > 1:
            destination = torch.empty(
                (split_k, m, n), dtype=torch.int32, device=A.device
            )
        grouped_gemm_w8a8_kernel[(max_tiles * triton.cdiv(n, bn) * split_k,)](
            A,
            As,
            B,
            Bs,
            destination,
            m_indices,
            m_rows,
            row_ends,
            tile_ends,
            n,
            k,
            e,
            bm,
            bn,
            bk,
            DIRECT=direct,
            EVEN_K=k % bk == 0,
            EVEN_N=n % bn == 0,
            SPLIT_K=split_k,
            M=m if split_k > 1 else 0,
            num_warps=4,
            num_stages=1 if large_gate_up else 2,
            enable_fp_fusion=False,
        )
        if split_k > 1:
            _grouped_gemm_w8a8_reduce_split_k[(triton.cdiv(m * n, 256),)](
                destination,
                As,
                Bs,
                m_indices,
                C,
                m,
                n,
                split_k,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
