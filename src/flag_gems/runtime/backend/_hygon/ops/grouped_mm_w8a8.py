# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .bmm_w8a8 import MAX_INT32_REDUCTION


@libentry()
@triton.jit
def grouped_mm_w8a8_prefix(
    Rows, RowEnds, TileEnds, E: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr
):
    experts = tl.arange(0, BLOCK)
    rows = tl.load(Rows + experts, experts < E, other=0)
    tl.store(RowEnds + experts, tl.cumsum(rows, 0), experts < E)
    tl.store(TileEnds + experts, tl.cumsum(tl.cdiv(rows, BM), 0), experts < E)


@libentry()
@triton.jit
def grouped_mm_w8a8_kernel(
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
):
    tiles_n: tl.constexpr = (N + BN - 1) // BN
    pid = tl.program_id(0)
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
    scale_b = tl.load(Bs + weight_col, col < N, other=0)
    for tile in range(tile_count):
        row = row_start + tile * BM + tl.arange(0, BM)
        acc = tl.zeros((BM, BN), tl.int32)
        for block in range(tl.cdiv(K, BK)):
            red = block * BK + tl.arange(0, BK)
            a = tl.load(
                A + row[:, None].to(tl.int64) * K + red[None, :],
                (row[:, None] < row_end) & (red[None, :] < K),
                other=0,
            )
            b = tl.load(
                B + weight_col[None, :] * K + red[:, None],
                (red[:, None] < K) & (col[None, :] < N),
                other=0,
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        scale_a = tl.load(As + row, row < row_end, other=0)
        value = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
        tl.store(
            C + row[:, None].to(tl.int64) * N + col[None, :],
            value,
            (row[:, None] < row_end) & (col[None, :] < N),
        )


def grouped_mm_w8a8(
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
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("grouped_mm_w8a8 expects tensor operands")
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("expected A[M,K] and B[E,N,K]")
    m, k = A.shape
    e, n, weight_k = B.shape
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("Hygon grouped_mm_w8a8 requires signed INT8 inputs")
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
    bm, bn, bk = (16, 64, 128) if direct else (64, 64, 64)
    with torch_device_fn.device(A.device):
        if direct:
            row_ends, tile_ends = None, None
            max_tiles = m
        else:
            metadata = torch.empty((2, e), dtype=torch.int32, device=A.device)
            row_ends, tile_ends = metadata[0], metadata[1]
            grouped_mm_w8a8_prefix[(1,)](
                m_rows, row_ends, tile_ends, e, bm, triton.next_power_of_2(e)
            )
            max_tiles = min(m, triton.cdiv(m, bm) + e - 1)
        grouped_mm_w8a8_kernel[(max_tiles * triton.cdiv(n, bn),)](
            A,
            As,
            B,
            Bs,
            C,
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
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
