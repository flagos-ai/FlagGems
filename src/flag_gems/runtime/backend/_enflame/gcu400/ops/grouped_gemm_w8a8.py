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

"""GCU400 implementation of the per-channel INT8 grouped NT GEMM.

``grouped_gemm_w8a8(A, As, B, Bs, C, m_indices, m_rows)`` computes, for every expert
``e`` owning the contiguous row range ``[start, start + m_rows[e])``::

    C[start:end] = ((A[start:end] @ B[e].T) * As[start:end] * Bs[e, :, 0]).bf16()

``A`` is ``[M, K]`` int8, ``B`` is ``[E, N, K]`` int8 (so ``B[e].T`` is the ``[K, N]``
operand), and the scales are per row of ``A`` and per column of ``B[e]``.

Two design points drive the implementation:

* **Exact integer accumulation.** ``tl.dot`` over int8 operands accumulates in int32,
  which is exact for the largest possible magnitude here
  (``128 * 128 * K`` with ``K <= 131071`` stays below ``2**31``). Rounding therefore
  happens in exactly one place -- ``.to(float32)`` -- mirroring the reference's
  ``double -> float`` step, and the two scale multiplies are applied in the reference
  order (``(acc * As) * Bs``) so the BF16 result matches bit for bit.
* **Expert-aligned tiles without host sync.** A tile is one ``BLOCK_M``-row slab of one
  expert, so every row of a tile shares the same ``B[e]`` slab and one ``tl.dot``.
  The tile -> (expert, row range) mapping needs the per-expert lengths, which live on
  the device; a single tiny metadata program turns ``m_rows`` into that mapping so the
  host never has to read the lengths back.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 128

# K above this cannot be exactly accumulated in int32 (128*128*131071 < 2**31).
_MAX_K = 131071


@libentry()
@triton.jit
def grouped_gemm_metadata_kernel(
    m_rows,
    tile_expert,
    tile_row_start,
    tile_row_count,
    num_tiles,
    E,
    BLOCK_M: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    """Serial prefix pass: map tile index -> (expert, first row, row count).

    ``E`` is at most a few hundred and the tile count is ``M / BLOCK_M + E``, so a
    single program is enough and no inter-program ordering is needed.
    """
    row_cursor = 0
    tile_cursor = 0
    for expert in tl.range(0, E):
        row_count = tl.load(m_rows + expert).to(tl.int32)
        block_count = (row_count + BLOCK_M - 1) // BLOCK_M
        for block_index in tl.range(0, block_count):
            if tile_cursor < MAX_TILES:
                tl.store(tile_expert + tile_cursor, expert)
                tl.store(
                    tile_row_start + tile_cursor, row_cursor + block_index * BLOCK_M
                )
                remaining = row_count - block_index * BLOCK_M
                tl.store(tile_row_count + tile_cursor, tl.minimum(remaining, BLOCK_M))
            tile_cursor += 1
        row_cursor += row_count
    tl.store(num_tiles, tile_cursor)


@libentry()
@triton.jit
def grouped_gemm_tiled_kernel(
    A,
    As,
    B,
    Bs,
    C,
    tile_expert,
    tile_row_start,
    tile_row_count,
    num_tiles,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One tile of ``(int8 A rows) @ (int8 B[e].T)`` scaled into BF16 ``C``."""
    tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_tiles = tl.load(num_tiles)
    if tile < total_tiles:
        expert = tl.load(tile_expert + tile)
        row_start = tl.load(tile_row_start + tile)
        row_count = tl.load(tile_row_count + tile)

        local_m = tl.arange(0, BLOCK_M)
        rows = row_start + local_m
        mask_m = local_m < row_count
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k_block in tl.range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a_block = tl.load(
                A + rows[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0,
            )
            # B is [E, N, K]; the dot wants [K, N], i.e. B[e] read transposed.
            b_block = tl.load(
                B + expert * N * K + offs_n[None, :] * K + offs_k[:, None],
                mask=mask_n[None, :] & mask_k[:, None],
                other=0,
            )
            acc += tl.dot(a_block, b_block)

        a_scale = tl.load(As + rows, mask=mask_m, other=0.0)
        b_scale = tl.load(Bs + expert * N + offs_n, mask=mask_n, other=0.0)
        # Same operand order as the reference: (dot * As) * Bs, one rounding each step.
        value = (acc.to(tl.float32) * a_scale[:, None]) * b_scale[None, :]
        tl.store(
            C + rows[:, None] * N + offs_n[None, :],
            value.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
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
    """Write the per-channel INT8 grouped NT GEMM into the supplied BF16 output.

    Rows of ``A``/``C`` are contiguous expert groups in ascending expert order;
    ``m_rows`` holds the nonnegative group lengths.
    """
    logger.debug("GEMS_ENFLAME GROUPED_GEMM_W8A8")
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("A must be [M,K] and B must be [E,N,K]")
    M, K = A.shape
    E, N, weight_k = B.shape
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    expected_dtypes = (
        torch.int8,
        torch.float32,
        torch.int8,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int32,
    )
    if any(t.dtype != dtype for t, dtype in zip(tensors, expected_dtypes)):
        raise NotImplementedError(
            "requires INT8 A/B, FP32 scales, BF16 C, INT32 indices"
        )
    if any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if not all(t.is_contiguous() for t in tensors):
        raise NotImplementedError("only contiguous tensors are supported")
    expected_shapes = ((M, 1), (E, N, 1), (M, N), (M,), (E,))
    if weight_k != K or any(
        tuple(t.shape) != shape
        for t, shape in zip((As, Bs, C, m_indices, m_rows), expected_shapes)
    ):
        raise ValueError("incompatible scales, output or expert metadata shapes")
    if K > _MAX_K:
        raise ValueError(f"requires K <= {_MAX_K}")
    if M == 0 or N == 0 or E == 0:
        return
    if K == 0:
        # A zero inner dimension makes every partial product zero, so the result is the
        # scaled zero. Take the direct path: the loop would run zero times and leaving
        # an unbroken tile pipeline on an empty K measured as garbage on GCU400.
        C.zero_()
        return

    max_tiles = triton.cdiv(M, _BLOCK_M) + E
    tile_expert = torch.empty(max_tiles, dtype=torch.int32, device=A.device)
    tile_row_start = torch.empty(max_tiles, dtype=torch.int32, device=A.device)
    tile_row_count = torch.empty(max_tiles, dtype=torch.int32, device=A.device)
    num_tiles = torch.empty(1, dtype=torch.int32, device=A.device)

    launch_grid = (max_tiles, triton.cdiv(N, _BLOCK_N))
    with torch_device_fn.device(A.device):
        grouped_gemm_metadata_kernel[(1,)](
            m_rows,
            tile_expert,
            tile_row_start,
            tile_row_count,
            num_tiles,
            E,
            BLOCK_M=_BLOCK_M,
            MAX_TILES=max_tiles,
        )
        grouped_gemm_tiled_kernel[launch_grid](
            A,
            As,
            B,
            Bs,
            C,
            tile_expert,
            tile_row_start,
            tile_row_count,
            num_tiles,
            N,
            K,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=4,
            num_stages=3,
        )


__all__ = ["grouped_gemm_w8a8"]
