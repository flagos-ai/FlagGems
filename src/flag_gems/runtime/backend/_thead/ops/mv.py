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

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .gemm_utils import _PPU_SMS, HAS_PPU_TLE
from .mm import _mv, _ppu_device_context


@libentry()
@triton.jit
def mv_column_major_kernel(
    A,
    X,
    Out,
    M,
    K,
    lda,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    tiles = tl.cdiv(K, BLOCK_K)
    if SPLIT_K > 1:
        split = tl.program_id(1)
        first_tile = split * tiles // SPLIT_K
        last_tile = (split + 1) * tiles // SPLIT_K
    else:
        split = 0
        first_tile = 0
        last_tile = tiles
    acc = tl.zeros((BLOCK_K, BLOCK_M), tl.float32)
    for tile in range(first_tile, last_tile):
        cols = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + cols[:, None].to(tl.int64) * lda + rows[None, :],
            (cols[:, None] < K) & (rows[None, :] < M),
            0,
        ).to(tl.float32)
        x = tl.load(X + cols, cols < K, 0).to(tl.float32)
        acc += a * x[:, None]
    tl.store(Out + split * M + rows, tl.sum(acc, axis=0), rows < M)


@libentry()
@triton.jit
def mv_column_major_reduce_kernel(
    Partial, Out, M, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    splits = tl.arange(0, SPLIT_K)
    acc = tl.load(Partial + splits[:, None] * M + rows[None, :], rows[None, :] < M, 0)
    tl.store(Out + rows, tl.sum(acc, axis=0), rows < M)


def _column_major_launch(rows, k, lda):
    aligned_columns = lda % 16 == 0
    if triton.cdiv(rows, 64) >= 2 * _PPU_SMS:
        return (64 if aligned_columns and k > 1024 else 32), 64, 1
    # A single row per program avoids a second launch for light reductions.
    if k <= 2048 and rows <= 8 * _PPU_SMS and rows * k <= 8 * _PPU_SMS * 1024:
        return 1, 128, 1
    if k <= 1024:
        return (4 if rows <= 32 * _PPU_SMS else 32), 64, 1

    # A narrower tile reduces wasted work on unaligned columns in small
    # output grids. Larger grids benefit from more contiguous rows per tile.
    block_m, block_k = (32, 64)
    if not aligned_columns and rows <= 32 * _PPU_SMS:
        block_m, block_k = (16, 128)
    row_tiles = triton.cdiv(rows, block_m)
    k_tiles = triton.cdiv(k, block_k)
    target_splits = triton.next_power_of_2(triton.cdiv(8 * _PPU_SMS, row_tiles))
    # Keep at least two K tiles per split on average. Cap workspace fan-out
    # once the row grid exposes several independent tiles.
    max_splits = 1 << max(0, (k_tiles // 2).bit_length() - 1)
    split_limit = 32 if row_tiles < _PPU_SMS // 4 else 16
    return block_m, block_k, min(target_splits, max_splits, split_limit)


def mv(inp: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if not (
        HAS_PPU_TLE
        and inp.ndim == 2
        and inp.stride(0) == 1
        and inp.shape[0] > 1
        # Short reductions do not amortize the column-major launch geometry.
        and inp.shape[1] >= 128
        and inp.stride(1) >= inp.shape[0]
        and vec.ndim == 1
        and inp.shape[1] == vec.shape[0]
        and inp.dtype in (torch.float16, torch.bfloat16)
        and vec.dtype == inp.dtype
        and vec.device == inp.device
        and vec.is_contiguous()
    ):
        return _mv(inp, vec)

    rows, k = inp.shape
    lda = inp.stride(1)
    block_m, block_k, split_k = _column_major_launch(rows, k, lda)
    out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
    partial = (
        torch.empty((split_k, rows), dtype=torch.float32, device=inp.device)
        if split_k > 1
        else out
    )
    with _ppu_device_context(inp):
        grid = (triton.cdiv(rows, block_m), split_k)
        mv_column_major_kernel[grid](
            inp,
            vec,
            partial,
            rows,
            k,
            lda,
            block_m,
            block_k,
            split_k,
            num_warps=4,
        )
        if split_k > 1:
            mv_column_major_reduce_kernel[(triton.cdiv(rows, 32),)](
                partial, out, rows, split_k, 32, num_warps=1
            )
    return out


__all__ = ["mv"]
