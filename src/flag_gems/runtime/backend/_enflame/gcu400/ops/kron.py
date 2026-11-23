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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


def prepare_tensor_for_kron(tensor_a, tensor_b):
    a_shape = list(tensor_a.shape)
    b_shape = list(tensor_b.shape)

    if tensor_a.numel() == 0 or tensor_b.numel() == 0:
        if not a_shape:
            a_shape = [0]
        if not b_shape:
            b_shape = [0]

        if len(a_shape) > len(b_shape):
            b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
        elif len(b_shape) > len(a_shape):
            a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

        out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
        return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape

    if len(a_shape) < 2:
        a_shape = [1] * (2 - len(a_shape)) + a_shape
    if len(b_shape) < 2:
        b_shape = [1] * (2 - len(b_shape)) + b_shape

    if len(a_shape) > len(b_shape):
        b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
    elif len(b_shape) > len(a_shape):
        a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

    out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
    return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kron"),
    key=["M", "N"],
)
@triton.jit
def kron_stride_constant_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    batch_size: tl.int64,
    M: tl.int64,
    N: tl.int64,
    M1: tl.int64,
    M2: tl.int64,
    N1: tl.int64,
    N2: tl.int64,
    a_stride_0: tl.constexpr,
    a_stride_1: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    c_stride_0: tl.constexpr,
    c_stride_1: tl.constexpr,
    a_batch_stride: tl.constexpr,
    b_batch_stride: tl.constexpr,
    c_batch_stride: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_I64: tl.constexpr,
):
    pid_base = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # block counts derived from the autotuned BLOCK_* (constexprs) — no need
    # to know the chosen config on the host.
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    num_blocks_per_batch = num_tiles_m * num_tiles_n
    total_tiles = batch_size * num_blocks_per_batch

    # grid-stride-loop over the flattened (batch, tile_m, tile_n) space so the
    # launch grid.x can be capped well below gcu400's hardware limit regardless
    # of how large batch_size * cdiv(M,BM) * cdiv(N,BN) grows.
    for pid in range(pid_base, total_tiles, num_programs):
        batch_id = pid // num_blocks_per_batch
        local_pid = pid % num_blocks_per_batch
        block_m = local_pid // num_tiles_n
        block_n = local_pid % num_tiles_n

        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # batch_id < batch_size always holds inside the grid-stride loop.  Do not
        # AND a scalar predicate into the mask: with wide BLOCK_N it breaks GCU
        # mask lowering (only the first lane stays True).
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        offset = batch_id * 2
        a_batch_idx = tl.load(map_ptr + offset)
        b_batch_idx = tl.load(map_ptr + offset + 1)

        a_row = offs_m[:, None] // M2
        a_col = offs_n[None, :] // N2
        b_row = offs_m[:, None] % M2
        b_col = offs_n[None, :] % N2

        a_idx = a_batch_idx * a_batch_stride + a_row * a_stride_0 + a_col * a_stride_1
        b_idx = b_batch_idx * b_batch_stride + b_row * b_stride_0 + b_col * b_stride_1

        # GCU400: masked gather faults / corrupts when the pointer offset is
        # out-of-bounds even where mask=False. Clamp indices to a legal address.
        a_idx = tl.where(mask, a_idx, 0)
        b_idx = tl.where(mask, b_idx, 0)

        a = tl.load(a_ptr + a_idx, mask=mask)
        b = tl.load(b_ptr + b_idx, mask=mask)
        c = a * b

        c_idx = (
            batch_id * c_batch_stride
            + offs_m[:, None] * c_stride_0
            + offs_n[None, :] * c_stride_1
        )
        c_idx = tl.where(mask, c_idx, 0)
        tl.store(c_ptr + c_idx, c, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kron"),
    key=["M", "N"],
)
@triton.jit
def kron_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    batch_size: tl.int32,
    M: tl.int32,
    N: tl.int32,
    M1: tl.int32,
    M2: tl.int32,
    N1: tl.int32,
    N2: tl.int32,
    a_stride_0: tl.int32,
    a_stride_1: tl.int32,
    b_stride_0: tl.int32,
    b_stride_1: tl.int32,
    c_stride_0: tl.int32,
    c_stride_1: tl.int32,
    a_batch_stride: tl.int32,
    b_batch_stride: tl.int32,
    c_batch_stride: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_I64: tl.constexpr,
):
    pid_base = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # block counts derived from the autotuned BLOCK_* (constexprs) — no need
    # to know the chosen config on the host.
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    num_blocks_per_batch = num_tiles_m * num_tiles_n
    total_tiles = batch_size * num_blocks_per_batch

    # grid-stride-loop over the flattened (batch, tile_m, tile_n) space so the
    # launch grid.x can be capped well below gcu400's hardware limit regardless
    # of how large batch_size * cdiv(M,BM) * cdiv(N,BN) grows.
    for pid in range(pid_base, total_tiles, num_programs):
        batch_id = pid // num_blocks_per_batch
        local_pid = pid % num_blocks_per_batch
        block_m = local_pid // num_tiles_n
        block_n = local_pid % num_tiles_n

        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # batch_id < batch_size always holds inside the grid-stride loop.  Do not
        # AND a scalar predicate into the mask: with wide BLOCK_N it breaks GCU
        # mask lowering (only the first lane stays True).
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        offset = batch_id * 2
        a_batch_idx = tl.load(map_ptr + offset)
        b_batch_idx = tl.load(map_ptr + offset + 1)

        a_row = offs_m[:, None] // M2
        a_col = offs_n[None, :] // N2
        b_row = offs_m[:, None] % M2
        b_col = offs_n[None, :] % N2

        a_idx = a_batch_idx * a_batch_stride + a_row * a_stride_0 + a_col * a_stride_1
        b_idx = b_batch_idx * b_batch_stride + b_row * b_stride_0 + b_col * b_stride_1

        # GCU400: masked gather faults / corrupts when the pointer offset is
        # out-of-bounds even where mask=False. Clamp indices to a legal address.
        a_idx = tl.where(mask, a_idx, 0)
        b_idx = tl.where(mask, b_idx, 0)

        a = tl.load(a_ptr + a_idx, mask=mask)
        b = tl.load(b_ptr + b_idx, mask=mask)
        c = a * b

        c_idx = (
            batch_id * c_batch_stride
            + offs_m[:, None] * c_stride_0
            + offs_n[None, :] * c_stride_1
        )
        c_idx = tl.where(mask, c_idx, 0)
        tl.store(c_ptr + c_idx, c, mask=mask)


@libentry()
@triton.jit
def kron_v3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    batch_size: tl.int32,
    M: tl.int32,
    N: tl.int32,
    M1: tl.int32,
    N1: tl.int32,
    M2: tl.int32,
    N2: tl.int32,
    a_batch_stride: tl.int32,
    b_batch_stride: tl.int32,
    c_batch_stride: tl.int64,
    a_stride_0: tl.int32,
    a_stride_1: tl.int32,
    c_stride_0: tl.int32,
    c_stride_1: tl.int32,
    BLOCK_A: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Fixed-grid fast path for kron: one CTA walks (a-tile, b-tile) pairs in a
    # grid-stride loop.  The host picks BLOCK_M/BLOCK_N/BLOCK_A so that the
    # working tile never exceeds the DSM budget, and B is always loaded as a
    # contiguous block (tile aligned on M2 x N2) instead of a per-element gather
    # (which is ~2 orders of magnitude slower on GCU400).
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_b_m = tl.cdiv(M2, BLOCK_M)
    num_b_n = tl.cdiv(N2, BLOCK_N)
    num_b_tiles = num_b_m * num_b_n
    num_a_tiles = tl.cdiv(batch_size * M1 * N1, BLOCK_A)
    total = num_a_tiles * num_b_tiles

    offs_a = tl.arange(0, BLOCK_A)
    offs_bm = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)

    for p in range(pid, total, num_programs):
        b_blk = p % num_b_tiles
        a_tile = p // num_b_tiles
        bm0 = (b_blk // num_b_n) * BLOCK_M
        bn0 = (b_blk % num_b_n) * BLOCK_N
        g_offs_bm = bm0 + offs_bm
        g_offs_bn = bn0 + offs_bn

        # NOTE: the mask must use the ABSOLUTE tile offsets (g_offs_*) -- using
        # the relative (offs_*) offsets faults/corrupts when B needs multiple
        # tiles and its dims are not powers of two.
        if NEED_MASK:
            b_mask = (g_offs_bm[:, None] < M2) & (g_offs_bn[None, :] < N2)
        else:
            b_mask = tl.full((BLOCK_M, BLOCK_N), 1, tl.int1)

        if SINGLE_BATCH:
            # ---- B block is shared by every A lane (batch size == 1) ----
            b_idx = g_offs_bm[:, None] * N2 + g_offs_bn[None, :]
            if NEED_MASK:
                b_idx = tl.where(b_mask, b_idx, 0)
                b_val = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
            else:
                b_val = tl.load(b_ptr + b_idx)

            if BLOCK_A == 1:
                # pure 2D: scalar A element x B tile
                a_lin = a_tile
                a_row = a_lin // N1
                a_col = a_lin % N1
                a_idx = a_row * a_stride_0 + a_col * a_stride_1
                a_val = tl.load(a_ptr + a_idx)

                c = a_val * b_val
                c_off = a_row * (M2 * N) + a_col * N2
                c_idx = (
                    c_off
                    + g_offs_bm[:, None] * c_stride_0
                    + g_offs_bn[None, :] * c_stride_1
                )
                if NEED_MASK:
                    c_idx = tl.where(b_mask, c_idx, 0)
                    tl.store(c_ptr + c_idx, c, mask=b_mask)
                else:
                    tl.store(c_ptr + c_idx, c)
            else:
                # 3D expansion: BLOCK_A A-elements x B tile (only for tiny B)
                a_lin = a_tile * BLOCK_A + offs_a
                a_ok = a_lin < batch_size * M1 * N1
                m1n1 = a_lin % (M1 * N1)
                a_row = m1n1 // N1
                a_col = m1n1 % N1
                a_idx = a_row * a_stride_0 + a_col * a_stride_1
                a_idx = tl.where(a_ok, a_idx, 0)
                a_val = tl.load(a_ptr + a_idx, mask=a_ok, other=0.0)

                c = a_val[:, None, None] * b_val[None, :, :]

                c_off = a_row[:, None, None] * (M2 * N) + a_col[:, None, None] * N2
                c_idx = (
                    c_off
                    + g_offs_bm[None, :, None] * c_stride_0
                    + g_offs_bn[None, None, :] * c_stride_1
                )
                c_idx = tl.where(b_mask[None, :, :], c_idx, 0)
                c_ok = a_ok[:, None, None] & b_mask[None, :, :]
                tl.store(c_ptr + c_idx, c, mask=c_ok)
        else:
            # ---- one A element per iteration; B depends on the batch ----
            a_lin = a_tile * BLOCK_A  # BLOCK_A == 1 here
            batch_id = a_lin // (M1 * N1)
            m1n1 = a_lin % (M1 * N1)
            a_row = m1n1 // N1
            a_col = m1n1 % N1

            off = batch_id * 2
            a_batch_idx = tl.load(map_ptr + off)
            b_batch_idx = tl.load(map_ptr + off + 1)

            a_idx = (
                a_batch_idx * a_batch_stride + a_row * a_stride_0 + a_col * a_stride_1
            )
            a_val = tl.load(a_ptr + a_idx)

            b_idx = (
                b_batch_idx * b_batch_stride
                + g_offs_bm[:, None] * N2
                + g_offs_bn[None, :]
            )
            if NEED_MASK:
                b_idx = tl.where(b_mask, b_idx, 0)
                b_val = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
            else:
                b_val = tl.load(b_ptr + b_idx)

            c = a_val * b_val

            c_off = batch_id * c_batch_stride + a_row * (M2 * N) + a_col * N2
            c_idx = (
                c_off
                + g_offs_bm[:, None] * c_stride_0
                + g_offs_bn[None, :] * c_stride_1
            )
            if NEED_MASK:
                c_idx = tl.where(b_mask, c_idx, 0)
                tl.store(c_ptr + c_idx, c, mask=b_mask)
            else:
                tl.store(c_ptr + c_idx, c)


@triton.jit
def kron_row_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    M1,
    N1,
    M2,
    N2,
    N,
    N1P: tl.constexpr,
    N2P: tl.constexpr,
):
    """Single-batch kron that writes one *contiguous* C row per step.

    C row ``m = m1 * M2 + m2`` holds ``[a[m1, n1] * b[m2, n2]]`` flattened, i.e.
    exactly the run ``C[m * N : (m + 1) * N]``. Building the row as one 2D outer
    product therefore turns the store into a single contiguous span instead of
    the strided ``(M2, N2)`` tiles the B-tile kernels emit. Measured on GCU400:
    2.6-3.9x faster than ``kron_v3_kernel`` and at/above parity with aten for
    32x32 ... 128x128 operands (the A/B rows stay in cache across the walk).
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_n1 = tl.arange(0, N1P)
    offs_n2 = tl.arange(0, N2P)
    for row in tl.range(pid, M, num_programs):
        m1 = row // M2
        m2 = row - m1 * M2
        a_val = tl.load(a_ptr + m1 * N1 + offs_n1)
        b_val = tl.load(b_ptr + m2 * N2 + offs_n2)
        c = a_val[:, None] * b_val[None, :]
        tl.store(c_ptr + row * N + offs_n1[:, None] * N2 + offs_n2[None, :], c)


@libentry()
@triton.jit
def kron_v4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    batch_size: tl.int32,
    M: tl.int32,
    N: tl.int32,
    M1: tl.int32,
    N1: tl.int32,
    M2: tl.int32,
    N2: tl.int32,
    a_batch_stride: tl.int32,
    b_batch_stride: tl.int32,
    c_batch_stride: tl.int64,
    a_stride_0: tl.int32,
    a_stride_1: tl.int32,
    BLOCK_N: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # C-contiguous-row kernel: every iteration covers BLOCK_N elements of one C
    # row p (= a_row*M2 + b_row).  C[p, o] = A[a_row, o//N2] * B[b_row, o%N2],
    # so A/B are gathered with a repeating pattern (L1 friendly) and C is
    # written fully contiguously -- the dominant traffic is coalesced writes.
    # Used for small B (M2*N2 < threshold) where v3's B-tile reuse is not
    # worth its scattered C stores.
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    total = batch_size * M * num_n_tiles

    offs = tl.arange(0, BLOCK_N)

    for p in range(pid, total, num_programs):
        row = p // num_n_tiles
        ntile = p % num_n_tiles
        batch_id = row // M
        local_row = row % M
        a_row = local_row // M2
        b_row = local_row % M2

        o = ntile * BLOCK_N + offs

        a_col = o // N2
        b_col = o % N2

        if SINGLE_BATCH:
            a_idx = a_row * a_stride_0 + a_col * a_stride_1
            b_idx = b_row * N2 + b_col
        else:
            off = batch_id * 2
            a_batch_idx = tl.load(map_ptr + off)
            b_batch_idx = tl.load(map_ptr + off + 1)
            a_idx = (
                a_batch_idx * a_batch_stride + a_row * a_stride_0 + a_col * a_stride_1
            )
            b_idx = b_batch_idx * b_batch_stride + b_row * N2 + b_col

        if NEED_MASK:
            mask = o < N
            a_idx = tl.where(mask, a_idx, 0)
            b_idx = tl.where(mask, b_idx, 0)
            a_val = tl.load(a_ptr + a_idx, mask=mask, other=0.0)
            b_val = tl.load(b_ptr + b_idx, mask=mask, other=0.0)
            c_val = a_val * b_val
            c_idx = batch_id * c_batch_stride + local_row * N + o
            c_idx = tl.where(mask, c_idx, 0)
            tl.store(c_ptr + c_idx, c_val, mask=mask)
        else:
            a_val = tl.load(a_ptr + a_idx)
            b_val = tl.load(b_ptr + b_idx)
            c_val = a_val * b_val
            c_idx = batch_id * c_batch_stride + local_row * N + o
            tl.store(c_ptr + c_idx, c_val)


# Host-side tile selection for kron_v3_kernel.
# - kron_v3_kernel is launched ONLY with NEED_MASK=False.  The caller routes
#   any shape whose B tile would need a 2D mask (M2/N2 not a tile multiple)
#   to kron_v4 instead: on GCU400 a masked 2D tile lowers ~120-270x slower
#   than the unmasked tile (measured), while v4's 1D mask is free.  BLOCK_M /
#   BLOCK_N therefore just cap B-tile size at 256x256 (2D DSM ~256 KB, safe).
# - 3D expansion (BLOCK_A > 1) is only worthwhile for tiny B tiles (<= 64
#   elements): it amortizes the grid-stride loop overhead when A is huge,
#   while staying within the 3D DSM budget (GCU pads 3D tiles to ~28 B/elem).
#   NOTE: the 3D BLOCK_A expansion measured SLOWER than BLOCK_A=1 on every
#   probe shape (GCU 3D-pad penalty), so BLOCK_A is always 1 now.
_KRON_MAX_BLOCK_2D = 256
# B element count below which kron_v4 (contiguous C rows) beats kron_v3
# (contiguous B tile).  Sweeps: v3 wins at 64x64 / 128x128 / 4x1024x1024
# (>= 4096), v4 wins at 16x16 / 24x36 / 4x4 (<= 864).
_KRON_V3_MIN_B = 2048
# v4 row-block: BLOCK_N = clamp(next_pow2(N), 256, this).  The 256 floor is
# important for small N: tiny BLOCK_N makes per-iteration overhead dominate
# (e.g. batched N=24 with BLOCK_N=32 is ~50x slower than BLOCK_N=256/1024).
_KRON_V4_MAX_BN = 1024
_KRON_V4_MIN_BN = 256
# Byte budget for the (N1 x N2) row tile of kron_row_kernel.  GCU400 needs about
# 4x the tile bytes of DSM, and the hardware limit is 917504 B, so 128 KB keeps
# a 4x margin.
_KRON_ROW_TILE_BYTES = 131072
# Programs for kron_row_kernel: 48 CTAs is one full GCU400 wave and measured
# best up to ~8K rows; wider krons (e.g. 128x128 operands, 16K rows) want 2-4
# waves so the grid-stride tail does not serialize.
_KRON_ROW_SMALL_GRID = 48
_KRON_ROW_LARGE_GRID = 192
_KRON_ROW_LARGE_ROWS = 8192


def _pick_kron_tiles(M2, N2, single_batch):
    # Only used when B's dims are exact tile multiples (NEED_MASK=False).
    BM = min(triton.next_power_of_2(M2), _KRON_MAX_BLOCK_2D)
    BN = min(triton.next_power_of_2(N2), _KRON_MAX_BLOCK_2D)
    return BM, BN, 1


# ---------------------------------------------------------------------------
# Outer-product path.
#
# Every kron output row r = i * M2 + k is
#
#     C[r, :] = repeat(A[i, j], N2) * tile(B[k, :], N1)
#
# i.e. the elementwise product of two vectors of length N = N1 * N2.  Both
# operands are tiny compared with C ((M1 + M2) * N elements against
# M1 * M2 * N), so they are expanded once into device buffers and the main
# kernel degenerates into a three-stream copy: two affine contiguous loads plus
# one affine contiguous store.
#
# Why bother: rebuilding the operands per output element (c // N2, c % N2)
# turns the loads into gathers.  Measured on GCU400 (kron 160x160, fp16):
# gather formulation 32.6 ms, per-row scalar form 30.3 ms (one dependent scalar
# load per 320 B store), this path 1.13 ms.  The expansion costs ~30 us because
# it only touches (M1 + M2) * N elements.
# ---------------------------------------------------------------------------
_KRON_OUTER_MIN_OPERAND = 8  # keeps the expansion below 25% of C's bytes
# Below this row length the expansion launch cost is not amortised (measured
# 64x64: row kernel 0.146 ms vs this path 0.180 ms; from 96x96 on it wins).
_KRON_OUTER_MIN_N = 8192
_KRON_OUTER_MAX_BLOCK = 16384
_KRON_OUTER_GRID = 48
_KRON_EXPAND_GRID = 48
_KRON_EXPAND_BLOCK_K = 16
_KRON_EXPAND_BLOCK_L = 256
_KRON_EXPAND_BLOCK_M = 32


@libentry()
@triton.jit
def kron_repeat_element_kernel(
    src_ptr,
    dst_ptr,
    M,
    K: tl.constexpr,
    K2: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """dst[m, k * K2 + l] = src[m, k]: repeat every source element K2 times."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_l = tl.arange(0, BLOCK_L)
    num_l_tiles = (K2 + BLOCK_L - 1) // BLOCK_L
    num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    total = M * num_k_tiles * num_l_tiles
    for t in tl.range(pid, total, num_programs):
        tile_l = t % num_l_tiles
        rest = t // num_l_tiles
        tile_k = rest % num_k_tiles
        row = rest // num_k_tiles
        k_idx = tile_k * BLOCK_K + offs_k
        l_idx = tile_l * BLOCK_L + offs_l
        k_ok = k_idx < K
        mask = k_ok[:, None] & (l_idx[None, :] < K2)
        value = tl.load(src_ptr + row * K + k_idx, mask=k_ok, other=0.0)
        out = tl.broadcast_to(value[:, None], (BLOCK_K, BLOCK_L))
        dst_idx = row * (K * K2) + k_idx[:, None] * K2 + l_idx[None, :]
        tl.store(dst_ptr + tl.where(mask, dst_idx, 0), out, mask=mask)


@libentry()
@triton.jit
def kron_repeat_tile_kernel(
    src_ptr,
    dst_ptr,
    M,
    K: tl.constexpr,
    K2T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """dst[m, t * K + l] = src[m, l]: tile every source row K2T times."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_l = tl.arange(0, BLOCK_L)
    num_l_tiles = (K + BLOCK_L - 1) // BLOCK_L
    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    total = num_m_tiles * num_l_tiles * K2T
    for t in tl.range(pid, total, num_programs):
        tile_t = t % K2T
        rest = t // K2T
        tile_l = rest % num_l_tiles
        tile_m = rest // num_l_tiles
        m_idx = tile_m * BLOCK_M + offs_m
        l_idx = tile_l * BLOCK_L + offs_l
        m_ok = m_idx < M
        mask = m_ok[:, None] & (l_idx[None, :] < K)
        value = tl.load(
            src_ptr + m_idx[:, None] * K + l_idx[None, :], mask=mask, other=0.0
        )
        dst_idx = m_idx[:, None] * (K * K2T) + tile_t * K + l_idx[None, :]
        tl.store(dst_ptr + tl.where(mask, dst_idx, 0), value, mask=mask)


@libentry()
@triton.jit
def kron_outer_kernel(
    a_rep_ptr,
    b_rep_ptr,
    c_ptr,
    num_rows,
    M2: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """C[r, :] = a_rep[r // M2, :] * b_rep[r % M2, :] with affine streaming."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    num_col_tiles: tl.constexpr = (N + BLOCK - 1) // BLOCK
    total = num_rows * num_col_tiles
    # Contiguous partition: every CTA streams one contiguous slice of C.
    chunk = (total + num_programs - 1) // num_programs
    start = pid * chunk
    end = min(start + chunk, total)
    for t in tl.range(start, end):
        col_tile = t % num_col_tiles
        row = t // num_col_tiles
        row_a = row // M2
        row_b = row - row_a * M2
        col0 = col_tile * BLOCK
        if NEED_MASK:
            mask = col0 + lanes < N
            a_val = tl.load(a_rep_ptr + row_a * N + col0 + lanes, mask=mask, other=0.0)
            b_val = tl.load(b_rep_ptr + row_b * N + col0 + lanes, mask=mask, other=0.0)
            tl.store(c_ptr + row * N + col0 + lanes, a_val * b_val, mask=mask)
        else:
            a_val = tl.load(a_rep_ptr + row_a * N + col0 + lanes)
            b_val = tl.load(b_rep_ptr + row_b * N + col0 + lanes)
            tl.store(c_ptr + row * N + col0 + lanes, a_val * b_val)


def _kron_outer_fits(M1, M2, M, N, single_batch):
    if not single_batch:
        return False
    if M1 < _KRON_OUTER_MIN_OPERAND or M2 < _KRON_OUTER_MIN_OPERAND:
        return False
    if N < _KRON_OUTER_MIN_N:
        return False
    # kernel offsets are int32
    return M * N < 2**31


def _launch_kron_outer(A2, B2, C_reshaped, M1, M2, N1, N2, M, N):
    expanded_a = torch.empty((M1, N), device=A2.device, dtype=A2.dtype)
    expanded_b = torch.empty((M2, N), device=B2.device, dtype=B2.dtype)
    kron_repeat_element_kernel[(_KRON_EXPAND_GRID,)](
        A2,
        expanded_a,
        M1,
        K=N1,
        K2=N2,
        BLOCK_K=_KRON_EXPAND_BLOCK_K,
        BLOCK_L=_KRON_EXPAND_BLOCK_L,
        num_warps=1,
    )
    kron_repeat_tile_kernel[(_KRON_EXPAND_GRID,)](
        B2,
        expanded_b,
        M2,
        K=N2,
        K2T=N1,
        BLOCK_M=_KRON_EXPAND_BLOCK_M,
        BLOCK_L=_KRON_EXPAND_BLOCK_L,
        num_warps=1,
    )
    block = min(triton.next_power_of_2(N), _KRON_OUTER_MAX_BLOCK)
    kron_outer_kernel[(_KRON_OUTER_GRID,)](
        expanded_a,
        expanded_b,
        C_reshaped,
        M,
        M2=M2,
        N=N,
        BLOCK=block,
        NEED_MASK=N % block != 0,
        num_warps=8,
    )


def kron(A, B):
    logger.debug("GEMS_ENFLAME KRON")
    if A.dim() == 0 and B.dim() == 0:
        return A * B

    if A.numel() == 0 or B.numel() == 0:
        A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
        output_dtype = torch.promote_types(A.dtype, B.dtype)
        return torch.empty(out_shape, device=A.device, dtype=output_dtype)

    if A.dim() == 0:
        return A.unsqueeze(0) * B
    if B.dim() == 0:
        return A * B.unsqueeze(0)

    A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]
    M, N = M1 * M2, N1 * N2

    batch_size = math.prod(out_shape[:-2]) if out_shape[:-2] else 1

    output_dtype = torch.promote_types(A.dtype, B.dtype)

    if output_dtype == torch.int64:
        output_dtype = torch.int32
    elif output_dtype == torch.uint64:
        output_dtype = torch.uint32

    C = torch.empty(out_shape, device=A.device, dtype=output_dtype)

    C_reshaped = C.view(-1, M, N)
    A_view = A_prepared.reshape(-1, M1, N1)
    B_view = B_prepared.reshape(-1, M2, N2)

    if not A_view.is_contiguous():
        A_view = A_view.contiguous()
    if not B_view.is_contiguous():
        B_view = B_view.contiguous()

    a_batch_stride = M1 * N1
    b_batch_stride = M2 * N2
    c_batch_stride = M * N
    # Pass strides as python ints so Triton can widen c_batch_stride to int64
    # when M*N exceeds int32 (large kron shapes).
    a_batch_stride = int(a_batch_stride)
    b_batch_stride = int(b_batch_stride)
    c_batch_stride = int(c_batch_stride)

    # NOTE: never build the batch map on device -- on GCU every small tensor op
    # is a separate kernel with a serialization sync, so both a Python loop of
    # scalar device writes (~1.4ms for 36 batches) AND ~20 elementwise ops
    # (~1.4ms) are unacceptable.  Build it in CPU then copy once.
    a_batch_dims = A_prepared.shape[:-2] or (1,)
    b_batch_dims = B_prepared.shape[:-2] or (1,)
    if batch_size > 1:
        out_batch_dims = tuple(a * b for a, b in zip(a_batch_dims, b_batch_dims))
        pairs = []
        for i in range(batch_size):
            remaining = i
            out_indices = []
            for dim_size in out_batch_dims[::-1]:
                out_indices.insert(0, remaining % dim_size)
                remaining //= dim_size
            a_mp = b_mp = 0
            for out_idx, (ad, bd) in zip(out_indices, zip(a_batch_dims, b_batch_dims)):
                a_mp = a_mp * ad + (out_idx // bd)
                b_mp = b_mp * bd + (out_idx % bd)
            pairs += [a_mp, b_mp]
        batch_indices = torch.tensor(pairs, device=A.device, dtype=torch.int32)
    else:
        batch_indices = torch.zeros(2, device=A.device, dtype=torch.int32)
    with torch_device_fn.device(A.device):
        single_batch = batch_size == 1
        # Pick the kernel by B size: v3 (contiguous B-tile load + scalar A) wins
        # when B is large enough to be reused as a whole tile; v4 (contiguous C
        # rows, repeating-pattern A/B gather) wins for small B where v3's per
        # A-scalar iterations and scattered C stores dominate.
        #
        # Mask rule: v3 is only used when its B tile needs NO mask (M2/N2 are
        # exact tile multiples).  A 2D masked tile is catastrophic on GCU400:
        # the masked-2D-tile lowering is ~120-270x slower than the same tile
        # without mask (measured: 96x96 B 73ms vs 0.27ms for the unmasked
        # 128x128 tile), independent of dtype/tile size.  v4's mask is 1D and
        # measured free, so any shape that would need a 2D mask routes to v4.
        # Row kernel: single batch, both operands power-of-two so the (N1, N2)
        # outer product covers one whole C row unmasked, tile inside the DSM
        # budget and the C offsets inside int32.  This is the cheapest layout
        # available: every store is one contiguous row of C.
        elem_bytes = C_reshaped.element_size()
        # Outer-product path: cheapest layout for wide krons (see the comment
        # block above the kernels).  Only for single-batch operands that are
        # large enough for the expansion to stay negligible.
        use_outer = _kron_outer_fits(M1, M2, M, N, single_batch)
        use_row = (
            single_batch
            and (N1 & (N1 - 1)) == 0
            and (N2 & (N2 - 1)) == 0
            and N1 * N2 * elem_bytes <= _KRON_ROW_TILE_BYTES
            and M * N < 2**31
        )
        use_v3 = False
        if not use_row:
            use_v3 = M2 * N2 >= _KRON_V3_MIN_B
            if use_v3:
                BM, BN, BLOCK_A = _pick_kron_tiles(M2, N2, single_batch)
                use_v3 = (M2 % BM == 0) and (N2 % BN == 0)

        if use_outer:
            _launch_kron_outer(
                A_view.reshape(M1, N1),
                B_view.reshape(M2, N2),
                C_reshaped,
                M1,
                M2,
                N1,
                N2,
                M,
                N,
            )
        elif use_row:
            num_rows = M
            num_programs = min(
                num_rows,
                _KRON_ROW_SMALL_GRID
                if num_rows <= _KRON_ROW_LARGE_ROWS
                else _KRON_ROW_LARGE_GRID,
            )
            kron_row_kernel[(num_programs,)](
                A_view,
                B_view,
                C_reshaped,
                M,
                M1,
                N1,
                M2,
                N2,
                N,
                N1P=N1,
                N2P=N2,
                num_warps=1,
            )
        elif use_v3:
            num_a_tiles = (batch_size * M1 * N1 + BLOCK_A - 1) // BLOCK_A
            num_b_tiles = (M2 + BM - 1) // BM * ((N2 + BN - 1) // BN)
            grid = (min(MAX_GRID_DIM, num_a_tiles * num_b_tiles),)
            kron_v3_kernel[grid](
                A_view,
                B_view,
                C_reshaped,
                batch_indices,
                batch_size,
                M,
                N,
                M1,
                N1,
                M2,
                N2,
                a_batch_stride,
                b_batch_stride,
                c_batch_stride,
                A_view.stride(1),
                A_view.stride(2),
                C_reshaped.stride(1),
                C_reshaped.stride(2),
                BLOCK_A=BLOCK_A,
                BLOCK_M=BM,
                BLOCK_N=BN,
                SINGLE_BATCH=single_batch,
                NEED_MASK=False,
            )
        else:
            BN = min(max(triton.next_power_of_2(N), _KRON_V4_MIN_BN), _KRON_V4_MAX_BN)
            NEED_MASK = N % BN != 0
            num_n_tiles = (N + BN - 1) // BN
            grid = (min(MAX_GRID_DIM, batch_size * M * num_n_tiles),)
            kron_v4_kernel[grid](
                A_view,
                B_view,
                C_reshaped,
                batch_indices,
                batch_size,
                M,
                N,
                M1,
                N1,
                M2,
                N2,
                a_batch_stride,
                b_batch_stride,
                c_batch_stride,
                A_view.stride(1),
                A_view.stride(2),
                BLOCK_N=BN,
                SINGLE_BATCH=single_batch,
                NEED_MASK=NEED_MASK,
            )

    if A.dim() <= 1 and B.dim() <= 1:
        return C.reshape(-1)

    return C
