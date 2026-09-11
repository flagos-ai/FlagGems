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


def _pick_kron_tiles(M2, N2, single_batch):
    # Only used when B's dims are exact tile multiples (NEED_MASK=False).
    BM = min(triton.next_power_of_2(M2), _KRON_MAX_BLOCK_2D)
    BN = min(triton.next_power_of_2(N2), _KRON_MAX_BLOCK_2D)
    return BM, BN, 1


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
        use_v3 = M2 * N2 >= _KRON_V3_MIN_B
        if use_v3:
            BM, BN, BLOCK_A = _pick_kron_tiles(M2, N2, single_batch)
            use_v3 = (M2 % BM == 0) and (N2 % BN == 0)
        if use_v3:
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
