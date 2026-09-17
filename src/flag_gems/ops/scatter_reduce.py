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

"""Triton implementation of torch.scatter_reduce for FlagGems.

Supports all reduce modes: sum, prod, mean, amax, amin.
Handles 1D-5D tensors with up to 5D coordinate decoding via padding.

Vendor compatibility:
  - NVIDIA: native atomic_max/min for amax/amin reduce
  - Iluvatar: CAS-based fallback for atomic_max/min (no native support)
  - Metax: larger BLOCK=256 for better occupancy

Performance notes:
  - Sum/mean use tl.atomic_add with relaxed semantics for throughput
  - Prod uses CAS loop with NaN detection guard (no tl.atomic_mul exists)
  - All offset arithmetic uses int64 to avoid overflow for N > 2^31
  - LOOP=4: each program processes LOOP*BLOCK elements to amortize launch overhead
  - 2D fast path: specialized kernels for 2D tensors avoid 5D coordinate decoding
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def heur_block(args):
    """Vendor-aware block size heuristic.

    Metax and Iluvatar GPUs benefit from larger blocks (256) for better
    occupancy. NVIDIA GPUs default to 128 which balances occupancy and
    register pressure.
    """
    if flag_gems.vendor_name in ["metax", "iluvatar"]:
        return 256
    return 128


def heur_loop(args):
    """Loop unrolling factor.

    Each program processes LOOP*BLOCK elements to amortize kernel launch
    overhead. LOOP=4 is optimal for Iluvatar BI-V150.
    """
    return 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad5(lst, fill):
    """Pad a list to exactly 5 elements from the left with `fill`.

    This enables uniform 5D coordinate decoding in kernels regardless
    of the actual tensor dimensionality (1D-5D). Shapes are padded with 1,
    strides with 0.
    """
    return [fill] * (5 - len(lst)) + lst if len(lst) < 5 else lst


def _needs_cas_fallback():
    """Check if the current vendor needs CAS-based fallback for atomic_max/min.

    Iluvatar GPUs lack native tl.atomic_max/min, so we fall back to a
    CAS (Compare-And-Swap) loop for amax/amin reduce modes.
    """
    return flag_gems.vendor_name in ["iluvatar"]


# ---------------------------------------------------------------------------
# 2D Fast Path Kernels with LOOP
# Specialized for 2D tensors to avoid 5D coordinate decoding overhead.
# Uses 1D grid with LOOP=4 to amortize kernel launch overhead.
# ---------------------------------------------------------------------------


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_sum_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)
        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_prod_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        # CAS loop for product
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur_val = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)
            new_val = tl.where(stop, cur_val, cur_val * src_val)
            is_nan = new_val != new_val
            new_val = tl.where(is_nan, src_val, new_val)
            cas_res = tl.atomic_cas(
                out_ptr + out_offsets, cur_val, new_val, sem="relaxed"
            )
            stop |= (cur_val == cas_res) | is_nan
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_mean_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    count_ptr,
    mask_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
        ones_f = tl.full((BLOCK,), 1.0, dtype=tl.float32)
        tl.atomic_add(count_ptr + out_offsets, ones_f, mask=mask, sem="relaxed")

        if USE_MASK:
            ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_amax_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    IS_AMAX: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CAS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur_val = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)
                if IS_AMAX:
                    new_val = tl.maximum(cur_val, src_val)
                else:
                    new_val = tl.minimum(cur_val, src_val)
                cas_res = tl.atomic_cas(
                    out_ptr + out_offsets, cur_val, new_val, sem="relaxed"
                )
                stop |= cur_val == cas_res
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if IS_AMAX:
                tl.atomic_max(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# Generic 5D Kernels with LOOP optimization
# For tensors with ndim != 2.
# ---------------------------------------------------------------------------


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_sum_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)
        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_prod_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        # CAS loop for product. NaN/Inf guard: if cur_val is NaN, mark as done
        # to prevent infinite spin (NaN != NaN causes CAS to always fail).
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur_val = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)
            new_val = tl.where(stop, cur_val, cur_val * src_val)
            # Detect NaN: if new_val != new_val (NaN check), use src_val directly
            is_nan = new_val != new_val
            new_val = tl.where(is_nan, src_val, new_val)
            cas_res = tl.atomic_cas(
                out_ptr + out_offsets, cur_val, new_val, sem="relaxed"
            )
            # Mark done if CAS succeeded OR if value is NaN (can't recover)
            stop |= (cur_val == cas_res) | is_nan
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_mean_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    count_ptr,
    mask_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
        ones_f = tl.full((BLOCK,), 1.0, dtype=tl.float32)
        tl.atomic_add(count_ptr + out_offsets, ones_f, mask=mask, sem="relaxed")

        if USE_MASK:
            ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_amax_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    IS_AMAX: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CAS: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur_val = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)
                if IS_AMAX:
                    new_val = tl.maximum(cur_val, src_val)
                else:
                    new_val = tl.minimum(cur_val, src_val)
                cas_res = tl.atomic_cas(
                    out_ptr + out_offsets, cur_val, new_val, sem="relaxed"
                )
                stop |= cur_val == cas_res
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if IS_AMAX:
                tl.atomic_max(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    """Triton-accelerated scatter_reduce operation.

    Scatters src values into the output tensor at positions determined by index,
    applying the specified reduction. Supports sum, prod, mean, amax, amin.

    Args:
        inp: Input tensor (1D-5D).
        dim: Dimension along which to scatter.
        index: Index tensor mapping source elements to output positions.
        src: Source tensor containing values to scatter.
        reduce: Reduction mode - "sum", "prod", "mean", "amax", or "amin".
        include_self: If True, include inp values in the reduction.

    Returns:
        Output tensor with same shape and dtype as inp.
    """
    logger.debug("GEMS SCATTER_REDUCE_TWO")

    assert reduce in (
        "sum",
        "prod",
        "mean",
        "amax",
        "amin",
    ), f"Unsupported reduce: {reduce}"
    assert inp.ndim <= 5, f"scatter_reduce supports up to 5D tensors, got {inp.ndim}D"

    dim = dim % inp.ndim
    padded_dim = dim + (5 - inp.ndim)

    out_stride_dim = inp.stride(dim)
    out_shape_dim = inp.size(dim)
    src_stride_dim = src.stride(dim)
    src_shape_dim = src.size(dim)
    N = index.numel()

    # Avoid double clone: merge contiguous + float32 cast
    inp_f32 = inp.to(torch.float32).contiguous()

    if include_self:
        out = inp_f32.clone()
    else:
        if reduce in ("sum", "mean"):
            out = torch.zeros_like(inp_f32)
        elif reduce == "prod":
            out = torch.ones_like(inp_f32)
        elif reduce == "amax":
            out = torch.full_like(inp_f32, float("-inf"))
        elif reduce == "amin":
            out = torch.full_like(inp_f32, float("inf"))

    if N == 0:
        return out.to(inp.dtype) if not include_self else inp_f32.to(inp.dtype)

    use_mask = not include_self
    if use_mask:
        reduced_mask = torch.zeros(out.shape, dtype=torch.int32, device=inp.device)

    if reduce == "mean":
        if include_self:
            count = torch.ones_like(out, dtype=torch.float32)
        else:
            count = torch.zeros_like(out, dtype=torch.float32)

    src = src.contiguous()
    index = index.contiguous()

    # Convert strides/shapes to int64 to avoid overflow in kernel arithmetic
    idx_shapes = [int(x) for x in _pad5(list(index.shape), 1)]
    src_strides_p = [int(x) for x in _pad5(list(src.stride()), 0)]
    idx_strides_p = [int(x) for x in _pad5(list(index.stride()), 0)]
    out_strides_p = [int(x) for x in _pad5(list(out.stride()), 0)]

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]),)

    dummy_mask = torch.empty(1, dtype=torch.int32, device=inp.device)
    mask_ptr = reduced_mask if use_mask else dummy_mask

    # Use 2D fast path for 2D tensors (most common case)
    use_2d = inp.ndim == 2

    # For 2D kernels, use raw dim (0 or 1) instead of padded_dim
    dim_2d = dim

    with torch_device_fn.device(inp.device):
        if reduce == "sum":
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_sum_2d_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    use_mask,
                )
            else:
                scatter_reduce_sum_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    use_mask,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )
        elif reduce == "prod":
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_prod_2d_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    use_mask,
                )
            else:
                scatter_reduce_prod_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    use_mask,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )
        elif reduce == "mean":
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_mean_2d_kernel[grid](
                    index,
                    src,
                    out,
                    count,
                    mask_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    use_mask,
                )
            else:
                scatter_reduce_mean_kernel[grid](
                    index,
                    src,
                    out,
                    count,
                    mask_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    use_mask,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )
            has_contributions = count > 0
            count = torch.clamp(count, min=1.0)
            out = out / count
            out = torch.where(has_contributions, out, inp_f32)
        elif reduce in ("amax", "amin"):
            use_cas = _needs_cas_fallback()
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_amax_2d_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    reduce == "amax",
                    use_mask,
                    use_cas,
                )
            else:
                scatter_reduce_amax_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    reduce == "amax",
                    use_mask,
                    use_cas,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )

    if use_mask and reduce != "mean":
        unreduced = reduced_mask == 0
        out = torch.where(unreduced, inp_f32, out)

    return out.to(inp.dtype)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    """In-place variant of scatter_reduce. Modifies inp in-place."""
    logger.debug("GEMS SCATTER_REDUCE_TWO_")

    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    """Out-variant of scatter_reduce. Writes result to out tensor if provided."""
    logger.debug("GEMS SCATTER_REDUCE_TWO_OUT")

    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    if out is not None:
        out.copy_(result)
        return out
    return result
