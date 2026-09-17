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

"""CDNA3 softmax: split the reduction across workgroups when rows are scarce.

The generic softmax_kernel_inner launches grid (M, 1, 1), one workgroup per row,
so parallelism is bounded by the row count rather than by the size of the
reduction. A 1-D input collapses to M=1, which leaves all but one of the 304 CUs
idle no matter how large N is, so a single long row runs at a small fraction of
what the part can sustain.

Here each row is split across NUM_BLOCKS workgroups. Pass 1 reduces one chunk
per workgroup into a partial (max, sum-of-exp) pair; pass 2 combines the partials
with the online-softmax identity and writes the normalized output. The grid
becomes (NUM_BLOCKS, M), so parallelism no longer depends on how many rows there
are. Traffic stays at two reads plus one write, the same as the generic loop
path, and the combine costs a few KB out of cache instead of a third launch.

Only the starved regime is taken over; softmax_out falls through to the generic
implementation everywhere else, which already fills the card once there are
enough rows to go around.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.ops.softmax import softmax_out as generic_softmax_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Warp count is not sensitive: sweeping it from 1 to 16 moves the median by less
# than the run-to-run spread, because these kernels are bound by how many
# workgroups are in flight rather than by how wide each one is.
#
# TILE_N is sensitive, and the useful range has a hard ceiling. Two fp32
# accumulators live across the reduction loop, so a tile costs 8 * TILE_N bytes of
# register state per workgroup; the top of the swept range spills and the kernel
# collapses. The rung just below that cliff was not enough faster than 2048 to pay
# for sitting next to it, so 2048 keeps two rungs of margin.
_TILE_N = 2048
_NUM_WARPS = 8

# Two blocks per CU: more than that measured no better across the shape set, and
# the grids it admits are the ones that stayed well behaved throughout the sweep.
_BLOCKS_PER_CU = 2

# Eligibility: a shortest reduction, a smallest useful number of chunks per row,
# and a row budget that depends on element width.
#
# The length limit is set by host cost, not by the GPU: two launches plus the
# partials allocation put a fixed floor on this path, so the split only pays once
# the generic kernel spends more than that floor on the reduction itself. It is
# measured end to end with the ops registered, because GPU-only timing does not
# charge the split for the floor and makes shorter lengths look like wins. The
# limit is the same for every element width.
#
# The chunk minimum is sharp: over both lengths and all three dtypes, every shape
# getting four or more chunks per row won and every shape that got two lost.
#
# The row budget is where element width matters. num_blocks is
# prev_pow2(_BLOCKS_PER_CU * CU / M), so a row count decides how many chunks it is
# cut into; a wide element measured poorly at exactly four, so its budget stops
# below every row count that would land there. Narrow elements are clean in that
# band and take the looser budget. half_to_float widens the store to fp32, so
# those calls take the wide budget too.
_MIN_SPLIT_N = 64 * 1024
_MIN_BLOCKS_PER_ROW = 4
_NARROW_MAX_ITEMSIZE = 2
_NARROW_MIN_CUS_PER_ROW = 2
_WIDE_MIN_CUS_PER_ROW = 4


@lru_cache(maxsize=8)
def _cu_count(device_index):
    return torch_device_fn.get_device_properties(device_index).multi_processor_count


def _prev_power_of_2(n):
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def _split_plan(m, n, itemsize, device_index):
    """Workgroups per row, or None to leave this shape to the generic kernel."""
    if itemsize <= _NARROW_MAX_ITEMSIZE:
        cus_per_row = _NARROW_MIN_CUS_PER_ROW
    else:
        cus_per_row = _WIDE_MIN_CUS_PER_ROW
    cu = _cu_count(device_index)
    if m > cu // cus_per_row or n < _MIN_SPLIT_N:
        return None
    target_blocks = _BLOCKS_PER_CU * cu
    num_blocks = _prev_power_of_2(max(1, target_blocks // m))
    # Cap so every workgroup still gets at least one whole tile.
    num_blocks = min(num_blocks, _prev_power_of_2(n // _TILE_N))
    return num_blocks if num_blocks >= _MIN_BLOCKS_PER_ROW else None


@libentry()
@triton.jit
def softmax_split_reduce_kernel(
    inp_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    N,
    NUM_BLOCKS,
    TILE_N: tl.constexpr,
):
    pid_b = ext.program_id(0)
    pid_m = ext.program_id(1)
    row = inp_ptr + pid_m * N

    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.zeros([TILE_N], dtype=tl.float32)

    stride = NUM_BLOCKS * TILE_N
    for off in range(pid_b * TILE_N, N, stride):
        n_offsets = off + tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(row + n_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, inp)
        # An all -inf window must keep z at 0 rather than accumulate exp(nan).
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
        m = m_new

    m_reduced = tl.max(m, 0)
    # Lanes still at -inf would give exp(-inf - -inf) = exp(nan) when the whole
    # chunk is -inf, so select the scale instead of computing it.
    scale = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_reduced))
    z_reduced = tl.sum(z * scale, 0)

    tl.store(partial_max_ptr + pid_m * NUM_BLOCKS + pid_b, m_reduced)
    tl.store(partial_sum_ptr + pid_m * NUM_BLOCKS + pid_b, z_reduced)


@libentry()
@triton.jit
def softmax_split_normalize_kernel(
    out_ptr,
    inp_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    N,
    NUM_BLOCKS,
    TILE_N: tl.constexpr,
    TILE_B: tl.constexpr,
):
    pid_b = ext.program_id(0)
    pid_m = ext.program_id(1)

    # Every workgroup redoes the combine over the NUM_BLOCKS partials. That is a
    # few KB already in cache, and it avoids both a third launch and passing the
    # row scalars through memory.
    b_offsets = tl.arange(0, TILE_B)
    b_mask = b_offsets < NUM_BLOCKS
    partial_offset = pid_m * NUM_BLOCKS + b_offsets
    partial_max = tl.load(
        partial_max_ptr + partial_offset, mask=b_mask, other=-float("inf")
    )
    partial_sum = tl.load(partial_sum_ptr + partial_offset, mask=b_mask, other=0.0)

    row_max = tl.max(partial_max, 0)
    scale = tl.where(partial_max == float("-inf"), 0.0, tl.exp(partial_max - row_max))
    row_sum = tl.sum(partial_sum * scale, 0)

    row_in = inp_ptr + pid_m * N
    row_out = out_ptr + pid_m * N

    stride = NUM_BLOCKS * TILE_N
    for off in range(pid_b * TILE_N, N, stride):
        n_offsets = off + tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(row_in + n_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        out = tl.exp(inp - row_max) / row_sum
        tl.store(row_out + n_offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def softmax_out(self, dim, half_to_float=False, *, out):
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    dim = dim % self.ndim
    N = self.shape[dim]
    M = 1
    for i in range(dim):
        M *= self.shape[i]
    K = self.numel() // M // N if self.numel() else 0

    # half_to_float widens the store to fp32, so the plan sees the wider element.
    itemsize = 4 if half_to_float else self.element_size()
    plan = (
        _split_plan(M, N, itemsize, self.device.index)
        if K == 1 and self.numel()
        else None
    )
    if plan is None:
        return generic_softmax_out(self, dim, half_to_float, out=out)

    logger.debug("GEMS_CDNA3 SOFTMAX_OUT SPLIT M=%d N=%d blocks=%d", M, N, plan)

    self = self.contiguous()
    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")

    # One allocation for both partials; the two rows stay contiguous.
    partials = torch.empty((2, M * plan), dtype=torch.float32, device=self.device)
    grid = (plan, M, 1)

    with torch_device_fn.device(self.device):
        softmax_split_reduce_kernel[grid](
            self,
            partials[0],
            partials[1],
            N,
            plan,
            TILE_N=_TILE_N,
            num_warps=_NUM_WARPS,
        )
        softmax_split_normalize_kernel[grid](
            out,
            self,
            partials[0],
            partials[1],
            N,
            plan,
            TILE_N=_TILE_N,
            TILE_B=triton.next_power_of_2(plan),
            num_warps=_NUM_WARPS,
        )
    return out


def softmax(self, dim, half_to_float=False):
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self, dtype=dtype)
    return softmax_out(self, dim, half_to_float, out=out)
