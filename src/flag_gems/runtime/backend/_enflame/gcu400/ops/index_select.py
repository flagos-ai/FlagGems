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

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

# GCU400 hard limit: grid.y <= 255 (same GCU_MAX_GRID_Y as addmm/_addmm_activation).
# When index_len / inner exceeds 255*BLOCK, the cdiv'd y dimension raises OutOfResources,
# so both kernels use a grid-stride loop over all y blocks (precedent: cummin dual-axis striding).
GCU_MAX_GRID_Y = 63


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    num_prog_x = tl.num_programs(axis=0)
    num_prog_y = tl.num_programs(axis=1)

    # y-block grid-stride is the outer loop: cols is computed once, and the k loop body is
    # instruction-for-instruction identical to the unclamped version (common case has 1 y block per program, zero extra
    # cost).
    for y_block in range(pid_y, tl.cdiv(index_len, BLOCK_N), num_prog_y):
        cols_offsets = y_block * BLOCK_N + tl.arange(0, BLOCK_N)
        cols_mask = cols_offsets < index_len

        for k in range(0, tl.cdiv(M, num_prog_x * BLOCK_M)):
            row_block_id = k * num_prog_x + pid_x

            rows_offsets = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
            rows_mask = rows_offsets < M

            out_mask = rows_mask & cols_mask

            indices = tl.load(index + cols_offsets, mask=cols_mask, other=0)
            inp_off = rows_offsets * N + indices[None, :]
            out_off = rows_offsets * index_len + cols_offsets[None, :]

            selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
            tl.store(out + out_off, selected, mask=out_mask)


@libentry()
@triton.jit
def index_select_contiguous_kernel(
    inp,  # contiguous, logical shape (outer, N, inner)
    out,  # contiguous, logical shape (outer, index_len, inner)
    index,
    N,
    inner,
    index_len,
    total_p,
    BLOCK_I: tl.constexpr,
    GRID_X: tl.constexpr,
    OUTER_ONE: tl.constexpr,
):
    # Do contiguous access directly in the input's original layout, replacing dim_compress's whole-tensor move:
    # for a fixed (o, j), the inner dimension is one contiguous segment, handled by a single contiguous load/store.
    # grid.x is fixed at GRID_X with a stride loop covering all (o, j) pairs; full expansion
    # (grid.x=total_p) measured 0.66~0.92x worse (too many CTAs, scheduling overhead),
    # so a fixed grid better matches the GCU compute unit count.
    pid_x = tl.program_id(0)
    pid_i = tl.program_id(1)
    num_prog_i = tl.num_programs(1)
    inner_blocks = tl.cdiv(inner, BLOCK_I)
    # i-block grid-stride is the outer loop: i_offs/i_mask are computed once per block, and the p loop body is
    # instruction-for-instruction identical to the unclamped version (common case has 1 i block per program, zero extra
    # cost).
    for bi in range(pid_i, inner_blocks, num_prog_i):
        i_offs = bi * BLOCK_I + tl.arange(0, BLOCK_I)
        i_mask = i_offs < inner
        for p in range(pid_x, total_p, GRID_X):
            if OUTER_ONE:
                # With dim == 0, outer is always 1 and p is j, so integer division / modulo is avoided
                o = 0
                j = p
            else:
                o = p // index_len
                j = p % index_len
            idx = tl.load(index + j).to(tl.int32)
            in_off = (o * N + idx) * inner + i_offs
            out_off = p * inner + i_offs
            val = tl.load(inp + in_off, mask=i_mask, other=0)
            tl.store(out + out_off, val, mask=i_mask)


def make_contiguous_with_correct_stride(tensor):
    """Ensure the tensor has the correct stride after calling contiguous."""
    if tensor.numel() == 0:
        # For empty tensors, manually build a version with the correct stride
        new_tensor = tensor.flatten().view(tensor.shape)
        return new_tensor.contiguous()
    return tensor.contiguous()


def index_select(inp, dim, index):
    logger.debug("GEMS_ENFLAME INDEX_SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    # Do not write `assert (... for i in index)`: creating the generator already iterates index,
    # triggering Tensor.unbind, measured at 0.31ms per call (90%+ of end-to-end), and a generator is always
    # truthy, so the check is invalid anyway. The common implementation src/flag_gems/ops/index_select.py has
    # no such check either; index range is guaranteed by the caller.

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()
    N = inp_shape[dim]

    inner = 1
    for s in inp_shape[dim + 1 :]:
        inner *= s
    outer = inp.numel() // (N * inner)
    total_p = outer * index_len

    # When dim is not the last dimension and the contiguous block after dim is large enough, access the original layout
    # contiguously, avoiding dim_compress's permute().contiguous() whole-tensor copy and the output-side
    # permute().contiguous() copy (two O(numel) moves are the main bottleneck).
    if dim != inp.ndim - 1 and inner >= 16 and index_len > 0:
        out_shape = list(inp_shape)
        out_shape[dim] = index_len
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        # grid.x is fixed to match the GCU compute unit count: full expansion (grid.x=total_p) measured
        # 0.66x worse (too many CTAs, scheduling overhead); a fixed 48 with a stride loop is optimal.
        grid_x = min(total_p, MAX_GRID_DIM)
        block_i = min(triton.next_power_of_2(inner), 1024)
        index_select_contiguous_kernel[
            (grid_x, min(triton.cdiv(inner, block_i), GCU_MAX_GRID_Y))
        ](
            inp,
            out,
            index,
            N,
            inner,
            index_len,
            total_p,
            block_i,
            grid_x,
            outer == 1,
            # Large inner (a single contiguous transfer >= 8192 elements) has higher throughput with 4 warps,
            # while small/medium inner uses 2 warps for cheaper launch / occupancy (measured better than a single value
            # for all paths).
            num_warps=4 if inner >= 8192 else 2,
        )
        return out

    # with dim_compress (when dim == last it is an identity permute, no copy)
    inp = dim_compress(inp, dim)
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        min(triton.cdiv(M, meta["BLOCK_M"]), MAX_GRID_DIM),
        min(triton.cdiv(index_len, meta["BLOCK_N"]), GCU_MAX_GRID_Y),
    )
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        res = make_contiguous_with_correct_stride(out.permute(order))
        return res.reshape(res.shape)
    else:
        return out
