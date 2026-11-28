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

from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_fits_ub,
    tile_rows_for_ub,
)
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store


@libentry()
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["index_select"])
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = ext.program_id(axis=0)
    pid_y = ext.program_id(axis=1)
    num_pid_x = ext.num_programs(axis=0)
    loop_count = tl.cdiv(M, num_pid_x)
    for loop in range(0, loop_count):
        rows_offsets = (pid_x * loop_count + loop) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        rows_mask = rows_offsets < M
        cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

        out_mask = rows_mask and (cols_offsets < index_len)

        indices = tl.load(
            index + cols_offsets, mask=(cols_offsets < index_len), other=0
        )
        inp_off = rows_offsets * N + indices[None, :]
        out_off = rows_offsets * index_len + cols_offsets[None, :]

        selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
        tl.store(out + out_off, selected, mask=out_mask)


def _index_select_generic(inp, dim, index):
    logger.debug("GEMS_ASCEND INDEX_SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    # NOTE: there used to be an
    #   assert ((i >= 0 and i < inp.size(dim)) for i in index)
    # here. The generator expression is always truthy, so it never validated
    # anything, but creating it still evaluates `iter(index)` eagerly, and
    # `Tensor.__iter__` unbinds the whole index on the host (~4096
    # `torch.as_strided` calls for a 4096-element index, ~11 ms per call).
    # Removed for #9127; range checking would need a device->host reduction.

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()

    # `index_select_kernel` tiles the *selected* axis (BLOCK_N spans `index`),
    # so with dim=0 on a wide contiguous tensor -- e.g. the (vocab, hidden)
    # embedding weight that vLLM looks up -- the lanes of a tile read one
    # element from a different row each and the loads are completely scattered.
    # Measured on Ascend 910C for a (32000, 4096) weight this costs ~37 ms,
    # against ~0.04 ms for the native row gather and ~0.3 ms for the adjacent
    # index kernel that copies contiguous row tiles (#9127). Delegate to it.
    if dim == 0 and inp.is_contiguous():
        from .index import index as adjacent_index

        return adjacent_index(inp, [index])

    # with dim_compress
    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    def grid(meta):
        dim0 = triton.cdiv(M, meta["BLOCK_M"])
        dim1 = triton.cdiv(index_len, meta["BLOCK_N"]) if index_len > 0 else 1
        while dim0 * dim1 >= 65536:
            dim0 = triton.cdiv(dim0, 2)
        return (
            dim0,
            dim1,
        )

    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order).contiguous()
    else:
        return out


# ---------------------------------------------------------------------------
# TensorView fast path
#
# Selecting along dim 0 keeps every selected row contiguous, so an index names a
# whole row. That is exactly the shape a GatherScatterView handles: the sparse
# dimension takes the index tensor and the payload (the row) transfers as one
# block DMA, instead of the generic kernel's masked element-wise gather. The
# output rows are consecutive, so they go out through a PartitionView.
#
# Only dim 0 qualifies. Any other dim would need the input transposed first,
# which costs more than the DMA saves.
# ---------------------------------------------------------------------------
BLOCK_M = 32

if HAS_TENSORVIEW:

    @triton.jit
    def index_select_tv_kernel(
        out,
        inp,
        index,
        num_indices,
        num_rows,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        # The tail tile reads -1 for the slots past the index. Bounds checking
        # makes the gather read those rows as padding rather than from outside
        # the input, and the output PartitionView clips them on store because
        # they lie past num_indices, so neither side needs a mask.
        rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(
            tl.int32
        )

        inp_view = tle_exp.make_gather_scatter_view(
            inp,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
            sparse_dim=[0],
            bounds_check=True,
        )
        # Reference UB through the module rather than capturing it as a global:
        # the address-space singleton is rebuilt on reimport, and Triton rejects
        # a cached kernel whose captured globals changed identity.
        selected = tile_load(
            inp_view, index=(rows, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )

        out_view = tle_exp.make_partition_view(
            out,
            shape=[num_indices, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_M, BLOCK_WIDTH],
        )
        tile_store(
            out_view,
            value=selected,
            index=(tile, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _index_select_tensorview(inp, dim, index, block_m=None):
        logger.debug("GEMS_ASCEND INDEX_SELECT(TENSORVIEW)")
        if index.ndim == 0:
            index = index.unsqueeze(0)
        inp = inp.contiguous()
        num_rows = inp.shape[0]
        # Trailing dimensions collapse into one contiguous row.
        width = math.prod(inp.shape[1:]) if inp.dim() > 1 else 1
        if block_m is None:
            # One gathered tile is live.
            block_m = tile_rows_for_ub(
                triton.next_power_of_2(width) * inp.element_size(), 1, BLOCK_M
            )
        num_indices = index.numel()
        out = torch.empty(
            (num_indices, *inp.shape[1:]), dtype=inp.dtype, device=inp.device
        )
        grid = (triton.cdiv(num_indices, block_m),)
        index_select_tv_kernel[grid](
            out,
            inp,
            index,
            num_indices,
            num_rows,
            WIDTH=width,
            BLOCK_M=block_m,
            BLOCK_WIDTH=triton.next_power_of_2(width),
        )
        return out

    def _tensorview_eligible(inp, dim, index):
        if not inp.is_contiguous() and inp.dim() > 1:
            # The fast path calls .contiguous(), but for a non-contiguous input
            # that copy can cost more than the DMA saves.
            return False
        if not (
            dim % max(inp.ndim, 1) == 0
            and index.ndim <= 1
            and index.numel() > 0
            and inp.numel() > 0
        ):
            return False
        # The trailing dimensions collapse into one row, so they can make a row
        # too wide to stage: (200, 40999, 3) is a 122997-element row, which
        # rounds up to a 131072-wide tile and overflows UB. Mirror the launch's
        # budget here, or that shape reaches the kernel and fails to compile.
        width = math.prod(inp.shape[1:]) if inp.dim() > 1 else 1
        return tile_fits_ub(triton.next_power_of_2(width) * inp.element_size(), 1)

    def index_select(inp, dim, index):
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        assert index.ndim <= 1, "Index should have dimension 1 or 0"
        if not _tensorview_eligible(inp, dim, index):
            return _index_select_generic(inp, dim, index)
        return tensorview_call(
            "INDEX_SELECT",
            _index_select_tensorview,
            _index_select_generic,
            inp,
            dim,
            index,
        )

else:
    index_select = _index_select_generic
