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

"""Ascend TensorView override for index_put / index_put_.

Only the narrow case of a single dim-0 integer index into a 2-D (or
row-collapsible) contiguous tensor is handled here. ``torch.index_put_``
accepts far more than that -- bool masks, several index tensors, broadcasting,
None placeholders -- and all of it routes to the generic implementation in
``flag_gems.ops.index_put``.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_put import index_put as _index_put_generic
from flag_gems.ops.index_put import index_put_ as _index_put__generic
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_rows_for_ub,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store

TV_BLOCK_T = 32

if HAS_TENSORVIEW:

    @triton.jit
    def index_put_tv_kernel(
        dst,
        values,
        index,
        num_indices,
        num_rows,
        WIDTH: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(
            tl.int32
        )

        values_view = tle_exp.make_partition_view(
            values,
            shape=[num_indices, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_T, BLOCK_WIDTH],
        )
        tile_values = tile_load(
            values_view, index=(tile, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )

        dst_view = tle_exp.make_gather_scatter_view(
            dst,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_T, BLOCK_WIDTH],
            sparse_dim=[0],
        )
        tile_store(
            dst_view,
            value=tile_values,
            index=(rows, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    @triton.jit
    def index_put_accumulate_tv_kernel(
        dst,
        values,
        index,
        num_indices,
        num_rows,
        WIDTH: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(
            tl.int32
        )

        values_view = tle_exp.make_partition_view(
            values,
            shape=[num_indices, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_T, BLOCK_WIDTH],
        )
        tile_values = tile_load(
            values_view, index=(tile, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )

        dst_view = tle_exp.make_gather_scatter_view(
            dst,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_T, BLOCK_WIDTH],
            sparse_dim=[0],
        )
        current = tile_load(
            dst_view, index=(rows, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )
        tile_store(
            dst_view,
            value=current + tile_values,
            index=(rows, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _launch_index_put_tv(out, index, values, accumulate, block_t=None):
        num_rows = out.shape[0]
        # Trailing dimensions collapse into one contiguous row.
        width = math.prod(out.shape[1:]) if out.dim() > 1 else 1
        if block_t is None:
            # Accumulating also keeps the destination tile live.
            block_t = tile_rows_for_ub(
                triton.next_power_of_2(width) * out.element_size(),
                2 if accumulate else 1,
                TV_BLOCK_T,
            )
        num_indices = index.numel()
        values = torch.broadcast_to(
            values, (num_indices, *out.shape[1:])
        ).contiguous()
        grid = (triton.cdiv(num_indices, block_t),)
        kernel = index_put_accumulate_tv_kernel if accumulate else index_put_tv_kernel
        with torch_device_fn.device(out.device):
            kernel[grid](
                out,
                values,
                index,
                num_indices,
                num_rows,
                WIDTH=width,
                BLOCK_T=block_t,
                BLOCK_WIDTH=triton.next_power_of_2(width),
            )
        return out

    def _index_put_tensorview(inp, indices, values, accumulate=False):
        logger.debug("GEMS_ASCEND INDEX_PUT(TENSORVIEW)")
        out = inp.contiguous().clone()
        return _launch_index_put_tv(
            out, indices[0].contiguous(), values, accumulate
        )

    def _index_put__tensorview(inp, indices, values, accumulate=False):
        logger.debug("GEMS_ASCEND INDEX_PUT_(TENSORVIEW)")
        index = indices[0].contiguous()
        if inp.is_contiguous():
            return _launch_index_put_tv(inp, index, values, accumulate)
        work = inp.contiguous()
        _launch_index_put_tv(work, index, values, accumulate)
        inp.copy_(work)
        return inp

    def _tv_eligible(inp, indices, values, accumulate):
        indices = list(indices)
        if len(indices) != 1:
            return False
        index = indices[0]
        if index is None or not torch.is_tensor(index):
            return False
        # Bool/byte masks expand into a different number of indices; leave the
        # whole preprocessing pipeline to the generic path.
        if index.dtype not in (torch.int32, torch.int64):
            return False
        if index.ndim != 1 or index.numel() == 0:
            return False
        if inp.ndim < 2 or inp.numel() == 0:
            return False
        if not torch.is_tensor(values) or values.dtype != inp.dtype:
            return False
        # The row width must match: values broadcast across rows only.
        row_shape = tuple(inp.shape[1:])
        if tuple(values.shape[-len(row_shape) :]) != row_shape:
            return False
        if values.ndim > 1 and values.shape[0] not in (1, index.numel()):
            return False
        if accumulate:
            # Concurrent tiles read the destination row before either writes it,
            # so a repeated index would drop one of the accumulates.
            if torch.unique(index).numel() != index.numel():
                return False
        return True

    def index_put(inp, indices, values, accumulate=False):
        if not _tv_eligible(inp, indices, values, accumulate):
            return _index_put_generic(inp, indices, values, accumulate)
        return tensorview_call(
            "INDEX_PUT",
            _index_put_tensorview,
            _index_put_generic,
            inp,
            indices,
            values,
            accumulate,
        )

    def index_put_(inp, indices, values, accumulate=False):
        if not _tv_eligible(inp, indices, values, accumulate):
            return _index_put__generic(inp, indices, values, accumulate)
        return tensorview_call(
            "INDEX_PUT_",
            _index_put__tensorview,
            _index_put__generic,
            inp,
            indices,
            values,
            accumulate,
        )

else:
    index_put = _index_put_generic
    index_put_ = _index_put__generic
