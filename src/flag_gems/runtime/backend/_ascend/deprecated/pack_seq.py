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

"""Ascend TensorView override for pack_seq_triton.

Scatter a ragged ``[N, D]`` tensor into a padded ``[B, Lmax, D]`` tensor.
Sequence ``b`` owns the ``lengths[b]`` rows that follow the rows of sequences
``0..b-1``. Both sides move as whole rows, so a StridedView on each end turns
the generic kernel's masked element-wise copy into a block DMA.

The load tile always starts inside the input: a tile that begins past the end of
the sequence is moved back to the sequence start, and the rows beyond the
sequence length are replaced by the pad value after the load. This keeps every
load on the whole-tile DMA path instead of the element-by-element fill that a
tile with no in-bounds row would take.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.deprecated.pack_seq import pack_seq_triton as _pack_seq_generic
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store

# Rows per tile, before the UB budget and the padded length cap it.
TV_BLOCK_T = 128

if HAS_TENSORVIEW:

    @triton.jit
    def pack_seq_tv_kernel(
        out,
        x,
        lengths,
        num_rows,
        max_len,
        PAD_VALUE: tl.constexpr,
        FEATURES: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        seq = tl.program_id(0)
        tile = tl.program_id(1)

        start = 0
        for i in range(seq):
            start += tl.load(lengths + i)
        length = tl.load(lengths + seq)

        x_view = tle_exp.make_strided_view(
            x,
            shape=[num_rows, FEATURES],
            strides=[FEATURES, 1],
            tile=[BLOCK_T, BLOCK_D],
            traversal_strides=[1, BLOCK_D],
        )
        # A tile that starts past the end of the sequence would have no
        # in-bounds row, so read from the sequence start instead and let the
        # pad below overwrite everything.
        first = tl.where(tile * BLOCK_T < length, tile * BLOCK_T, 0)
        rows = tile_load(
            x_view,
            index=(start + first, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        )

        valid = (tile * BLOCK_T + tl.arange(0, BLOCK_T)) < length
        pad = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, rows.dtype)
        rows = tl.where(valid[:, None], rows, pad)

        # The view ends at the last row of this sequence's slot, so a tail tile
        # never spills into the next sequence.
        out_view = tle_exp.make_strided_view(
            out,
            shape=[(seq + 1) * max_len, FEATURES],
            strides=[FEATURES, 1],
            tile=[BLOCK_T, BLOCK_D],
            traversal_strides=[1, BLOCK_D],
        )
        tile_store(
            out_view,
            value=rows,
            index=(seq * max_len + tile * BLOCK_T, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _tv_rows_that_fit(features, element_size):
        """Rows per tile UB can hold. One payload tile plus the padded copy of
        it are live, so budget for two."""
        return tile_rows_for_ub(
            triton.next_power_of_2(features) * element_size, 2, TV_BLOCK_T
        )

    def _tv_block_t(features, element_size, max_len):
        rows = max(_tv_rows_that_fit(features, element_size), MIN_TILE_ROWS)
        return min(rows, max(triton.next_power_of_2(max_len), MIN_TILE_ROWS))

    def _pack_seq_tensorview(
        x, lengths, pad_value=-float("inf"), _block_t=None, _block_d=None
    ):
        """``_block_t`` / ``_block_d`` are the generic path's tile sizes. The
        tile here is sized from the UB budget, so they are accepted and
        ignored, which keeps one argument list for both paths."""
        logger.debug("GEMS_ASCEND PACK_SEQ_TRITON(TENSORVIEW)")
        is_uint8 = x.dtype == torch.uint8
        original_shape = x.shape
        x = x.reshape(original_shape[0], -1).contiguous()
        num_rows, features = x.shape
        num_seqs = lengths.numel()
        max_len = int(lengths.max().item())
        lengths = lengths.to(torch.int32).contiguous()

        block_t = _tv_block_t(features, x.element_size(), max_len)
        out = torch.empty((num_seqs, max_len, features), dtype=x.dtype, device=x.device)
        grid = (num_seqs, triton.cdiv(max_len, block_t))
        with torch_device_fn.device(x.device):
            pack_seq_tv_kernel[grid](
                out,
                x,
                lengths,
                num_rows,
                max_len,
                PAD_VALUE=int(pad_value) if is_uint8 else float(pad_value),
                FEATURES=features,
                BLOCK_T=block_t,
                BLOCK_D=triton.next_power_of_2(features),
            )
        return out.reshape((num_seqs, max_len) + tuple(original_shape[1:]))

    def _tv_eligible(x, lengths, pad_value):
        if x.numel() == 0 or lengths.numel() == 0:
            return False
        if x.device != lengths.device:
            return False
        # lengths is read on device by the kernel and its max sizes the output.
        if lengths.device.type == "cpu":
            return False
        if x.dim() < 2:
            return False
        features = x.numel() // x.shape[0]
        if features == 0:
            return False
        if x.dtype == torch.uint8 and not (
            isinstance(pad_value, int) and 0 <= pad_value <= 255
        ):
            return False
        # The prefix scan over lengths runs per program, so a long sequence list
        # costs more in the scan than the DMA saves.
        if lengths.numel() > 1024:
            return False
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return False
        return _tv_rows_that_fit(features, x.element_size()) >= MIN_TILE_ROWS

    def pack_seq_triton(
        x: torch.Tensor,
        lengths: torch.Tensor,
        pad_value: float | int = -float("inf"),
        block_t: int = 64,
        block_d: int = 64,
    ) -> torch.Tensor:
        if not _tv_eligible(x, lengths, pad_value):
            return _pack_seq_generic(x, lengths, pad_value, block_t, block_d)
        return tensorview_call(
            "PACK_SEQ_TRITON",
            _pack_seq_tensorview,
            _pack_seq_generic,
            x,
            lengths,
            pad_value,
            # The fast path derives its own tile height, and the generic
            # fallback takes these as its block_t / block_d.
            block_t,
            block_d,
        )

else:
    pack_seq_triton = _pack_seq_generic
