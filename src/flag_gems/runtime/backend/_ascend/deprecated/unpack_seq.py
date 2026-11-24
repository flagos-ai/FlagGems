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

"""Ascend TensorView override for unpack_seq_triton.

Gather a padded ``[B, Lmax, D]`` tensor back into a ragged ``[N, D]`` tensor.
Sequence ``b`` owns the ``lengths[b]`` rows that follow the rows of sequences
``0..b-1`` in the output. Both sides move as whole rows, so a StridedView on
each end turns the generic kernel's masked element-wise copy into a block DMA.

The output view ends at the last valid row of the sequence, so the rows of a
tile that lie past the sequence length are dropped by the store and never reach
the next sequence.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.deprecated.unpack_seq import unpack_seq_triton as _unpack_seq_generic
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
    def unpack_seq_tv_kernel(
        out,
        packed,
        lengths,
        max_len,
        FEATURES: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        seq = tl.program_id(0)
        tile = tl.program_id(1)

        start = 0
        for i in range(seq):
            start += tl.load(lengths + i)
        length = tl.minimum(tl.load(lengths + seq), max_len)

        packed_view = tle_exp.make_strided_view(
            packed,
            shape=[(seq + 1) * max_len, FEATURES],
            strides=[FEATURES, 1],
            tile=[BLOCK_T, BLOCK_D],
            traversal_strides=[1, BLOCK_D],
        )
        rows = tile_load(
            packed_view,
            index=(seq * max_len + tile * BLOCK_T, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        )

        # The view stops at the last valid row of this sequence, so the rows of
        # a tail tile past the length are dropped instead of overwriting the
        # next sequence.
        out_view = tle_exp.make_strided_view(
            out,
            shape=[start + length, FEATURES],
            strides=[FEATURES, 1],
            tile=[BLOCK_T, BLOCK_D],
            traversal_strides=[1, BLOCK_D],
        )
        tile_store(
            out_view,
            value=rows,
            index=(start + tile * BLOCK_T, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _tv_rows_that_fit(features, element_size):
        """Rows per tile UB can hold. Only the gathered tile is live."""
        return tile_rows_for_ub(
            triton.next_power_of_2(features) * element_size, 1, TV_BLOCK_T
        )

    def _tv_block_t(features, element_size, max_len):
        rows = max(_tv_rows_that_fit(features, element_size), MIN_TILE_ROWS)
        return min(rows, max(triton.next_power_of_2(max_len), MIN_TILE_ROWS))

    def _unpack_seq_tensorview(packed_tensor, lengths, _block_t=None, _block_d=None):
        """``_block_t`` / ``_block_d`` are the generic path's tile sizes. The
        tile here is sized from the UB budget, so they are accepted and
        ignored, which keeps one argument list for both paths."""
        logger.debug("GEMS_ASCEND UNPACK_SEQ_TRITON(TENSORVIEW)")
        original_shape = packed_tensor.shape
        num_seqs, max_len = original_shape[:2]
        packed = packed_tensor.reshape(num_seqs, max_len, -1).contiguous()
        features = packed.shape[2]
        total = int(lengths.sum().item())
        lengths = lengths.to(torch.int32).contiguous()

        block_t = _tv_block_t(features, packed.element_size(), max_len)
        out = torch.empty((total, features), dtype=packed.dtype, device=packed.device)
        grid = (num_seqs, triton.cdiv(max_len, block_t))
        with torch_device_fn.device(packed.device):
            unpack_seq_tv_kernel[grid](
                out,
                packed,
                lengths,
                max_len,
                FEATURES=features,
                BLOCK_T=block_t,
                BLOCK_D=triton.next_power_of_2(features),
            )
        return out.reshape((total,) + tuple(original_shape[2:]))

    def _tv_eligible(packed_tensor, lengths):
        if packed_tensor.numel() == 0 or lengths.numel() == 0:
            return False
        if packed_tensor.dim() < 3:
            return False
        if packed_tensor.device != lengths.device:
            return False
        # lengths is read on device by the kernel and its sum sizes the output.
        if lengths.device.type == "cpu":
            return False
        num_seqs, max_len = packed_tensor.shape[:2]
        if num_seqs == 0 or max_len == 0:
            return False
        # The kernel's prefix scan walks lengths[0..seq) for its own sequence
        # id, so there has to be one length per padded sequence slot. (The
        # launch makes lengths contiguous itself, so that is not checked here.)
        if lengths.numel() != num_seqs:
            return False
        features = packed_tensor.numel() // (num_seqs * max_len)
        if features == 0:
            return False
        if int(lengths.sum().item()) <= 0:
            return False
        # The prefix scan over lengths runs per program, so a long sequence list
        # costs more in the scan than the DMA saves.
        if num_seqs > 1024:
            return False
        return (
            _tv_rows_that_fit(features, packed_tensor.element_size()) >= MIN_TILE_ROWS
        )

    def unpack_seq_triton(
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
        block_t: int = 64,
        block_d: int = 64,
    ) -> torch.Tensor:
        if not _tv_eligible(packed_tensor, lengths):
            return _unpack_seq_generic(packed_tensor, lengths, block_t, block_d)
        return tensorview_call(
            "UNPACK_SEQ_TRITON",
            _unpack_seq_tensorview,
            _unpack_seq_generic,
            packed_tensor,
            lengths,
            # The fast path derives its own tile height, and the generic
            # fallback takes these as its block_t / block_d.
            block_t,
            block_d,
        )

else:
    unpack_seq_triton = _unpack_seq_generic
