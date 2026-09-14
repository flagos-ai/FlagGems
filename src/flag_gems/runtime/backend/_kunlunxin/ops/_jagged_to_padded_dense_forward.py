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
# Kunlunxin (XPU) specialization for _jagged_to_padded_dense_forward.
#
# Kernel body is kept structurally identical to the proven generic
# implementation (padding-fill loop + masked-load copy loop). The only change
# is the launch BLOCK_SIZE: sized to next_power_of_2(max_length) (capped at
# 256) so each row is covered by whole blocks without half-empty 128-lane
# masked blocks for small max_lengths.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _jagged_to_padded_dense_forward_kernel(
    values,
    offsets,
    output,
    padding_value: tl.constexpr,
    batch_size: tl.constexpr,
    max_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    batch_idx = pid

    if batch_idx >= batch_size:
        return

    # Get the start and end offset for this sequence
    seq_start = tl.load(offsets + batch_idx)
    seq_end = tl.load(offsets + batch_idx + 1)

    # Calculate the actual sequence length
    seq_length = seq_end - seq_start

    # Compute the row offset in the output
    row_offset = batch_idx * max_length

    # Fill with padding value (vectorized per block)
    for j in tl.range(0, max_length, BLOCK_SIZE):
        out_offsets = row_offset + j + tl.arange(0, BLOCK_SIZE)
        out_mask = (j + tl.arange(0, BLOCK_SIZE)) < max_length
        tl.store(output + out_offsets, padding_value, mask=out_mask)

    # Copy actual values (vectorized per block)
    for j in tl.range(0, seq_length, BLOCK_SIZE):
        offsets_vec = seq_start + j + tl.arange(0, BLOCK_SIZE)
        mask = offsets_vec < seq_end

        values_vec = tl.load(values + offsets_vec, mask=mask, other=padding_value)

        out_offsets = row_offset + j + tl.arange(0, BLOCK_SIZE)
        out_mask = (j + tl.arange(0, BLOCK_SIZE)) < seq_length

        tl.store(output + out_offsets, values_vec, mask=out_mask)


def _jagged_to_padded_dense_forward(values, offsets, max_lengths, padding_value=0.0):
    """Convert a jagged (variable-length) tensor to a padded dense tensor."""
    logger.debug("GEMS JAGGED TO PADDED DENSE FORWARD")

    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets]
    if not isinstance(max_lengths, (list, tuple)):
        max_lengths = [max_lengths]

    num_batch_dims = len(offsets)
    assert (
        num_batch_dims == 1
    ), f"Only single batch dimension is supported, got {num_batch_dims}"

    offsets_0 = offsets[0]
    batch_size = offsets_0.numel() - 1
    max_length = max_lengths[0]

    # Compute output shape
    # For single batch dim: (batch_size, max_length)
    output_shape = (batch_size, max_length)
    output = torch.empty(output_shape, dtype=values.dtype, device=values.device)

    grid = lambda meta: (batch_size,)
    _jagged_to_padded_dense_forward_kernel[grid](
        values,
        offsets_0,
        output,
        padding_value,
        batch_size,
        max_length,
        # Two-tier block sizing (measured on P800):
        #  - max_length <= 128: keep the generic 128 (best for small rows;
        #    smaller blocks measurably regress, e.g. max_length=64 with 64
        #    lanes is ~1.6x slower);
        #  - max_length > 128: one whole-row block up to 256 lanes so the
        #    fill/copy loops run a single iteration instead of two (e.g.
        #    batch=512/max_length=256: 0.364 -> 0.244 ms fp16).
        BLOCK_SIZE=128 if max_length <= 128 else min(triton.next_power_of_2(max_length), 256),
    )

    return output
