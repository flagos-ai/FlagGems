# Copyright 2026, The FlagOS Contributors.
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

import triton
import triton.language as tl

# Use the gems implementations for allocation/fill instead of raw torch calls.
import flag_gems.ops as _general_ops
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Row loop width: keeps the grid at cdiv(batch, ROWS_PER_PROG) programs (avoids
# the XPU launch-bound floor of one program per row) without unrolling a huge
# body (ROWS_PER_PROG x BLOCK lanes per program).
_ROWS_PER_PROG = 16
# Column block cap: avoids giant constexpr tiles (IR explosion); the static
# column loop covers max_length in BLOCK-wide steps.
_BLOCK_N_CAP = 512


@libentry()
@triton.jit
def _jagged_to_padded_dense_forward_kernel(
    values,
    offsets,
    output,
    padding_value: tl.constexpr,
    batch_size: tl.constexpr,
    max_length: tl.constexpr,
    total_length: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Write the real (non-padding) elements of a padded dense output.

    Grid: ``(cdiv(batch_size, ROWS_PER_PROG),)``.  Program ``pid`` owns rows
    ``[pid*ROWS_PER_PROG, (pid+1)*ROWS_PER_PROG)``.  For each row ``r`` with
    ``seq_len`` and ``seq_start`` loaded from ``offsets``:
      ``output[r, c] = values[seq_start + c]`` for ``c < seq_len``.
    Elements with ``c >= seq_len`` (and OOB rows) are never stored; the output
    tensor is pre-filled with ``padding_value`` by the caller.
    """
    pid = ext.program_id(0)
    row0 = pid * ROWS_PER_PROG

    for r in tl.static_range(0, ROWS_PER_PROG):
        row = row0 + r
        if row < batch_size:
            seq_start = tl.load(offsets + row)
            seq_len = tl.load(offsets + row + 1) - seq_start
            for j in tl.range(0, max_length, BLOCK_N):
                col = j + tl.arange(0, BLOCK_N)
                # Clamp so a masked load that is lowered as unmasked (XPU
                # tail-block limitation) cannot read past the allocation.
                src = tl.minimum(seq_start + col, total_length - 1)
                val = tl.load(values + src, mask=col < seq_len, other=padding_value)
                # Store only the real region: padded lanes are never written
                # (they already hold padding_value from the pre-fill).
                tl.store(
                    output + row * max_length + col,
                    val,
                    mask=col < seq_len,
                )


def _jagged_to_padded_dense_forward(values, offsets, max_lengths, padding_value=0.0):
    """Convert a jagged (variable-length) tensor to a padded dense tensor.

    Supports the single-batch-dimension case (1-D ``values``, 1-D ``offsets``);
    same calling convention as ``flag_gems.ops._jagged_to_padded_dense_forward``.
    """
    logger.debug("GEMS_KUNLUNXIN _JAGGED_TO_PADDED_DENSE_FORWARD")

    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets]
    if not isinstance(max_lengths, (list, tuple)):
        max_lengths = [max_lengths]

    num_batch_dims = len(offsets)
    assert (
        num_batch_dims == 1
    ), f"Only single batch dimension is supported, got {num_batch_dims}"

    offsets_0 = offsets[0]
    batch_size = int(offsets_0.numel()) - 1
    max_length = int(max_lengths[0])

    if batch_size <= 0 or max_length <= 0:
        return _general_ops.empty(
            (max(batch_size, 0), max(max_length, 0)),
            dtype=values.dtype,
            device=values.device,
        )

    output = _general_ops.full(
        (batch_size, max_length),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )

    total_length = int(values.numel())
    if total_length == 0:
        # Every row is empty: output is already all padding_value.
        return output

    rows_per_prog = min(_ROWS_PER_PROG, batch_size)
    block_n = min(triton.next_power_of_2(max_length), _BLOCK_N_CAP)
    grid = (triton.cdiv(batch_size, rows_per_prog),)
    _jagged_to_padded_dense_forward_kernel[grid](
        values,
        offsets_0,
        output,
        padding_value,
        batch_size,
        max_length,
        total_length,
        ROWS_PER_PROG=rows_per_prog,
        BLOCK_N=block_n,
    )

    return output
