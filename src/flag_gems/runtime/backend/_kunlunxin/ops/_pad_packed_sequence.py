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

"""Kunlunxin (XPU) implementation of ``_pad_packed_sequence``.

The generic kernel walks one program per ``(t, b)`` row and branches on a
runtime scalar ``b < batch_sizes[t]`` to choose between a copy path and a
padding path.  Two XPU hazards make that miscompile (padding region garbage):

* the scalar ``tl.load(batch_sizes + t, mask=..., other=0)`` silently ignores
  the ``other`` clause, so ``t >= num_steps`` reads out-of-bounds metadata and
  the copy branch fires where padding was intended;
* the data-dependent ``if b < bs_t`` branch materialises both the gather-load
  copy path and the scalar-store padding path (see KERNEL_OPT_EXPERIENCE #3/6).

This override keeps every address affine: one program owns a contiguous chunk
of a time-step row and loads the (start, length) metadata for that step from a
single packed int64 (experience #33: scalar metadata loads are the expensive
``gm2lm_v3`` + ``mfence`` path, so both fields are folded into one load).  Two
passes cover the row without any data-dependent address arithmetic:

1. copy ``[0, copy_len)`` from the packed source at ``copy_start`` — source and
   destination are both pure lane offsets, i.e. a contiguous block copy;
2. fill the tail ``[copy_len, row_total)`` with ``padding_value`` — the base is
   the scalar ``copy_len``.

The previous revision used one pass with the source address
``copy_start + where(is_copy, p, 0)``; the non-affine address made the whole
load lower to a per-lane discrete gather, which measured 3.6x slower than this
split on every benchmark shape (experience #32 is about affine source access).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _pad_packed_sequence_copy_kernel(
    data_ptr,
    packed_ptr,
    out_ptr,
    num_steps,
    row_total,
    BLOCK: tl.constexpr,
):
    """Copy the packed prefix of time-step ``t`` into row ``t`` of the output.

    Both the source (``copy_start + p``) and the destination (``t*row_total +
    p``) are affine in the lane index, so the load/store lower to a contiguous
    block transfer.  The mask is also contiguous (``p < copy_len``), so masked
    lanes never reach memory; whatever ``other`` returns for them is discarded
    by the masked store, which keeps this safe against the unreliable XPU
    masked-load ``other`` value (#51).
    """
    t = tl.program_id(0)
    block = tl.program_id(1)
    t_safe = tl.minimum(t, num_steps - 1)
    # packed = (copy_start << 32) | copy_len, both fields int32 element counts.
    packed = tl.load(packed_ptr + t_safe).to(tl.int64)
    copy_start = (packed >> 32).to(tl.int32)
    copy_len = (packed & 0xFFFFFFFF).to(tl.int32)
    copy_len = tl.where(t < num_steps, copy_len, 0)

    p = block * BLOCK + tl.arange(0, BLOCK)
    mask = p < copy_len
    val = tl.load(data_ptr + copy_start + p, mask=mask, other=0.0)
    tl.store(out_ptr + t.to(tl.int64) * row_total + p, val, mask=mask)


@libentry()
@triton.jit
def _pad_packed_sequence_pad_kernel(
    out_ptr,
    packed_ptr,
    padding_value,
    num_steps,
    row_total,
    BLOCK: tl.constexpr,
):
    """Fill the tail ``[copy_len, row_total)`` of every output row.

    The destination is ``copy_len + p`` with ``copy_len`` a scalar, so the
    store is affine; the mask trims the lanes past the row end.
    """
    t = tl.program_id(0)
    block = tl.program_id(1)
    t_safe = tl.minimum(t, num_steps - 1)
    packed = tl.load(packed_ptr + t_safe).to(tl.int64)
    copy_len = (packed & 0xFFFFFFFF).to(tl.int32)
    copy_len = tl.where(t < num_steps, copy_len, 0)

    p = block * BLOCK + tl.arange(0, BLOCK)
    dst = copy_len + p
    tl.store(
        out_ptr + t.to(tl.int64) * row_total + dst,
        padding_value,
        mask=dst < row_total,
    )


def _pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length):
    """Pad a packed variable-length sequence back into a dense tensor."""
    logger.debug("GEMS_KUNLUNXIN _PAD_PACKED_SEQUENCE")

    num_steps = batch_sizes.numel()
    max_batch = int(batch_sizes[0].item())

    total_steps = num_steps
    if total_length > 0:
        total_steps = max(total_steps, int(total_length))

    feature_shape = data.shape[1:]
    feature_size = 1
    for s in feature_shape:
        feature_size *= s

    # Metadata arithmetic stays on the host as Python integer bookkeeping;
    # host functions must not invoke PyTorch compute operators. Fold the two
    # int32 fields into a single int64 so the kernel performs one scalar load
    # instead of two.
    batch_sizes_list = batch_sizes.to(dtype=torch.int64, device="cpu").tolist()
    time_offsets = []
    offset = 0
    for batch_size in batch_sizes_list:
        time_offsets.append(offset)
        offset += batch_size

    # Per-batch length = last time step at which the batch is still active.
    # ``batch_sizes`` is non-increasing, so batch ``b`` leaves the packed layout
    # once ``b >= batch_sizes[t]``: walking the steps once and stamping
    # ``[batch_sizes[t], batch_sizes[t-1])`` is O(max_batch) instead of the
    # O(max_batch * num_steps) double loop this replaces (the double loop cost
    # more than the two kernels at the largest benchmark shape).
    lengths_list = [0] * max_batch
    prev = max_batch
    for time, batch_size in enumerate(batch_sizes_list):
        for batch_index in range(batch_size, prev):
            lengths_list[batch_index] = time
        prev = batch_size
    for batch_index in range(prev):
        lengths_list[batch_index] = num_steps

    packed_list = [
        (start * feature_size << 32) | (batch_size * feature_size)
        for start, batch_size in zip(time_offsets, batch_sizes_list)
    ]
    packed_dev = torch.tensor(packed_list, dtype=torch.int64, device=data.device)
    lengths = torch.tensor(lengths_list, dtype=torch.int64)

    output = torch.empty(
        (total_steps, max_batch, *feature_shape),
        dtype=data.dtype,
        device=data.device,
    )
    out_flat = output.view(total_steps, max_batch, feature_size)

    row_total = max_batch * feature_size
    # Two affine passes instead of one data-dependent pass.  The previous
    # single kernel built the source address as ``where(is_copy, p, 0)``, which
    # is not affine in the lane index and therefore lowered to a per-lane
    # discrete gather (~3.6x slower on every benchmark shape).  Splitting into
    # "copy the contiguous packed prefix" + "fill the contiguous tail" keeps
    # both passes affine.  The tile is sized so each output row needs as few
    # blocks as possible: wide tiles amortise the per-program launch cost,
    # which dominates at these sizes.
    BLOCK = 8192 if row_total <= 16384 else 16384
    num_blocks = triton.cdiv(row_total, BLOCK)
    grid = (total_steps, num_blocks)

    flat_data = data.reshape(-1)
    with torch_device_fn.device(data.device):
        _pad_packed_sequence_copy_kernel[grid](
            flat_data,
            packed_dev,
            out_flat,
            num_steps,
            row_total,
            BLOCK=BLOCK,
            num_warps=8,
        )
        _pad_packed_sequence_pad_kernel[grid](
            out_flat,
            packed_dev,
            padding_value,
            num_steps,
            row_total,
            BLOCK=BLOCK,
            num_warps=8,
        )

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, lengths.cpu()
