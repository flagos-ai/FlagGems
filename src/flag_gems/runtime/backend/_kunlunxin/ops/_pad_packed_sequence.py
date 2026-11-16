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

This override keeps the source access affine: one program owns a contiguous
chunk of a time-step row, loading the (start, length) metadata for that step
from a single packed int64 (experience #33: scalar metadata loads are the
expensive ``gm2lm_v3`` + ``mfence`` path, so both fields are folded into one
load).  The packed source address ``copy_start + p`` is a pure vector offset
within the row, so the copy is a contiguous block copy rather than a per-lane
gather (experience #32).
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
def _pad_packed_sequence_kernel(
    data_ptr,
    packed_ptr,
    out_ptr,
    padding_value,
    num_steps,
    row_total,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    block = tl.program_id(1)
    t_safe = tl.minimum(t, num_steps - 1)
    # packed = (copy_start << 32) | copy_len, both fields int32 element counts.
    packed = tl.load(packed_ptr + t_safe).to(tl.int64)
    copy_start = (packed >> 32).to(tl.int32)
    copy_len = (packed & 0xFFFFFFFF).to(tl.int32)
    copy_len = tl.where(t < num_steps, copy_len, 0)

    out_base = t * row_total
    p = block * BLOCK + tl.arange(0, BLOCK)
    p_mask = p < row_total
    is_copy = (p < copy_len) & p_mask
    p_safe = tl.where(is_copy, p, 0)

    src = copy_start + p_safe
    val = tl.load(data_ptr + src)
    val = tl.where(is_copy, val, padding_value)

    tl.store(out_ptr + out_base + p, val, mask=p_mask)


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

    lengths_list = []
    for batch_index in range(max_batch):
        length = 0
        for time, batch_size in enumerate(batch_sizes_list):
            if batch_index < batch_size:
                length = time + 1
        lengths_list.append(length)

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
    BLOCK = 4096
    num_blocks = triton.cdiv(row_total, BLOCK)
    grid = (total_steps, num_blocks)

    with torch_device_fn.device(data.device):
        _pad_packed_sequence_kernel[grid](
            data.reshape(-1),
            packed_dev,
            out_flat,
            padding_value,
            num_steps,
            row_total,
            BLOCK=BLOCK,
            num_warps=8,
        )

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, lengths.cpu()
