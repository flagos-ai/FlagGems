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

"""Hygon pad_sequence candidate; requires validation on HCU hardware."""

import logging
import math
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _pad_sequence_kernel(
    Out,
    Sources,
    Lengths: tl.constexpr,
    Batch: tl.constexpr,
    BatchOffset: tl.constexpr,
    MaxLength: tl.constexpr,
    Feature: tl.constexpr,
    Padding: tl.constexpr,
    BatchFirst: tl.constexpr,
    Block: tl.constexpr,
):
    # Each program owns one sequence, so source selection is uniform across
    # every lane. No concatenation, device pointer table or temporary output.
    IndexType: tl.constexpr = (
        tl.int64 if Batch * MaxLength * Feature >= 2**31 else tl.int32
    )
    owner = tl.program_id(1)
    source = Sources[0]
    length = tl.full((), Lengths[0], IndexType)
    for index in tl.static_range(1, len(Lengths)):
        source = tl.where(owner == index, Sources[index], source)
        length = tl.where(owner == index, Lengths[index], length)

    offsets = tl.program_id(0).to(IndexType) * Block + tl.arange(0, Block)
    valid_output = offsets < MaxLength * Feature
    values = tl.load(
        source + offsets,
        mask=valid_output & (offsets < length * Feature),
        other=Padding,
    )
    batch_index = (owner + BatchOffset).to(IndexType)
    if BatchFirst:
        destination = batch_index * MaxLength * Feature + offsets
    else:
        destination = (
            (offsets // Feature) * (Batch * Feature)
            + batch_index * Feature
            + offsets % Feature
        )
    tl.store(Out + destination, values, mask=valid_output)


@libentry()
@triton.jit
def _pad_sequence_2(
    Out,
    P0,
    P1,
    Lengths: tl.constexpr,
    Batch: tl.constexpr,
    BatchOffset: tl.constexpr,
    MaxLength: tl.constexpr,
    Feature: tl.constexpr,
    Padding: tl.constexpr,
    BatchFirst: tl.constexpr,
    Block: tl.constexpr,
):
    _pad_sequence_kernel(
        Out,
        (P0, P1),
        Lengths,
        Batch,
        BatchOffset,
        MaxLength,
        Feature,
        Padding,
        BatchFirst,
        Block,
    )


@libentry()
@triton.jit
def _pad_sequence_4(
    Out,
    P0,
    P1,
    P2,
    P3,
    Lengths: tl.constexpr,
    Batch: tl.constexpr,
    BatchOffset: tl.constexpr,
    MaxLength: tl.constexpr,
    Feature: tl.constexpr,
    Padding: tl.constexpr,
    BatchFirst: tl.constexpr,
    Block: tl.constexpr,
):
    _pad_sequence_kernel(
        Out,
        (P0, P1, P2, P3),
        Lengths,
        Batch,
        BatchOffset,
        MaxLength,
        Feature,
        Padding,
        BatchFirst,
        Block,
    )


@libentry()
@triton.jit
def _pad_sequence_8(
    Out,
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    Lengths: tl.constexpr,
    Batch: tl.constexpr,
    BatchOffset: tl.constexpr,
    MaxLength: tl.constexpr,
    Feature: tl.constexpr,
    Padding: tl.constexpr,
    BatchFirst: tl.constexpr,
    Block: tl.constexpr,
):
    _pad_sequence_kernel(
        Out,
        (P0, P1, P2, P3, P4, P5, P6, P7),
        Lengths,
        Batch,
        BatchOffset,
        MaxLength,
        Feature,
        Padding,
        BatchFirst,
        Block,
    )


@libentry()
@triton.jit
def _pad_sequence_16(
    Out,
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
    P11,
    P12,
    P13,
    P14,
    P15,
    Lengths: tl.constexpr,
    Batch: tl.constexpr,
    BatchOffset: tl.constexpr,
    MaxLength: tl.constexpr,
    Feature: tl.constexpr,
    Padding: tl.constexpr,
    BatchFirst: tl.constexpr,
    Block: tl.constexpr,
):
    _pad_sequence_kernel(
        Out,
        (P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15),
        Lengths,
        Batch,
        BatchOffset,
        MaxLength,
        Feature,
        Padding,
        BatchFirst,
        Block,
    )


@libentry()
@triton.jit
def _pad_sequence_32(
    Out,
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
    P11,
    P12,
    P13,
    P14,
    P15,
    P16,
    P17,
    P18,
    P19,
    P20,
    P21,
    P22,
    P23,
    P24,
    P25,
    P26,
    P27,
    P28,
    P29,
    P30,
    P31,
    Lengths: tl.constexpr,
    Batch: tl.constexpr,
    BatchOffset: tl.constexpr,
    MaxLength: tl.constexpr,
    Feature: tl.constexpr,
    Padding: tl.constexpr,
    BatchFirst: tl.constexpr,
    Block: tl.constexpr,
):
    _pad_sequence_kernel(
        Out,
        (
            P0,
            P1,
            P2,
            P3,
            P4,
            P5,
            P6,
            P7,
            P8,
            P9,
            P10,
            P11,
            P12,
            P13,
            P14,
            P15,
            P16,
            P17,
            P18,
            P19,
            P20,
            P21,
            P22,
            P23,
            P24,
            P25,
            P26,
            P27,
            P28,
            P29,
            P30,
            P31,
        ),
        Lengths,
        Batch,
        BatchOffset,
        MaxLength,
        Feature,
        Padding,
        BatchFirst,
        Block,
    )


_KERNELS = {
    2: _pad_sequence_2,
    4: _pad_sequence_4,
    8: _pad_sequence_8,
    16: _pad_sequence_16,
    32: _pad_sequence_32,
}


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    logger.debug("GEMS_HYGON PAD_SEQUENCE")
    batch = len(sequences)
    if batch == 0:
        raise RuntimeError("pad_sequence empty input")

    first = sequences[0]
    if first.ndim == 0:
        raise RuntimeError("pad_sequence expects tensors with at least one dimension")
    tail = first.shape[1:]
    dtype = first.dtype
    device = first.device
    seqs = []
    lengths = []
    for sequence in sequences:
        if sequence.ndim == 0 or sequence.shape[1:] != tail:
            raise RuntimeError("pad_sequence expects matching trailing dimensions")
        if sequence.device != device:
            raise RuntimeError("pad_sequence expects all tensors on the same device")
        # Native pad_sequence uses the first tensor's dtype for the output.
        # Keep the common homogeneous case free of dtype conversion kernels.
        if sequence.dtype != dtype:
            sequence = sequence.to(dtype=dtype)
        if not sequence.is_contiguous():
            sequence = sequence.contiguous()
        seqs.append(sequence)
        lengths.append(sequence.shape[0])

    max_length = max(lengths)
    feature = math.prod(tail)
    shape = (batch, max_length, *tail) if batch_first else (max_length, batch, *tail)
    output = torch.empty(shape, dtype=dtype, device=device)
    if max_length == 0 or feature == 0:
        return output

    # Reinterpret doubles to preserve both the input bits and the Python
    # scalar's full precision; an inferred fp32 literal would round padding.
    kernel_output = output
    if dtype == torch.float64:
        kernel_output = output.view(torch.int64)
        seqs = [sequence.view(torch.int64) for sequence in seqs]
        padding_value = struct.unpack("q", struct.pack("d", float(padding_value)))[0]

    # Keep compiler/kernel argument growth bounded for arbitrarily long lists.
    # All current benchmark batches fit in one launch.
    group_size = 32
    block = 256 if max_length * feature <= 4096 else 2048
    grid_x = triton.cdiv(max_length * feature, block)
    with torch_device_fn.device(device):
        for start in range(0, batch, group_size):
            sources = seqs[start : start + group_size]
            count = len(sources)
            width = max(2, triton.next_power_of_2(count))
            sources += [sources[0]] * (width - count)
            _KERNELS[width][(grid_x, count)](
                kernel_output,
                *sources,
                tuple(lengths[start : start + group_size]),
                batch,
                start,
                max_length,
                feature,
                padding_value,
                batch_first,
                block,
                num_warps=4,
            )
    return output
