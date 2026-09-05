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
import struct
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_ASCEND_AIV_CORE_COUNT = 40
_BLOCK_SIZE = 8192
_TRITON_VERSION = tuple(int(v) for v in triton.__version__.split(".")[:2])
_PASS_CONSTEXPRS = (3, 3) <= _TRITON_VERSION <= (3, 6)
_GET_RAW_STREAM = triton.runtime.driver.active.get_current_stream


@triton.jit
def _copy_one(
    out_ptr,
    seq_ptr,
    length: tl.constexpr,
    batch_size: tl.constexpr,
    batch_index: tl.constexpr,
    padding_value,
    MAX_LEN: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
):
    index_type: tl.constexpr = (
        tl.int32 if batch_size * MAX_LEN * FEATURE_SIZE < 2**31 else tl.int64
    )
    pid = ext.program_id(0).to(index_type)
    if BATCH_FIRST:
        span = MAX_LEN * FEATURE_SIZE
        lanes = tl.arange(0, BLOCK_SIZE)
        for tile in tl.range(pid, tl.cdiv(span, BLOCK_SIZE), CORES):
            start = tile * BLOCK_SIZE
            offsets = start + lanes
            if start < length * FEATURE_SIZE:
                values = tl.load(
                    seq_ptr + offsets,
                    offsets < length * FEATURE_SIZE,
                    other=padding_value,
                )
            else:
                values = tl.full((BLOCK_SIZE,), padding_value, out_ptr.dtype.element_ty)
            tl.store(
                out_ptr + batch_index * span + offsets,
                values,
                offsets < span,
            )
    else:
        row_lanes = tl.arange(0, BLOCK_T)
        for tile in tl.range(
            pid, tl.cdiv(MAX_LEN, BLOCK_T) * tl.cdiv(FEATURE_SIZE, BLOCK_F), CORES
        ):
            row_start = (tile // tl.cdiv(FEATURE_SIZE, BLOCK_F)) * BLOCK_T
            col_start = (tile % tl.cdiv(FEATURE_SIZE, BLOCK_F)) * BLOCK_F
            cols = col_start + tl.arange(0, BLOCK_F)
            rows = row_start + row_lanes
            if row_start < length:
                values = tl.load(
                    seq_ptr + rows[:, None] * FEATURE_SIZE + cols[None, :],
                    (rows[:, None] < length) & (cols[None, :] < FEATURE_SIZE),
                    other=padding_value,
                )
            else:
                values = tl.full(
                    (BLOCK_T, BLOCK_F), padding_value, out_ptr.dtype.element_ty
                )
            output_offsets = (
                rows[:, None] * (batch_size * FEATURE_SIZE)
                + batch_index * FEATURE_SIZE
                + cols[None, :]
            )
            tl.store(
                out_ptr + output_offsets,
                values,
                (rows[:, None] < MAX_LEN) & (cols[None, :] < FEATURE_SIZE),
            )


@libentry()
@triton.jit
def _pad_owned_2(
    out_ptr,
    p0,
    p1,
    n0: tl.constexpr,
    n1: tl.constexpr,
    batch_size: tl.constexpr,
    batch_offset: tl.constexpr,
    padding_value,
    MAX_LEN: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
):
    owner = ext.program_id(1).to(tl.int32)
    if owner == 0:
        _copy_one(
            out_ptr,
            p0,
            n0,
            batch_size,
            batch_offset,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    else:
        _copy_one(
            out_ptr,
            p1,
            n1,
            batch_size,
            batch_offset + 1,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )


@libentry()
@triton.jit
def _pad_owned_4(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    n0: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    n3: tl.constexpr,
    batch_size: tl.constexpr,
    batch_offset: tl.constexpr,
    padding_value,
    MAX_LEN: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
):
    owner = ext.program_id(1).to(tl.int32)
    if owner == 0:
        _copy_one(
            out_ptr,
            p0,
            n0,
            batch_size,
            batch_offset,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 1:
        _copy_one(
            out_ptr,
            p1,
            n1,
            batch_size,
            batch_offset + 1,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 2:
        _copy_one(
            out_ptr,
            p2,
            n2,
            batch_size,
            batch_offset + 2,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    else:
        _copy_one(
            out_ptr,
            p3,
            n3,
            batch_size,
            batch_offset + 3,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )


@libentry()
@triton.jit
def _pad_owned_8(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    n3: tl.constexpr,
    n4: tl.constexpr,
    n5: tl.constexpr,
    n6: tl.constexpr,
    n7: tl.constexpr,
    batch_size: tl.constexpr,
    batch_offset: tl.constexpr,
    padding_value,
    MAX_LEN: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BATCH_FIRST: tl.constexpr,
):
    owner = ext.program_id(1).to(tl.int32)
    if owner == 0:
        _copy_one(
            out_ptr,
            p0,
            n0,
            batch_size,
            batch_offset,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 1:
        _copy_one(
            out_ptr,
            p1,
            n1,
            batch_size,
            batch_offset + 1,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 2:
        _copy_one(
            out_ptr,
            p2,
            n2,
            batch_size,
            batch_offset + 2,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 3:
        _copy_one(
            out_ptr,
            p3,
            n3,
            batch_size,
            batch_offset + 3,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 4:
        _copy_one(
            out_ptr,
            p4,
            n4,
            batch_size,
            batch_offset + 4,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 5:
        _copy_one(
            out_ptr,
            p5,
            n5,
            batch_size,
            batch_offset + 5,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    elif owner == 6:
        _copy_one(
            out_ptr,
            p6,
            n6,
            batch_size,
            batch_offset + 6,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )
    else:
        _copy_one(
            out_ptr,
            p7,
            n7,
            batch_size,
            batch_offset + 7,
            padding_value,
            MAX_LEN,
            FEATURE_SIZE,
            CORES,
            BLOCK_SIZE,
            BLOCK_T,
            BLOCK_F,
            BATCH_FIRST,
        )


class _ChunkPlan:
    """Shape-only metadata; compiled launchers never retain input tensors."""

    def __init__(self, lengths, offset, batch_size, span, feature, batch_first, block):
        count = len(lengths)
        width = 2 if count <= 2 else (4 if count <= 4 else 8)
        self.kernel_fn = {2: _pad_owned_2, 4: _pad_owned_4, 8: _pad_owned_8}[width]
        max_len = span // feature
        block_f = min(block, triton.next_power_of_2(feature))
        block_t = max(1, min(32, block // block_f))
        block_size = min(block, triton.next_power_of_2(span))
        tiles = (
            triton.cdiv(span, block_size)
            if batch_first
            else triton.cdiv(max_len, block_t) * triton.cdiv(feature, block_f)
        )
        cores = min(tiles, max(1, _ASCEND_AIV_CORE_COUNT // count))
        self.indices = tuple(range(offset, offset + count)) + (offset,) * (width - count)
        self.scalars = tuple(lengths) + (0,) * (width - count) + (batch_size, offset)
        self.runtime_scalars = self.scalars if _PASS_CONSTEXPRS else ()
        self.grid = (cores, count, 1)
        self.meta = dict(
            MAX_LEN=max_len,
            FEATURE_SIZE=feature,
            CORES=cores,
            BLOCK_SIZE=block_size,
            BLOCK_T=block_t,
            BLOCK_F=block_f,
            BATCH_FIRST=batch_first,
        )
        self.abi_args = tuple(self.meta.values()) if _PASS_CONSTEXPRS else ()
        self.launchers = {}

    def run(self, out, sequences, padding, pad_key, device_index, dtype, stream):
        pointers = tuple(sequences[i] for i in self.indices)
        alignment = (out.data_ptr() % 16,) + tuple(p.data_ptr() % 16 for p in pointers)
        key = (device_index, dtype, pad_key, alignment)
        runner = self.launchers.get(key)
        if runner is None:
            kernel, _ = self.kernel_fn[self.grid](
                out, *pointers, *self.scalars, padding, **self.meta
            )
            if len(self.launchers) >= 64:
                self.launchers.pop(next(iter(self.launchers)))
            self.launchers[key] = kernel[self.grid]
        else:
            runner(
                out,
                *pointers,
                *self.runtime_scalars,
                padding,
                *self.abi_args,
                stream=stream
            )


@lru_cache(maxsize=256)
def _make_plan(lengths, trailing_shape, batch_first, block):
    batch_size = len(lengths)
    max_len = max(lengths)
    feature = math.prod(trailing_shape)
    output_shape = (
        (batch_size, max_len, *trailing_shape)
        if batch_first
        else (max_len, batch_size, *trailing_shape)
    )
    span = max_len * feature
    if span == 0:
        return output_shape, ()
    chunks = tuple(
        _ChunkPlan(
            lengths[offset : offset + 8],
            offset,
            batch_size,
            span,
            feature,
            batch_first,
            block,
        )
        for offset in range(0, batch_size, 8)
    )
    return output_shape, chunks


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """Copy and pad variable-length sequences on Ascend."""
    logger.debug("GEMS_ASCEND PAD_SEQUENCE")
    if len(sequences) == 0:
        raise RuntimeError("pad_sequence empty input")
    first = sequences[0]
    first_shape = first.shape
    if len(first_shape) == 0:
        raise RuntimeError("pad_sequence requires at least one dimension")
    dtype = first.dtype
    device = first.device
    trailing_shape = first_shape[1:]
    lengths = [first_shape[0]]
    contiguous = [first if first.is_contiguous() else first.contiguous()]
    for sequence in sequences[1:]:
        shape = sequence.shape
        if len(shape) == 0:
            raise RuntimeError("pad_sequence requires at least one dimension")
        if shape[1:] != trailing_shape:
            raise RuntimeError(
                "The size of tensor a must match the size of tensor b at non-singleton dimension"
            )
        if sequence.dtype != dtype:
            raise RuntimeError(
                "pad_sequence expects all input tensors to have the same dtype"
            )
        if sequence.device != device:
            raise RuntimeError(
                "pad_sequence expects all input tensors to be on the same device"
            )
        lengths.append(shape[0])
        contiguous.append(
            sequence if sequence.is_contiguous() else sequence.contiguous()
        )
    output_shape, chunks = _make_plan(
        tuple(lengths), trailing_shape, bool(batch_first), _BLOCK_SIZE
    )
    out = torch.empty(output_shape, dtype=dtype, device=device)
    if not chunks:
        return out
    kernel_out = out
    kernel_dtype = dtype
    padding = padding_value
    if dtype == torch.float64:
        kernel_out = out.view(torch.int64)
        contiguous = [s.view(torch.int64) for s in contiguous]
        kernel_dtype = torch.int64
        padding = struct.unpack("<q", struct.pack("<d", float(padding_value)))[0]
    elif dtype == torch.bool:
        # Ascend represents bool loads as int8; keep both branches byte-typed.
        kernel_out = out.view(torch.uint8)
        contiguous = [s.view(torch.uint8) for s in contiguous]
        kernel_dtype = torch.uint8
        padding = int(bool(padding_value))
    pad_key = (
        (type(padding), padding)
        if isinstance(padding, int)
        else (type(padding), struct.pack("<d", float(padding)))
    )
    if torch.npu.current_device() == device.index:
        stream = _GET_RAW_STREAM(device.index)
        for chunk in chunks:
            chunk.run(
                kernel_out,
                contiguous,
                padding,
                pad_key,
                device.index,
                kernel_dtype,
                stream,
            )
    else:
        with torch_device_fn.device(device):
            stream = _GET_RAW_STREAM(device.index)
            for chunk in chunks:
                chunk.run(
                    kernel_out,
                    contiguous,
                    padding,
                    pad_key,
                    device.index,
                    kernel_dtype,
                    stream,
                )
    return out