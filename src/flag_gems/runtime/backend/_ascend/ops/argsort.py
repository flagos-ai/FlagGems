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

import hashlib
import inspect
import json
import logging
import os
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.backends.ascend.compiler as compiler
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler import ASTSource
from triton.runtime.jit import JITFunction

from flag_gems.ops.argsort import (
    _argsort_merge,
    _argsort_row_offset,
    _argsort_tiles,
    _packed_merge,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

# Scoped paired-sort lowering; no process-wide compiler replacement.
original = compiler.ttir_to_linalg


def _require(condition, message):
    if not condition:
        raise RuntimeError("Unsupported Ascend paired-sort lowering: " + message)


def bridge(text):
    pattern = (
        "(%[\\w]+) = (?:call|func.call) @triton_sort\\((%[\\w]+), (%[\\w]+), (%[\\w]+)\\)"
        " : \\(tensor<(\\d+(?:x\\d+)*)xf32>, i64, i1\\) -> tensor<\\5xf32>"
    )
    hits = list(re.finditer(pattern, text))
    _require(len(hits) == 1, "expected one sort call")
    m = hits[0]
    dst, src, axis, desc, n = m.groups()
    d = re.search(re.escape(desc) + r" = arith.constant (true|false)", text)
    _require(d is not None, "static direction missing")
    axis_match = re.search(re.escape(axis) + r" = arith.constant (\d+) : i64", text)
    _require(axis_match is not None, "static sort axis missing")
    sort_axis = int(axis_match.group(1))
    replacement = (
        f"%paired_value_empty = tensor.empty() : tensor<{n}xf32>\n"
        f"    %paired_index_empty = tensor.empty() : tensor<{n}xi32>\n"
        f"    {dst}, %paired_index = hivm.hir.vsort ins({src} : tensor<{n}xf32>) "
        f"outs(%paired_value_empty, %paired_index_empty : tensor<{n}xf32>, tensor<{n}xi32>) "
        f"descending = {d.group(1)} sort_axis = {sort_axis} -> tensor<{n}xf32>, tensor<{n}xi32>"
    )
    replacement += f"\n    %paired_index64 = arith.extsi %paired_index : tensor<{n}xi32> to tensor<{n}xi64>"
    # Rewire the sort-dependent index placeholder before masked slicing.
    # A constant placeholder can be broadcast-compressed by the frontend.
    tail = text[m.end() :]
    cast_pattern = (
        r"(?m)^ *([%][\w]+) = arith.bitcast "
        + re.escape(dst)
        + r" : tensor<"
        + n
        + r"xf32> to tensor<"
        + n
        + r"xi32>[^\n]*\n"
    )
    casts = list(re.finditer(cast_pattern, tail))
    _require(len(casts) == 1, "expected one sort-dependent index placeholder")
    placeholder = casts[0].group(1)
    tail = tail[: casts[0].start()] + tail[casts[0].end() :]
    tail, count = re.subn(re.escape(placeholder) + r"(?![\w])", "%paired_index", tail)
    _require(count >= 1, "missing paired index consumer")
    text = text[: m.start()] + replacement + tail
    return text


BRIDGE_HASH = hashlib.sha256(inspect.getsource(bridge).encode()).hexdigest()


class _PairedSortSource(ASTSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext = "mlirbc"

    def hash(self):
        return hashlib.sha256(
            (super().hash() + BRIDGE_HASH + "masked-paired-v2-dependent").encode()
        ).hexdigest()

    def make_ir(self, target, options, codegen_fns, module_map, context):
        _require(
            target.backend == "npu" and options.use_bytecode,
            "paired sort requires Ascend bytecode compilation",
        )
        mod = super().make_ir(target, options, codegen_fns, module_map, context)
        metadata = {"target": target, **options.__dict__}
        mod = compiler.make_ttir(mod, metadata, options)
        text = original(mod, metadata, options, named_ops=True)
        return bridge(text)


class PairedSortJITFunction(JITFunction):
    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = _PairedSortSource
        return result


_paired_sort_jit = PairedSortJITFunction


@triton.jit
def _argsort_short_lsd_fp32_key(values, DESC: tl.constexpr):
    bits = values.to(tl.int32, bitcast=True)
    zero = tl.full((), 0, tl.int32)
    magnitude_mask = tl.full((), 2147483647, tl.int32)
    nan_threshold = tl.full((), 2139095040, tl.int32)
    value_mask = tl.full((), -1, tl.int32)
    sign_bit = tl.full((), -2147483648, tl.int32)
    magnitude = bits & magnitude_mask
    bits = tl.where(magnitude == zero, zero, bits)
    key = bits ^ tl.where(bits < zero, value_mask, sign_bit)
    key = tl.where(magnitude > nan_threshold, value_mask, key)
    if DESC:
        key = key ^ value_mask
    return key


@libentry()
@triton.jit
def _argsort_short_lsd_local(
    inp,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(1 <= N and N <= BLOCK and (BLOCK <= 256))
    tl.static_assert(1 << LOG_BLOCK == BLOCK)
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    input_base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + input_base + lane.to(tl.int64) * AXIS_STRIDE, valid, other=0)
    key = _argsort_short_lsd_fp32_key(values, DESC)
    digit_mask = tl.full((), 65535, tl.int32)
    digit_shift = tl.full((), 16, tl.int32)
    low_digit = (key & digit_mask).to(tl.int32)
    high_digit = (key >> digit_shift & digit_mask).to(tl.int32)
    low_code = low_digit << LOG_BLOCK | lane
    low_code = tl.where(valid, low_code, 1 << 24)
    sorted_low_code = al.sort(low_code.to(tl.float32), dim=-1, descending=False).to(
        tl.int32
    )
    permutation_low = sorted_low_code & BLOCK - 1
    current_high_digit = tl.gather(
        high_digit.to(tl.float32), permutation_low, axis=0
    ).to(tl.int32)
    high_code = current_high_digit << LOG_BLOCK | lane
    high_code = tl.where(valid, high_code, 1 << 24)
    sorted_high_code = al.sort(high_code.to(tl.float32), dim=-1, descending=False).to(
        tl.int32
    )
    permutation_high = sorted_high_code & BLOCK - 1
    original_index = tl.gather(
        permutation_low.to(tl.float32), permutation_high, axis=0
    ).to(tl.int32)
    output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(
        out + output_base + lane.to(tl.int64) * OUT_AXIS_STRIDE,
        original_index.to(tl.int64),
        valid,
    )


def _argsort_short_lsd(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = max(32, triton.next_power_of_2(n))
    log_block = block.bit_length() - 1
    warps = 4
    with torch_device_fn.device(inp.device):
        _argsort_short_lsd_local[rows,](
            inp,
            out,
            n,
            shape,
            strides,
            axis_stride,
            out_strides,
            out_axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
        )
    return out


@triton.jit
def _argsort_short_narrow_key(values, DESC: tl.constexpr):
    VALUE_BITS: tl.constexpr = values.dtype.primitive_bitwidth
    tl.static_assert(VALUE_BITS == 8 or VALUE_BITS == 16)
    zero = tl.full((), 0, tl.int32)
    value_mask = tl.full((), (1 << VALUE_BITS) - 1, tl.int32)
    sign_bit = tl.full((), 1 << VALUE_BITS - 1, tl.int32)
    if values.dtype.is_floating():
        tl.static_assert(VALUE_BITS == 16)
        bits = values.to(tl.int16, bitcast=True).to(tl.int32) & value_mask
        magnitude_mask = tl.full((), 32767, tl.int32)
        magnitude = bits & magnitude_mask
        if values.dtype == tl.float16:
            nan_threshold = tl.full((), 31744, tl.int32)
        else:
            tl.static_assert(values.dtype == tl.bfloat16)
            nan_threshold = tl.full((), 32640, tl.int32)
        bits = tl.where(magnitude == zero, zero, bits)
        key = bits ^ tl.where(bits & sign_bit != zero, value_mask, sign_bit)
        key = tl.where(magnitude > nan_threshold, value_mask, key)
    else:
        bits = values.to(tl.int32) & value_mask
        if values.dtype.is_int_signed():
            key = bits ^ sign_bit
        else:
            key = bits
    if DESC:
        key = key ^ value_mask
    return key


@libentry()
@triton.jit
def _argsort_short_narrow_sort(
    inp,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(1 <= N and N <= BLOCK and (BLOCK <= 256))
    tl.static_assert(1 << LOG_BLOCK == BLOCK)
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    input_base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + input_base + lane.to(tl.int64) * AXIS_STRIDE, valid, other=0)
    key = _argsort_short_narrow_key(values, DESC)
    code = key << LOG_BLOCK | lane
    code = tl.where(valid, code, 1 << 24)
    sorted_code = al.sort(code.to(tl.float32), dim=-1, descending=False).to(tl.int32)
    original_index = sorted_code & BLOCK - 1
    output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(
        out + output_base + lane.to(tl.int64) * OUT_AXIS_STRIDE,
        original_index.to(tl.int64),
        valid,
    )


def _argsort_short_narrow(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = max(32, triton.next_power_of_2(n))
    log_block = block.bit_length() - 1
    warps = 4
    with torch_device_fn.device(inp.device):
        _argsort_short_narrow_sort[rows,](
            inp,
            out,
            n,
            shape,
            strides,
            axis_stride,
            out_strides,
            out_axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
        )
    return out


@libentry()
@triton.jit
def _argsort_normal_narrow_sort(
    inp,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(256 < N and N <= BLOCK and (BLOCK <= 4096))
    tl.static_assert(1 << LOG_BLOCK == BLOCK)
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    input_base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + input_base + lane.to(tl.int64) * AXIS_STRIDE, valid, other=0)
    key = _argsort_short_narrow_key(values, DESC)
    code = key << LOG_BLOCK | lane
    padding = tl.full((), 1 << 16 + LOG_BLOCK, tl.int32)
    bias = tl.full((), 8388608, tl.int32)
    code = tl.where(valid, code, padding)
    encoded = (code + bias).to(tl.float32, bitcast=True)
    sorted_code = (
        al.sort(encoded, dim=-1, descending=False).to(tl.int32, bitcast=True) - bias
    )
    original_index = sorted_code & BLOCK - 1
    output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(
        out + output_base + lane.to(tl.int64) * OUT_AXIS_STRIDE,
        original_index.to(tl.int64),
        valid,
    )


@libentry()
@triton.jit
def _argsort_normal_encode(
    inp,
    codes,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + base + lane.to(tl.int64) * AXIS_STRIDE, lane < N, other=0)
    key = _argsort_short_narrow_key(values, DESC)
    code = tl.where(lane < N, key << LOG_BLOCK | lane, 1 << 16 + LOG_BLOCK)
    encoded = (code + 8388608).to(tl.float32, bitcast=True)
    tl.store(codes + row.to(tl.int64) * BLOCK + lane, encoded)


@libentry()
@triton.jit
def _argsort_normal_decode(
    codes,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    encoded = tl.load(codes + row.to(tl.int64) * BLOCK + lane)
    sorted_code = (
        al.sort(encoded, dim=-1, descending=False).to(tl.int32, bitcast=True) - 8388608
    )
    base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(
        out + base + lane.to(tl.int64) * OUT_AXIS_STRIDE,
        (sorted_code & BLOCK - 1).to(tl.int64),
        lane < N,
    )


@libentry()
@triton.jit
def _argsort_scalar_byte_encode(
    inp,
    codes,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
):
    tl.static_assert(1 <= N and N <= BLOCK and (BLOCK <= 4096))
    tl.static_assert(1 << LOG_BLOCK == BLOCK)
    row = tl.program_id(0)
    lane = tl.program_id(1)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    value = tl.load(inp + base + lane.to(tl.int64) * AXIS_STRIDE, lane < N, other=0)
    tl.static_assert(value.dtype.primitive_bitwidth == 8)
    byte_mask = tl.full((), 255, tl.int32)
    sign_bit = tl.full((), 128, tl.int32)
    bits = value.to(tl.int16).to(tl.int32) & byte_mask
    if value.dtype.is_int_signed():
        key = bits ^ sign_bit
    else:
        key = bits
    if DESC:
        key = key ^ byte_mask
    padding = tl.full((), 1 << 16 + LOG_BLOCK, tl.int32)
    bias = tl.full((), 8388608, tl.int32)
    code = tl.where(lane < N, key << LOG_BLOCK | lane, padding)
    encoded = (code + bias).to(tl.float32, bitcast=True)
    tl.store(codes + row.to(tl.int64) * BLOCK + lane, encoded)


def _argsort_normal_narrow(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = max(32, triton.next_power_of_2(n))
    log_block = block.bit_length() - 1
    warps = 4
    with torch_device_fn.device(inp.device):
        if axis_stride != 1 or block == 4096:
            codes = torch.empty((rows, block), dtype=torch.float32, device=inp.device)
            if inp.dtype in (torch.int8, torch.uint8) and axis_stride != 1:
                _argsort_scalar_byte_encode[rows, block](
                    inp,
                    codes,
                    n,
                    shape,
                    strides,
                    axis_stride,
                    block,
                    log_block,
                    descending,
                    num_warps=warps,
                )
            else:
                chunk = 2048 if axis_stride == 1 else 256
                _argsort_normal_encode[rows, block // chunk](
                    inp,
                    codes,
                    n,
                    shape,
                    strides,
                    axis_stride,
                    block,
                    log_block,
                    descending,
                    CHUNK=chunk,
                    num_warps=warps,
                )
            split_output = out_axis_stride != 1
            permutation = (
                torch.empty((rows, block), dtype=torch.int32, device=inp.device)
                if split_output
                else out
            )
            _argsort_normal_decode[rows,](
                codes,
                permutation,
                n,
                (rows,) if split_output else shape,
                (block,) if split_output else out_strides,
                1 if split_output else out_axis_stride,
                block,
                num_warps=warps,
            )
            if split_output:
                inner = 1
                for size in inp.shape[dim + 1 :]:
                    inner *= size
                _argsort_integer_unpack[triton.cdiv(inp.numel(), 256),](
                    permutation,
                    StridedBuffer(
                        out, shape=(out.numel() * 2,), strides=(1,), dtype=torch.int32
                    ),
                    n,
                    block,
                    rows,
                    inner,
                    256,
                    num_warps=warps,
                )
            return out
        _argsort_normal_narrow_sort[rows,](
            inp,
            out,
            n,
            shape,
            strides,
            axis_stride,
            out_strides,
            out_axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
        )
    return out


@libentry()
@triton.jit
def _ascend_fp32_merge_tiles(
    inp,
    keys_out,
    indices_out,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(BLOCK == 1024 and LOG_BLOCK == 10)
    task = tl.program_id(0)
    row = task // TILES
    tile = task - row * TILES
    lane = tl.arange(0, BLOCK)
    start = tile * BLOCK
    col = start + lane
    valid = col < N
    input_base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + input_base + col.to(tl.int64) * AXIS_STRIDE, valid, other=0)
    if values.dtype == tl.int32:
        key = values ^ tl.full((), -2147483648, tl.int32)
        if DESC:
            key = key ^ tl.full((), -1, tl.int32)
    elif values.dtype == tl.float32:
        key = _argsort_short_lsd_fp32_key(values, DESC)
    else:
        key = _argsort_short_narrow_key(values, DESC)
    _lsd_lane = tl.arange(0, BLOCK)
    _lsd_valid = _lsd_lane < tl.minimum(BLOCK, N - start)
    _lsd_word_mask = tl.full((), 65535, tl.int32)
    _lsd_shift = tl.full((), 16, tl.int32)
    _lsd_zero = tl.full((), 0, tl.int32)
    _lsd_sign_bias = tl.full((), -2147483648, tl.int32)
    _lsd_low_word = key & _lsd_word_mask
    _lsd_high_word = key >> _lsd_shift & _lsd_word_mask
    _lsd_low_code = _lsd_low_word << LOG_BLOCK | _lsd_lane
    _lsd_low_code = tl.where(_lsd_valid, _lsd_low_code, 1 << 16 + LOG_BLOCK)
    _lsd_sorted_low = (
        al.sort(
            (_lsd_low_code + 8388608).to(tl.float32, bitcast=True),
            dim=-1,
            descending=False,
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    _lsd_low_permutation = _lsd_sorted_low & BLOCK - 1
    _lsd_current_high = tl.gather(
        _lsd_high_word.to(tl.float32), _lsd_low_permutation, axis=0
    ).to(tl.int32)
    _lsd_high_code = _lsd_current_high << LOG_BLOCK | _lsd_lane
    _lsd_high_code = tl.where(_lsd_valid, _lsd_high_code, 1 << 16 + LOG_BLOCK)
    _lsd_sorted_high = (
        al.sort(
            (_lsd_high_code + 8388608).to(tl.float32, bitcast=True),
            dim=-1,
            descending=False,
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    _lsd_high_permutation = _lsd_sorted_high & BLOCK - 1
    _lsd_permutation = tl.gather(
        _lsd_low_permutation.to(tl.float32), _lsd_high_permutation, axis=0
    ).to(tl.int32)
    _lsd_low_in_current_order = tl.where(
        _lsd_valid, _lsd_sorted_low >> LOG_BLOCK, _lsd_zero
    )
    _lsd_ordered_low = tl.gather(
        _lsd_low_in_current_order.to(tl.float32), _lsd_high_permutation, axis=0
    ).to(tl.int32)
    _lsd_ordered_high = tl.where(_lsd_valid, _lsd_sorted_high >> LOG_BLOCK, _lsd_zero)
    _lsd_ordered_bits = _lsd_ordered_high << _lsd_shift | _lsd_ordered_low
    ordered_key, permutation = (_lsd_ordered_bits ^ _lsd_sign_bias, _lsd_permutation)
    base = row.to(tl.int64) * N
    tl.store(keys_out + base + col, ordered_key, valid)
    tl.store(indices_out + base + col, start + permutation, valid)


@triton.jit
def _ascend_fp32_partition_batch(
    keys,
    cuts,
    N: tl.constexpr,
    RUN: tl.constexpr,
    TILE: tl.constexpr,
    BATCH: tl.constexpr,
    STEPS: tl.constexpr,
):
    row = tl.program_id(1)
    tile = tl.program_id(0) * BATCH + tl.arange(0, BATCH)
    tiles: tl.constexpr = triton.cdiv(N, TILE)
    valid = tile < tiles
    start = tl.minimum(tile * TILE, N)
    pair = start // (2 * RUN) * (2 * RUN)
    a_len = tl.minimum(RUN, N - pair)
    b_start = pair + RUN
    b_len = tl.maximum(0, tl.minimum(RUN, N - b_start))
    diagonal = start - pair
    low = tl.maximum(0, diagonal - b_len)
    high = tl.minimum(diagonal, a_len)
    src = keys + row.to(tl.int64) * N
    for _ in range(STEPS):
        mid = (low + high) // 2
        j = diagonal - mid
        active = valid & (low < high)
        ai = tl.where(active & (mid < a_len), pair + mid, 0)
        bi = tl.where(active & (j > 0), b_start + j - 1, 0)
        av = tl.load(src + ai, active & (mid < a_len), other=0)
        bv = tl.load(src + bi, active & (j > 0), other=0)
        take_a = (j > 0) & (mid < a_len) & (av <= bv)
        low = tl.where(active & take_a, mid + 1, low)
        high = tl.where(active & ~take_a, mid, high)
    tl.store(cuts + row.to(tl.int64) * tiles + tile, low, valid)


@libentry()
@triton.jit
def _ascend_fp32_merge_pass(
    keys_in,
    indices_in,
    cached_cuts,
    keys_out,
    indices_out,
    N: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    FINAL: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(BLOCK == 1024 and LOG_BLOCK == 10)
    tl.static_assert(N <= 262144)
    tiles: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // tiles
    start = (task - row * tiles) * BLOCK
    pair_start = start // (2 * RUN) * (2 * RUN)
    a_len = tl.minimum(RUN, N - pair_start)
    b_start = pair_start + RUN
    b_len = tl.maximum(0, tl.minimum(RUN, N - b_start))
    k0 = start - pair_start
    k1 = tl.minimum(k0 + BLOCK, a_len + b_len)
    base = row.to(tl.int64) * N
    cut_base = row.to(tl.int64) * tiles
    tile_id = task - row * tiles
    a0 = tl.load(cached_cuts + cut_base + tile_id)
    at_pair_end = k1 == a_len + b_len
    a1_cached = tl.load(
        cached_cuts + cut_base + tile_id + 1,
        ~at_pair_end & (tile_id + 1 < tiles),
        other=0,
    )
    a1 = tl.where(at_pair_end, a_len, a1_cached)
    b0 = k0 - a0
    na = a1 - a0
    count = k1 - k0
    lane = tl.arange(0, BLOCK)
    valid = lane < count
    a_valid = lane < na
    b_valid = lane < count - na
    a_key = tl.load(keys_in + base + pair_start + a0 + lane, a_valid, other=0)
    b_key = tl.load(keys_in + base + b_start + b0 + lane, b_valid, other=0)
    a_index = tl.load(indices_in + base + pair_start + a0 + lane, a_valid, other=0)
    b_index = tl.load(indices_in + base + b_start + b0 + lane, b_valid, other=0)
    compact_word_mask = tl.full((), 65535, tl.int32)
    compact_shift = tl.full((), 16, tl.int32)
    b_lane = tl.minimum(tl.maximum(lane - na, 0), BLOCK - 1)
    compact_b_low = tl.gather(
        (b_key & compact_word_mask).to(tl.float32), b_lane, axis=0
    ).to(tl.int32)
    compact_b_high = tl.gather(
        (b_key >> compact_shift & compact_word_mask).to(tl.float32), b_lane, axis=0
    ).to(tl.int32)
    compact_b_key = compact_b_high << compact_shift | compact_b_low
    compact_b_index = tl.gather(b_index.to(tl.float32), b_lane, axis=0).to(tl.int32)
    biased_key = tl.where(a_valid, a_key, compact_b_key)
    original_index = tl.where(a_valid, a_index, compact_b_index)
    sign_bias = tl.full((), -2147483648, tl.int32)
    key = biased_key ^ sign_bias
    _lsd_lane = tl.arange(0, BLOCK)
    _lsd_valid = _lsd_lane < count
    _lsd_word_mask = tl.full((), 65535, tl.int32)
    _lsd_shift = tl.full((), 16, tl.int32)
    _lsd_zero = tl.full((), 0, tl.int32)
    _lsd_sign_bias = tl.full((), -2147483648, tl.int32)
    _lsd_low_word = key & _lsd_word_mask
    _lsd_high_word = key >> _lsd_shift & _lsd_word_mask
    _lsd_low_code = _lsd_low_word << LOG_BLOCK | _lsd_lane
    _lsd_low_code = tl.where(_lsd_valid, _lsd_low_code, 1 << 16 + LOG_BLOCK)
    _lsd_sorted_low = (
        al.sort(
            (_lsd_low_code + 8388608).to(tl.float32, bitcast=True),
            dim=-1,
            descending=False,
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    _lsd_low_permutation = _lsd_sorted_low & BLOCK - 1
    _lsd_current_high = tl.gather(
        _lsd_high_word.to(tl.float32), _lsd_low_permutation, axis=0
    ).to(tl.int32)
    _lsd_high_code = _lsd_current_high << LOG_BLOCK | _lsd_lane
    _lsd_high_code = tl.where(_lsd_valid, _lsd_high_code, 1 << 16 + LOG_BLOCK)
    _lsd_sorted_high = (
        al.sort(
            (_lsd_high_code + 8388608).to(tl.float32, bitcast=True),
            dim=-1,
            descending=False,
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    _lsd_high_permutation = _lsd_sorted_high & BLOCK - 1
    _lsd_permutation = tl.gather(
        _lsd_low_permutation.to(tl.float32), _lsd_high_permutation, axis=0
    ).to(tl.int32)
    _lsd_low_in_current_order = tl.where(
        _lsd_valid, _lsd_sorted_low >> LOG_BLOCK, _lsd_zero
    )
    _lsd_ordered_low = tl.gather(
        _lsd_low_in_current_order.to(tl.float32), _lsd_high_permutation, axis=0
    ).to(tl.int32)
    _lsd_ordered_high = tl.where(_lsd_valid, _lsd_sorted_high >> LOG_BLOCK, _lsd_zero)
    _lsd_ordered_bits = _lsd_ordered_high << _lsd_shift | _lsd_ordered_low
    ordered_key, permutation = (_lsd_ordered_bits ^ _lsd_sign_bias, _lsd_permutation)
    ordered_index = tl.gather(original_index.to(tl.float32), permutation, axis=0).to(
        tl.int32
    )
    col = start + lane
    if FINAL:
        output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
        tl.store(
            indices_out + output_base + col.to(tl.int64) * OUT_AXIS_STRIDE,
            ordered_index.to(tl.int64),
            valid,
        )
    else:
        tl.store(keys_out + base + col, ordered_key, valid)
        tl.store(indices_out + base + col, ordered_index, valid)


def _argsort_ascend_fp32_merge(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    assert 4096 < n <= 262144 and inp.dtype in (
        torch.float32,
        torch.int32,
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int8,
        torch.uint8,
    )
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim)
    out_axis_stride = out.stride(dim)
    block = 1024
    log_block = 10
    tiles = triton.cdiv(n, block)
    keys = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    next_keys = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    next_indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    cached_cuts = torch.empty((rows, tiles), dtype=torch.int32, device=inp.device)
    partition_batch = min(128, triton.next_power_of_2(tiles))
    inner = 1
    for size in inp.shape[dim + 1 :]:
        inner *= size
    warps = 4
    with torch_device_fn.device(inp.device):
        _ascend_fp32_merge_tiles[rows * tiles,](
            inp,
            keys,
            indices,
            n,
            tiles,
            shape,
            strides,
            axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
        )
        run = block
        while run < n:
            final = 2 * run >= n
            _ascend_fp32_partition_batch[triton.cdiv(tiles, partition_batch), rows](
                keys,
                cached_cuts,
                n,
                run,
                block,
                partition_batch,
                (run + 1).bit_length(),
                num_warps=warps,
            )
            _ascend_fp32_merge_pass[rows * tiles,](
                keys,
                indices,
                cached_cuts,
                next_keys,
                out if final and inner == 1 else next_indices,
                n,
                run,
                block,
                log_block,
                (run + 1).bit_length(),
                shape,
                out_strides,
                out_axis_stride,
                final and inner == 1,
                warps,
                num_warps=warps,
            )
            keys, next_keys = (next_keys, keys)
            indices, next_indices = (next_indices, indices)
            run *= 2
        if inner > 1:
            _argsort_integer_unpack[triton.cdiv(inp.numel(), 256),](
                indices,
                StridedBuffer(
                    out, shape=(out.numel() * 2,), strides=(1,), dtype=torch.int32
                ),
                n,
                n,
                rows,
                inner,
                256,
                num_warps=4,
            )
    return out


@libentry()
@triton.jit
def _argsort_row_prepare_keys(
    inp,
    keys,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    PADDED: tl.constexpr,
    DESC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(
        inp + base + column.to(tl.int64) * AXIS_STRIDE, column < N, other=0
    )
    key = _argsort_short_lsd_fp32_key(values, DESC)
    key = tl.where(column < N, key, -1)
    tl.store(keys + row.to(tl.int64) * PADDED + column, key)


@libentry()
@triton.jit
def _argsort_row_lsd_low(
    inp,
    permutation,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(1 <= N and N <= BLOCK and (BLOCK <= 4096))
    tl.static_assert(1 << LOG_BLOCK == BLOCK)
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    key = tl.load(inp + row.to(tl.int64) * BLOCK + lane)
    digit_mask = tl.full((), 65535, tl.int32)
    digit = (key & digit_mask).to(tl.int32)
    code = digit << LOG_BLOCK | lane
    encoded = (code + 8388608).to(tl.float32, bitcast=True)
    sorted_code = al.sort(encoded, dim=-1, descending=False).to(tl.int32, bitcast=True)
    original_index = sorted_code & BLOCK - 1
    tl.store(permutation + row.to(tl.int64) * BLOCK + lane, original_index)


@libentry()
@triton.jit
def _argsort_row_lsd_high(
    inp,
    permutation,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(1 <= N and N <= BLOCK and (BLOCK <= 4096))
    tl.static_assert(1 << LOG_BLOCK == BLOCK)
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    scratch_base = row.to(tl.int64) * BLOCK
    original_index = tl.load(permutation + scratch_base + lane)
    key = tl.load(inp + row.to(tl.int64) * BLOCK + lane)
    digit_shift = tl.full((), 16, tl.int32)
    digit_mask = tl.full((), 65535, tl.int32)
    digit = (key >> digit_shift & digit_mask).to(tl.int32)
    digit = tl.gather(digit.to(tl.float32), original_index, axis=0).to(tl.int32)
    code = digit << LOG_BLOCK | lane
    encoded = (code + 8388608).to(tl.float32, bitcast=True)
    sorted_code = al.sort(encoded, dim=-1, descending=False).to(tl.int32, bitcast=True)
    source_position = sorted_code & BLOCK - 1
    tl.store(out + scratch_base + lane, source_position)


@libentry()
@triton.jit
def _argsort_row_compose(
    permutation,
    positions,
    out,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    scratch_base = row.to(tl.int64) * BLOCK
    position = tl.load(positions + scratch_base + lane)
    original = tl.load(permutation + scratch_base + lane)
    result = tl.gather(original.to(tl.float32), position, axis=0).to(tl.int64)
    base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(out + base + lane.to(tl.int64) * OUT_AXIS_STRIDE, result, valid)


def _argsort_row_lsd(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = max(32, triton.next_power_of_2(n))
    log_block = block.bit_length() - 1
    warps = 4
    permutation = torch.empty((rows, block), dtype=torch.int32, device=inp.device)
    keys = torch.empty((rows, block), dtype=torch.int32, device=inp.device)
    positions = torch.empty((rows, block), dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        _argsort_row_prepare_keys[rows, triton.cdiv(block, 256)](
            inp,
            keys,
            n,
            rows * n,
            shape,
            strides,
            axis_stride,
            block,
            descending,
            256,
            num_warps=4,
            num_stages=1,
        )
        _argsort_row_lsd_low[rows,](
            keys,
            permutation,
            n,
            shape,
            strides,
            axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
            num_stages=1,
        )
        kernel = _argsort_row_lsd_high
        kernel[rows,](
            keys,
            permutation,
            positions,
            n,
            shape,
            strides,
            axis_stride,
            out_strides,
            out_axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
            num_stages=1,
        )
        _argsort_row_compose[rows,](
            permutation,
            positions,
            out,
            n,
            rows * n,
            shape,
            out_strides,
            out_axis_stride,
            block,
            num_warps=4,
            num_stages=1,
        )
    return out


@libentry()
@triton.jit
def _argsort_medium_lsd_local(
    inp,
    out,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(256 < N and N <= BLOCK and (BLOCK <= 1024))
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + base + lane.to(tl.int64) * AXIS_STRIDE, valid, other=0)
    key = _argsort_short_lsd_fp32_key(values, DESC)
    DIGIT_BITS: tl.constexpr = 16
    permutation = lane
    for step in tl.static_range(triton.cdiv(32, DIGIT_BITS)):
        digit = (
            key >> step * DIGIT_BITS
            & (1 << min(DIGIT_BITS, 32 - step * DIGIT_BITS)) - 1
        ).to(tl.float32)
        if step == 0:
            current_digit = digit.to(tl.int32)
        else:
            current_digit = tl.gather(digit, permutation, axis=0).to(tl.int32)
        code = current_digit << LOG_BLOCK | lane
        code = tl.where(valid, code, 1 << 16 + LOG_BLOCK)
        encoded = (code + 8388608).to(tl.float32, bitcast=True)
        sorted_code = (
            al.sort(encoded, dim=-1, descending=False).to(tl.int32, bitcast=True)
            - 8388608
        )
        current_lane = sorted_code & BLOCK - 1
        if step == 0:
            permutation = current_lane
        else:
            permutation = tl.gather(
                permutation.to(tl.float32), current_lane, axis=0
            ).to(tl.int32)
    output_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(
        out + output_base + lane.to(tl.int64) * OUT_AXIS_STRIDE,
        permutation.to(tl.int64),
        valid,
    )


def _argsort_medium_lsd(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    block = max(32, triton.next_power_of_2(n))
    log_block = block.bit_length() - 1
    warps = 4
    with torch_device_fn.device(inp.device):
        _argsort_medium_lsd_local[rows,](
            inp,
            out,
            n,
            shape,
            strides,
            axis_stride,
            out_strides,
            out_axis_stride,
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
        )
    return out


@triton.jit
def _paired_row_offset(row, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full(row.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        if axis == 0:
            coord = row
        else:
            coord = row % SHAPE[axis]
            row = row // SHAPE[axis]
        offset += coord.to(tl.int64) * STRIDES[axis]
    return offset


@_paired_sort_jit
def _argsort_paired_rows(
    inp,
    values,
    out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
    DESC: tl.constexpr,
    PACKED: tl.constexpr,
):
    row = tl.program_id(0) * M + tl.arange(0, M)
    col = tl.arange(0, B)
    mask = (row[:, None] < ROWS) & (col[None, :] < N)
    if PACKED:
        x = tl.load(
            inp + row[:, None].to(tl.int64) * B + col[None, :],
            row[:, None] < ROWS,
            other=0,
        )
    else:
        base = _paired_row_offset(row, SHAPE, STRIDES)
        x = tl.load(
            inp + base[:, None] + col[None, :].to(tl.int64) * AXIS_STRIDE, mask, other=0
        )
    x = tl.where(x != x, float("nan"), x)
    sorted_values = al.sort(x, dim=-1, descending=DESC)
    tl.store(
        values + row[:, None] * B + col[None, :], sorted_values, row[:, None] < ROWS
    )
    output_base = _paired_row_offset(row, SHAPE, OUT_STRIDES)
    tl.store(
        out + output_base[:, None] + col[None, :].to(tl.int64) * OUT_AXIS_STRIDE,
        sorted_values.to(tl.int32, bitcast=True).to(tl.int64),
        mask,
    )


@triton.jit
def _argsort_paired_pack(
    inp,
    packed,
    N: tl.constexpr,
    B: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    DESC: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * 256 + tl.arange(0, 256)
    base = _paired_row_offset(row, SHAPE, STRIDES)
    x = tl.load(inp + base + col.to(tl.int64) * AXIS_STRIDE, col < N, other=0)
    x = tl.where(x != x, float("nan"), x)
    x = tl.where(col < N, x, float("-inf") if DESC else float("nan"))
    tl.store(packed + row.to(tl.int64) * B + col, x)


@triton.jit
def _argsort_paired_unpack(
    indices,
    out,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    INNER: tl.constexpr,
):
    flat = tl.program_id(0).to(tl.int64) * 256 + tl.arange(0, 256)
    row = flat // (N * INNER) * INNER + flat % INNER
    col = flat // INNER % N
    value = tl.load(indices + row * B + col, flat < TOTAL, other=0)
    tl.store(out + flat, value, flat < TOTAL)


def _argsort_paired(inp, dim, descending):
    dim %= inp.ndim
    n = inp.shape[dim]
    rows = inp.numel() // n
    block = triton.next_power_of_2(n)
    batch = (
        4
        if block <= 1024 and n == block and dim == inp.ndim - 1 and inp.is_contiguous()
        else 1
    )
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    values = torch.empty((rows, block), dtype=torch.float32, device=inp.device)
    shape = tuple(s for i, s in enumerate(inp.shape) if i != dim)
    strides = tuple(s for i, s in enumerate(inp.stride()) if i != dim)
    out_strides = tuple(s for i, s in enumerate(out.stride()) if i != dim)
    unpack = dim != inp.ndim - 1
    indices = (
        torch.empty((rows, block), dtype=torch.int64, device=inp.device)
        if unpack
        else out
    )
    kernel_shape = (rows,) if unpack else shape
    kernel_out_strides = (block,) if unpack else out_strides
    kernel_out_axis_stride = 1 if unpack else out.stride(dim)
    inner = 1
    for size in inp.shape[dim + 1 :]:
        inner *= size
    use_packed = n != block or not inp.is_contiguous() or dim != inp.ndim - 1
    packed = (
        torch.empty((rows, block), dtype=torch.float32, device=inp.device)
        if use_packed
        else inp
    )
    with torch_device_fn.device(inp.device):
        if use_packed:
            _argsort_paired_pack[(rows, triton.cdiv(block, 256))](
                inp,
                packed,
                n,
                block,
                shape,
                strides,
                inp.stride(dim),
                descending,
                num_stages=1,
            )
        _argsort_paired_rows[(triton.cdiv(rows, batch),)](
            packed,
            values,
            indices,
            n,
            rows,
            kernel_shape,
            strides,
            kernel_out_strides,
            inp.stride(dim),
            kernel_out_axis_stride,
            block,
            batch,
            descending,
            use_packed,
            num_stages=1,
        )
        if unpack:
            _argsort_paired_unpack[(triton.cdiv(inp.numel(), 256),)](
                indices, out, inp.numel(), n, block, inner, num_stages=1
            )
    return out


@triton.jit
def _argsort_integer_pack(
    inp,
    words,
    N: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    WORDS: tl.constexpr,
    AXIS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DESC: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    quotient = row
    base = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        base += (quotient % SHAPE[axis]) * STRIDES[axis]
        quotient = quotient // SHAPE[axis]
    lane = (tl.program_id(1) * T + tl.arange(0, T)).to(tl.int64)
    ptr32 = inp
    element = base + lane * AXIS
    low = tl.load(ptr32 + element * (WORDS // 2), lane < N, other=0)
    if WORDS == 4:
        high = tl.load(ptr32 + element * 2 + 1, lane < N, other=0)
    for word in tl.static_range(WORDS):
        if word < 2:
            digit = (low >> (16 * (word % 2))) & 65535
        else:
            digit = (high >> (16 * (word % 2))) & 65535
        if word == WORDS - 1:
            digit = digit ^ 32768
        digit = tl.where(lane < N, digit.to(tl.float32), -1.0 if DESC else 65536.0)
        tl.store(words + (word * ROWS + row) * B + lane, digit, lane < B)


@PairedSortJITFunction
def _argsort_integer_pass(
    words,
    previous,
    out,
    N: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    WORD: tl.constexpr,
    FIRST: tl.constexpr,
    DESC: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    keys = tl.load(words + (WORD * ROWS + row) * B + lane)
    if FIRST:
        permutation = lane
    else:
        permutation = tl.load(previous + row * B + lane)
    keys = tl.gather(keys, permutation, 0)
    # Distinct finite positive float encodings make tie order explicit.
    # DESC reverses digit order while retaining increasing current position.
    tie = B - 1 - lane if DESC else lane
    bits = 0x3F000000 + keys.to(tl.int32) * B + tie
    ordered = al.sort(bits.to(tl.float32, bitcast=True), dim=-1, descending=DESC)
    order = ordered.to(tl.int32, bitcast=True)
    permutation = tl.gather(permutation.to(tl.float32), order, 0).to(tl.int32)
    tl.store(out + row * B + lane, permutation)


@triton.jit
def _argsort_integer_unpack(
    inp,
    out,
    N: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    INNER: tl.constexpr,
    T: tl.constexpr,
):
    if ROWS * B * 2 >= 2147483648:
        base = tl.program_id(0).to(tl.int64) * T
    else:
        base = tl.program_id(0) * T
    flat = base + tl.arange(0, T)
    row = flat // (N * INNER) * INNER + flat % INNER
    col = flat // INNER % N
    index = tl.load(inp + row * B + col, flat < ROWS * N, other=0)
    values = tl.reshape(tl.join(index, tl.full((T,), 0, tl.int32)), (T * 2,))
    slot = base * 2 + tl.arange(0, T * 2)
    tl.store(out + slot, values, slot < ROWS * N * 2)


@triton.jit
def _argsort_integer_zero(out, TOTAL: tl.constexpr):
    flat = tl.program_id(0).to(tl.int64) * 256 + tl.arange(0, 256)
    tl.store(out + flat, 0, flat < TOTAL)


def _argsort_integer_paired(x, dim=-1, descending=False):
    if x.dtype not in (torch.int32, torch.int64):
        raise TypeError("integer paired sort supports int32/int64 only")
    rank = x.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    out = torch.empty(x.shape, device=x.device, dtype=torch.int64)
    if x.numel() == 0:
        return out
    axis = dim % max(rank, 1)
    n = x.shape[axis] if rank else 1
    if n == 1:
        with torch_device_fn.device(x.device):
            _argsort_integer_zero[(triton.cdiv(x.numel(), 256),)](
                out, x.numel(), num_warps=4
            )
        return out
    if n > 4096:
        raise NotImplementedError("integer paired sort requires N<=4096")
    rows = x.numel() // n
    b = triton.next_power_of_2(n)
    words = x.element_size() // 2
    shape = tuple(s for i, s in enumerate(x.shape) if i != axis)
    strides = tuple(s for i, s in enumerate(x.stride()) if i != axis)
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    scratch = torch.empty((words, rows, b), device=x.device, dtype=torch.float32)
    p = torch.empty((rows, b), device=x.device, dtype=torch.int32)
    q = torch.empty_like(p)
    with torch_device_fn.device(x.device):
        _argsort_integer_pack[(rows, triton.cdiv(b, 256))](
            StridedBuffer(
                x,
                shape=(
                    (1 + sum((s - 1) * t for s, t in zip(x.shape, x.stride())))
                    * (words // 2),
                ),
                strides=(1,),
                dtype=torch.int32,
            ),
            scratch,
            n,
            b,
            rows,
            words,
            x.stride(axis) if rank else 1,
            shape,
            strides,
            descending,
            256,
            num_warps=4,
        )
        for word in range(words):
            _argsort_integer_pass[(rows,)](
                scratch,
                p,
                q,
                n,
                b,
                rows,
                word,
                word == 0,
                descending,
                num_warps=1,
                num_stages=1,
                multibuffer=False,
            )
            p, q = q, p
        _argsort_integer_unpack[(triton.cdiv(rows * n, 256),)](
            p,
            StridedBuffer(
                out, shape=(out.numel() * 2,), strides=(1,), dtype=torch.int32
            ),
            n,
            b,
            rows,
            inner,
            256,
            num_warps=4,
        )
    return out


@libentry()
@triton.jit
def _argsort_narrow_tiles(
    inp,
    values_out,
    indices_out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    FINAL: tl.constexpr,
    PACKED: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    WARPS: tl.constexpr,
):
    # Signature-compatible initial-run replacement. The original merge follows.
    tl.static_assert(BLOCK == 256 and LOG_BLOCK == 8 and ROW_BLOCK == 1)
    tl.static_assert(not FINAL and not PACKED)
    chunks: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // chunks
    chunk = task - row * chunks
    lane = tl.arange(0, BLOCK)
    col = chunk * BLOCK + lane
    valid = (row < ROWS) & (col < N)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    values = tl.load(inp + base + col.to(tl.int64) * AXIS_STRIDE, valid, other=0)
    key = _argsort_short_narrow_key(values, DESC)
    # Largest valid code is 2**24-1. The exact FP32 padding code sorts last
    # in either direction because DESC is encoded in key, not sort direction.
    code = tl.where(valid, (key << 8) | lane, 1 << 24)
    sorted_code = al.sort(code.to(tl.float32), dim=-1, descending=False).to(tl.int32)
    permutation = sorted_code & 255
    # Gather integer bit patterns via exact FP32 integers, never convert the
    # original floating value. Preserve subnormals, signed zero and NaN payload.
    BITS: tl.constexpr = values.dtype.primitive_bitwidth
    if BITS == 16:
        raw = values.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    else:
        tl.static_assert(BITS == 8)
        raw = values.to(tl.int8, bitcast=True).to(tl.int32) & 255
    sorted_raw = tl.gather(raw.to(tl.float32), permutation, axis=0).to(tl.int32)
    if BITS == 16:
        sorted_values = sorted_raw.to(tl.int16).to(values.dtype, bitcast=True)
    else:
        sorted_values = sorted_raw.to(tl.int8).to(values.dtype, bitcast=True)
    scratch_offset = row.to(tl.int64) * N + col
    tl.store(values_out + scratch_offset, sorted_values, valid)
    tl.store(indices_out + scratch_offset, chunk * BLOCK + permutation, valid)


@libentry()
@triton.jit
def _argsort_scalar_byte_pack(
    inp,
    packed,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
):
    # Scalar source load avoids the strided byte-vector layout expansion.
    row = tl.program_id(0)
    col = tl.program_id(1)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    value = tl.load(inp + base + col.to(tl.int64) * AXIS_STRIDE, col < N, other=0)
    tl.static_assert(value.dtype.primitive_bitwidth == 8)
    # Numeric widening sign-extends int8 and zero-extends uint8, both exactly.
    tl.store(packed + row.to(tl.int64) * N + col, value.to(tl.int16), col < N)


def _argsort_merge_entry(inp, dim=-1, descending=False):
    """Stable indices using bounded tiles and launch-separated merge passes."""
    logger.debug("GEMS ARGSORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    if inp.numel() == 0:
        return out
    dim = dim % max(rank, 1)
    n = inp.shape[dim] if rank else 1
    if n >= 2**31:
        raise NotImplementedError("argsort supports fewer than 2**31 elements per row")
    rows = inp.numel() // n
    shape = tuple((s for i, s in enumerate(inp.shape) if i != dim))
    strides = tuple((s for i, s in enumerate(inp.stride()) if i != dim))
    out_strides = tuple((s for i, s in enumerate(out.stride()) if i != dim))
    axis_stride = inp.stride(dim) if rank else 1
    out_axis_stride = out.stride(dim) if rank else 1
    if inp.dtype == torch.int64:
        block_limit = 32
    elif inp.dtype == torch.int32:
        block_limit = 64
    else:
        block_limit = 256
    block = min(triton.next_power_of_2(n), block_limit)
    row_block = max(1, 128 // block)
    index_bits = (n - 1).bit_length()
    use_packed = False
    if use_packed:
        block = min(triton.next_power_of_2(n), 1024)
    row_block = max(1, 128 // block)
    merge_block = min(block, 1024)
    tile_warps = 4
    merge_warps = 4
    packed_warps = 4
    pair_merge_kernel = _argsort_merge
    with torch_device_fn.device(inp.device):
        if n <= block:
            _argsort_tiles[triton.cdiv(rows, row_block),](
                inp,
                out,
                out,
                n,
                rows,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                block.bit_length() - 1,
                row_block,
                descending,
                True,
                use_packed,
                index_bits,
                tile_warps,
                num_warps=tile_warps,
            )
        elif use_packed:
            bits = inp.element_size() * 8 + index_bits
            key_dtype = torch.int32 if bits <= 32 else torch.int64
            keys = torch.empty((rows, n), dtype=key_dtype, device=inp.device)
            next_keys = torch.empty_like(keys)
            _argsort_tiles[rows * triton.cdiv(n, block),](
                inp,
                keys,
                out,
                n,
                rows,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                block.bit_length() - 1,
                1,
                descending,
                False,
                True,
                index_bits,
                tile_warps,
                num_warps=tile_warps,
            )
            packed_merge_kernel = _packed_merge
            run = block
            while run < n:
                final = run * 2 >= n
                packed_merge_kernel[rows * triton.cdiv(n, merge_block),](
                    keys,
                    next_keys,
                    out,
                    n,
                    run,
                    merge_block,
                    (run + 1).bit_length(),
                    merge_block.bit_length() - 1,
                    shape,
                    out_strides,
                    out_axis_stride,
                    index_bits,
                    final,
                    packed_warps,
                    num_warps=packed_warps,
                )
                keys, next_keys = (next_keys, keys)
                run *= 2
        else:
            long_byte = n > 4096 and inp.dtype in (torch.int8, torch.uint8)
            if long_byte:
                values = torch.empty((rows, n), dtype=torch.int16, device=inp.device)
            else:
                values = torch.empty((rows, n), dtype=inp.dtype, device=inp.device)
            indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
            if long_byte:
                values_tmp = torch.empty(
                    (rows, n), dtype=torch.int16, device=inp.device
                )
            else:
                values_tmp = torch.empty_like(values)
            indices_tmp = torch.empty_like(indices)
            initial_inp = inp
            initial_shape = shape
            initial_strides = strides
            initial_axis_stride = axis_stride
            if long_byte and axis_stride != 1:
                initial_inp = torch.empty(
                    (rows, n), dtype=torch.int16, device=inp.device
                )
                _argsort_scalar_byte_pack[rows, n](
                    inp, initial_inp, n, shape, strides, axis_stride, num_warps=4
                )
                initial_shape = (rows,)
                initial_strides = (n,)
                initial_axis_stride = 1
            initial_tiles = (
                _argsort_narrow_tiles
                if n > 4096
                and inp.dtype
                in (
                    torch.float16,
                    torch.bfloat16,
                    torch.int16,
                    torch.int8,
                    torch.uint8,
                )
                else _argsort_tiles
            )
            initial_tiles[rows * triton.cdiv(n, block),](
                initial_inp,
                values,
                indices,
                n,
                rows,
                initial_shape,
                initial_strides,
                initial_axis_stride,
                out_strides,
                out_axis_stride,
                block,
                block.bit_length() - 1,
                1,
                descending,
                False,
                use_packed,
                index_bits,
                tile_warps,
                num_warps=tile_warps,
            )
            run = block
            while run < n:
                final = run * 2 >= n
                pair_merge_kernel[rows * triton.cdiv(n, merge_block),](
                    values,
                    indices,
                    values_tmp,
                    out if final else indices_tmp,
                    n,
                    run,
                    merge_block,
                    (run + 1).bit_length(),
                    merge_block.bit_length() - 1,
                    shape,
                    out_strides,
                    out_axis_stride,
                    descending,
                    final,
                    merge_warps,
                    num_warps=merge_warps,
                )
                values, values_tmp = (values_tmp, values)
                indices, indices_tmp = (indices_tmp, indices)
                run *= 2
    return out


def _argsort_short_output(x, dim=-1, descending=False):
    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        raise TypeError("short output candidate supports FP16/BF16/FP32/I16/I8/U8")
    rank = x.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    out = torch.empty(x.shape, device=x.device, dtype=torch.int64)
    if x.numel() == 0:
        return out
    axis = dim % max(rank, 1)
    n = x.shape[axis] if rank else 1
    if n > 256:
        raise NotImplementedError("short output candidate requires N<=256")
    rows = x.numel() // n
    shape = tuple((s for i, s in enumerate(x.shape) if i != axis))
    strides = tuple((s for i, s in enumerate(x.stride()) if i != axis))
    axis_stride = x.stride(axis) if rank else 1
    block = max(32, triton.next_power_of_2(n))
    log_block = block.bit_length() - 1
    permutation = torch.empty((rows, block), dtype=torch.int32, device=x.device)
    row_stride = block
    output_strides = []
    for size in reversed(shape):
        output_strides.append(row_stride)
        row_stride *= size
    output_strides = tuple(reversed(output_strides))
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    warps = 4
    with torch_device_fn.device(x.device):
        if x.dtype in (torch.int8, torch.uint8) and axis_stride != 1:
            codes = torch.empty((rows, block), dtype=torch.float32, device=x.device)
            _argsort_scalar_byte_encode[rows, block](
                x,
                codes,
                n,
                shape,
                strides,
                axis_stride,
                block,
                log_block,
                descending,
                num_warps=warps,
            )
            _argsort_normal_decode[rows,](
                codes, permutation, n, shape, output_strides, 1, block, num_warps=warps
            )
        else:
            kernel = (
                _argsort_short_lsd_local
                if x.dtype == torch.float32
                else _argsort_short_narrow_sort
            )
            kernel[rows,](
                x,
                permutation,
                n,
                shape,
                strides,
                axis_stride,
                output_strides,
                1,
                block,
                log_block,
                descending,
                warps,
                num_warps=warps,
            )
        _argsort_integer_unpack[triton.cdiv(x.numel(), 256),](
            permutation,
            StridedBuffer(
                out, shape=(out.numel() * 2,), strides=(1,), dtype=torch.int32
            ),
            n,
            block,
            rows,
            inner,
            256,
            num_warps=warps,
        )
    return out


@libentry()
@triton.jit
def _argsort_int64_merge_tiles(
    inp32,
    high_out,
    low_out,
    indices_out,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESC: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(BLOCK == 1024 and LOG_BLOCK == 10)
    tl.static_assert(N <= 262144)
    task = tl.program_id(0)
    row = task // TILES
    start = (task - row * TILES) * BLOCK
    lane = tl.arange(0, BLOCK)
    col = start + lane
    valid = col < N
    input_base = _argsort_row_offset(row, SHAPE, STRIDES)
    element = input_base + col.to(tl.int64) * AXIS_STRIDE
    raw_low = tl.load(inp32 + element * 2, valid, other=0)
    raw_high = tl.load(inp32 + element * 2 + 1, valid, other=0)
    key_high = raw_high
    key_low = raw_low ^ tl.full((), -2147483648, tl.int32)
    if DESC:
        key_high = key_high ^ tl.full((), -1, tl.int32)
        key_low = key_low ^ tl.full((), -1, tl.int32)
    word_mask = tl.full((), 65535, tl.int32)
    sign = tl.full((), -2147483648, tl.int32)
    low_bits = key_low ^ sign
    high_bits = key_high ^ sign
    word0 = (low_bits & word_mask).to(tl.float32)
    word1 = ((low_bits >> 16) & word_mask).to(tl.float32)
    word2 = (high_bits & word_mask).to(tl.float32)
    word3 = ((high_bits >> 16) & word_mask).to(tl.float32)
    code = (word0.to(tl.int32) << LOG_BLOCK) | lane
    code = tl.where(valid, code, 1 << (16 + LOG_BLOCK))
    sorted_code = (
        al.sort(
            (code + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    permutation = sorted_code & (BLOCK - 1)
    digit1 = tl.gather(word1, permutation, axis=0).to(tl.int32)
    code1 = (digit1 << LOG_BLOCK) | lane
    code1 = tl.where(valid, code1, 1 << (16 + LOG_BLOCK))
    sorted_code1 = (
        al.sort(
            (code1 + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    order1 = sorted_code1 & (BLOCK - 1)
    permutation = tl.gather(permutation.to(tl.float32), order1, axis=0).to(tl.int32)
    digit2 = tl.gather(word2, permutation, axis=0).to(tl.int32)
    code2 = (digit2 << LOG_BLOCK) | lane
    code2 = tl.where(valid, code2, 1 << (16 + LOG_BLOCK))
    sorted_code2 = (
        al.sort(
            (code2 + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    order2 = sorted_code2 & (BLOCK - 1)
    permutation = tl.gather(permutation.to(tl.float32), order2, axis=0).to(tl.int32)
    digit3 = tl.gather(word3, permutation, axis=0).to(tl.int32)
    code3 = (digit3 << LOG_BLOCK) | lane
    code3 = tl.where(valid, code3, 1 << (16 + LOG_BLOCK))
    sorted_code3 = (
        al.sort(
            (code3 + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    order3 = sorted_code3 & (BLOCK - 1)
    permutation = tl.gather(permutation.to(tl.float32), order3, axis=0).to(tl.int32)
    ordered_word0 = tl.gather(word0, permutation, axis=0).to(tl.int32)
    ordered_word1 = tl.gather(word1, permutation, axis=0).to(tl.int32)
    ordered_word2 = tl.gather(word2, permutation, axis=0).to(tl.int32)
    ordered_word3 = tl.where(valid, sorted_code3 >> LOG_BLOCK, 0)
    ordered_low = ((ordered_word1 << 16) | ordered_word0) ^ sign
    ordered_high = ((ordered_word3 << 16) | ordered_word2) ^ sign
    base = row.to(tl.int64) * N
    tl.store(high_out + base + col, ordered_high, valid)
    tl.store(low_out + base + col, ordered_low, valid)
    tl.store(indices_out + base + col, start + permutation, valid)


@triton.jit
def _argsort_int64_partition_batch(
    high,
    low_words,
    cuts,
    N: tl.constexpr,
    RUN: tl.constexpr,
    TILE: tl.constexpr,
    BATCH: tl.constexpr,
    STEPS: tl.constexpr,
):
    row = tl.program_id(1)
    tile = tl.program_id(0) * BATCH + tl.arange(0, BATCH)
    tiles: tl.constexpr = triton.cdiv(N, TILE)
    valid = tile < tiles
    start = tl.minimum(tile * TILE, N)
    pair = start // (2 * RUN) * (2 * RUN)
    a_len = tl.minimum(RUN, N - pair)
    b_start = pair + RUN
    b_len = tl.maximum(0, tl.minimum(RUN, N - b_start))
    diagonal = start - pair
    low = tl.maximum(0, diagonal - b_len)
    upper = tl.minimum(diagonal, a_len)
    base = row.to(tl.int64) * N
    for _ in range(STEPS):
        mid = (low + upper) // 2
        j = diagonal - mid
        active = valid & (low < upper)
        ai = tl.where(active & (mid < a_len), pair + mid, 0)
        bi = tl.where(active & (j > 0), b_start + j - 1, 0)
        ah = tl.load(high + base + ai, active & (mid < a_len), other=0)
        bh = tl.load(high + base + bi, active & (j > 0), other=0)
        alow = tl.load(low_words + base + ai, active & (mid < a_len), other=0)
        blow = tl.load(low_words + base + bi, active & (j > 0), other=0)
        before_or_equal = (ah < bh) | ((ah == bh) & (alow <= blow))
        take_a = (j > 0) & (mid < a_len) & before_or_equal
        low = tl.where(active & take_a, mid + 1, low)
        upper = tl.where(active & ~take_a, mid, upper)
    tl.store(cuts + row.to(tl.int64) * tiles + tile, low, valid)


@libentry()
@triton.jit
def _argsort_int64_merge_pass(
    high_in,
    low_in,
    indices_in,
    cached_cuts,
    high_out,
    low_out,
    indices_out,
    N: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    WARPS: tl.constexpr,
):
    tl.static_assert(BLOCK == 1024 and LOG_BLOCK == 10)
    tl.static_assert(N <= 262144)
    tiles: tl.constexpr = triton.cdiv(N, BLOCK)
    task = tl.program_id(0)
    row = task // tiles
    start = (task - row * tiles) * BLOCK
    pair_start = start // (2 * RUN) * (2 * RUN)
    a_len = tl.minimum(RUN, N - pair_start)
    b_start = pair_start + RUN
    b_len = tl.maximum(0, tl.minimum(RUN, N - b_start))
    k0 = start - pair_start
    k1 = tl.minimum(k0 + BLOCK, a_len + b_len)
    base = row.to(tl.int64) * N
    cut_base = row.to(tl.int64) * tiles
    tile_id = task - row * tiles
    a0 = tl.load(cached_cuts + cut_base + tile_id)
    at_pair_end = k1 == a_len + b_len
    a1_cached = tl.load(
        cached_cuts + cut_base + tile_id + 1,
        ~at_pair_end & (tile_id + 1 < tiles),
        other=0,
    )
    a1 = tl.where(at_pair_end, a_len, a1_cached)
    b0 = k0 - a0
    na = a1 - a0
    count = k1 - k0
    lane = tl.arange(0, BLOCK)
    valid = lane < count
    a_valid = lane < na
    b_valid = lane < count - na
    ah = tl.load(high_in + base + pair_start + a0 + lane, a_valid, other=0)
    bh = tl.load(high_in + base + b_start + b0 + lane, b_valid, other=0)
    alow = tl.load(low_in + base + pair_start + a0 + lane, a_valid, other=0)
    blow = tl.load(low_in + base + b_start + b0 + lane, b_valid, other=0)
    ai = tl.load(indices_in + base + pair_start + a0 + lane, a_valid, other=0)
    bi = tl.load(indices_in + base + b_start + b0 + lane, b_valid, other=0)
    compact_mask = tl.full((), 65535, tl.int32)
    b_lane = tl.minimum(tl.maximum(lane - na, 0), BLOCK - 1)
    compact_h0 = tl.gather((bh & compact_mask).to(tl.float32), b_lane, axis=0).to(
        tl.int32
    )
    compact_h1 = tl.gather(
        ((bh >> 16) & compact_mask).to(tl.float32), b_lane, axis=0
    ).to(tl.int32)
    compact_l0 = tl.gather((blow & compact_mask).to(tl.float32), b_lane, axis=0).to(
        tl.int32
    )
    compact_l1 = tl.gather(
        ((blow >> 16) & compact_mask).to(tl.float32), b_lane, axis=0
    ).to(tl.int32)
    compact_index = tl.gather(bi.to(tl.float32), b_lane, axis=0).to(tl.int32)
    key_high = tl.where(a_valid, ah, (compact_h1 << 16) | compact_h0)
    key_low = tl.where(a_valid, alow, (compact_l1 << 16) | compact_l0)
    original_index = tl.where(a_valid, ai, compact_index)
    word_mask = tl.full((), 65535, tl.int32)
    sign = tl.full((), -2147483648, tl.int32)
    low_bits = key_low ^ sign
    high_bits = key_high ^ sign
    word0 = (low_bits & word_mask).to(tl.float32)
    word1 = ((low_bits >> 16) & word_mask).to(tl.float32)
    word2 = (high_bits & word_mask).to(tl.float32)
    word3 = ((high_bits >> 16) & word_mask).to(tl.float32)
    code = (word0.to(tl.int32) << LOG_BLOCK) | lane
    code = tl.where(valid, code, 1 << (16 + LOG_BLOCK))
    sorted_code = (
        al.sort(
            (code + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    permutation = sorted_code & (BLOCK - 1)
    digit1 = tl.gather(word1, permutation, axis=0).to(tl.int32)
    code1 = (digit1 << LOG_BLOCK) | lane
    code1 = tl.where(valid, code1, 1 << (16 + LOG_BLOCK))
    sorted_code1 = (
        al.sort(
            (code1 + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    order1 = sorted_code1 & (BLOCK - 1)
    permutation = tl.gather(permutation.to(tl.float32), order1, axis=0).to(tl.int32)
    digit2 = tl.gather(word2, permutation, axis=0).to(tl.int32)
    code2 = (digit2 << LOG_BLOCK) | lane
    code2 = tl.where(valid, code2, 1 << (16 + LOG_BLOCK))
    sorted_code2 = (
        al.sort(
            (code2 + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    order2 = sorted_code2 & (BLOCK - 1)
    permutation = tl.gather(permutation.to(tl.float32), order2, axis=0).to(tl.int32)
    digit3 = tl.gather(word3, permutation, axis=0).to(tl.int32)
    code3 = (digit3 << LOG_BLOCK) | lane
    code3 = tl.where(valid, code3, 1 << (16 + LOG_BLOCK))
    sorted_code3 = (
        al.sort(
            (code3 + 8388608).to(tl.float32, bitcast=True), dim=-1, descending=False
        ).to(tl.int32, bitcast=True)
        - 8388608
    )
    order3 = sorted_code3 & (BLOCK - 1)
    permutation = tl.gather(permutation.to(tl.float32), order3, axis=0).to(tl.int32)
    ordered_word0 = tl.gather(word0, permutation, axis=0).to(tl.int32)
    ordered_word1 = tl.gather(word1, permutation, axis=0).to(tl.int32)
    ordered_word2 = tl.gather(word2, permutation, axis=0).to(tl.int32)
    ordered_word3 = tl.where(valid, sorted_code3 >> LOG_BLOCK, 0)
    ordered_low = ((ordered_word1 << 16) | ordered_word0) ^ sign
    ordered_high = ((ordered_word3 << 16) | ordered_word2) ^ sign
    ordered_index = tl.gather(original_index.to(tl.float32), permutation, axis=0).to(
        tl.int32
    )
    col = start + lane
    tl.store(high_out + base + col, ordered_high, valid)
    tl.store(low_out + base + col, ordered_low, valid)
    tl.store(indices_out + base + col, ordered_index, valid)


def _argsort_int64_merge(inp, dim, descending):
    rank = inp.ndim
    dim %= max(rank, 1)
    n = inp.shape[dim] if rank else 1
    assert inp.dtype == torch.int64 and 4096 < n <= 262144
    rows = inp.numel() // n
    shape = tuple(s for i, s in enumerate(inp.shape) if i != dim)
    strides = tuple(s for i, s in enumerate(inp.stride()) if i != dim)
    inner = 1
    for size in inp.shape[dim + 1 :]:
        inner *= size
    out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    high = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    low = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    next_high = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    next_low = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    next_indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    block, log_block, warps = 1024, 10, 4
    tiles = triton.cdiv(n, block)
    cached_cuts = torch.empty((rows, tiles), dtype=torch.int32, device=inp.device)
    batch = min(128, triton.next_power_of_2(tiles))
    with torch_device_fn.device(inp.device):
        _argsort_int64_merge_tiles[rows * tiles,](
            StridedBuffer(
                inp,
                shape=(
                    (1 + sum((s - 1) * t for s, t in zip(inp.shape, inp.stride()))) * 2,
                ),
                strides=(1,),
                dtype=torch.int32,
            ),
            high,
            low,
            indices,
            n,
            tiles,
            shape,
            strides,
            inp.stride(dim),
            block,
            log_block,
            descending,
            warps,
            num_warps=warps,
        )
        run = block
        while run < n:
            _argsort_int64_partition_batch[triton.cdiv(tiles, batch), rows](
                high,
                low,
                cached_cuts,
                n,
                run,
                block,
                batch,
                (run + 1).bit_length(),
                num_warps=warps,
            )
            _argsort_int64_merge_pass[rows * tiles,](
                high,
                low,
                indices,
                cached_cuts,
                next_high,
                next_low,
                next_indices,
                n,
                run,
                block,
                log_block,
                warps,
                num_warps=warps,
            )
            high, next_high = next_high, high
            low, next_low = next_low, low
            indices, next_indices = next_indices, indices
            run *= 2
        _argsort_integer_unpack[triton.cdiv(inp.numel(), 256),](
            indices,
            StridedBuffer(
                out, shape=(out.numel() * 2,), strides=(1,), dtype=torch.int32
            ),
            n,
            n,
            rows,
            inner,
            256,
            num_warps=4,
        )
    return out


# ASC source is kept with the backend implementation. Its hardware algorithm is
# the validated stable four-way merge; Triton supplies the original launch ABI.
_ASC_SORT_SOURCE = r"""#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include "kernel_operator.h"

// Standalone dav-2201 diagnostic. GM proposals contain {float score, uint32 index}.
// All three kernels use the same ACL stream. This is not a FlagGems implementation.
constexpr uint32_t V179_RUN = 4096;
constexpr uint32_t V179_WINDOW = 2048;
constexpr uint32_t V179_STATUS_WORDS = 64; // Isolate per-block scalar GM status writes.

__aicore__ inline uint32_t V179Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline float V179FloatBits(uint32_t bits)
{
    union Bits { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}

__aicore__ inline void V179ScalarToVector()
{
    // The installed CreateVecIndex implementation uses this same S -> V fence.
    auto event = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
}

__aicore__ inline void V179Record(GM_ADDR status, uint32_t blocks,
    uint32_t code, uint32_t tasks, uint32_t rounds, uint32_t badTask)
{
    AscendC::GlobalTensor<uint32_t> meta;
    meta.SetGlobalBuffer((__gm__ uint32_t*)status, blocks * V179_STATUS_WORDS);
    const uint32_t base = AscendC::GetBlockIdx() * V179_STATUS_WORDS;
    meta.SetValue(base, code);
    meta.SetValue(base + 1, tasks);
    meta.SetValue(base + 2, rounds);
    meta.SetValue(base + 3, badTask);
}

class V179Initial {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR status,
        uint32_t rows, uint32_t n, uint32_t descending, uint32_t blocks,
        AscendC::TPipe* pipe)
    {
        input_ = input;
        output_ = output;
        status_ = status;
        rows_ = rows; n_ = n; descending_ = descending; blocks_ = blocks;
        // Values and generated indices share one 32 KiB input allocation.
        pipe->InitBuffer(in_, 1, V179_RUN * 8);
        pipe->InitBuffer(out_, 1, V179_RUN * 8);
        pipe->InitBuffer(scratch_, 1, V179_RUN * 8);
        pipe->InitBuffer(mask_, 1, V179_RUN / 8);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tiles = (n_ + V179_RUN - 1) / V179_RUN;
        const uint32_t total = rows_ * tiles;
        uint32_t tasks = 0;
        for (uint32_t task = AscendC::GetBlockIdx(); task < total; task += blocks_) {
            const uint32_t row = task / tiles;
            const uint32_t start = (task % tiles) * V179_RUN;
            const uint32_t count = V179Min(V179_RUN, n_ - start);
            const uint64_t base = uint64_t(row) * uint64_t(n_) + uint64_t(start);
            AscendC::GlobalTensor<float> source;
            source.SetGlobalBuffer((__gm__ float*)input_ + base, count);
            auto input = in_.AllocTensor<float>();
            AscendC::DataCopyPad(input, source,
                {1, static_cast<uint16_t>(count * 4), 0, 0}, {false, 0, 0, 0});
            in_.EnQue(input);
            input = in_.DeQue<float>();
            AscendC::PipeBarrier<PIPE_ALL>();

            // Set every invalid UB lane explicitly after DMA. No assumption is
            // made about DataCopyPad's bytes beyond an unaligned true length.
            auto raw = input.ReinterpretCast<uint32_t>();
            const uint32_t pad = descending_ ? 0xff800000u : 0x7fc00000u;
            for (uint32_t lane = count; lane < V179_RUN; ++lane) raw.SetValue(lane, pad);
            V179ScalarToVector();
            auto mask = mask_.AllocTensor<uint8_t>();
            AscendC::Compare(mask, input, input, AscendC::CMPMODE::EQ, V179_RUN);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(input, mask, input, V179FloatBits(0x7fc00000u),
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, V179_RUN);
            AscendC::PipeBarrier<PIPE_V>();
            auto indices = input.ReinterpretCast<int32_t>()[V179_RUN];
            AscendC::CreateVecIndex(indices, static_cast<int32_t>(start), V179_RUN);
            AscendC::PipeBarrier<PIPE_V>();
            if (!descending_) {
                auto scoreBits = input.ReinterpretCast<int32_t>();
                // Integer Adds wraps the sign bit exactly, as in Sort1D.cpp.
                AscendC::Adds(scoreBits, scoreBits, int32_t(-2147483647 - 1), int32_t(V179_RUN));
                AscendC::PipeBarrier<PIPE_V>();
            }

            auto output = out_.AllocTensor<float>();
            auto scratch = scratch_.AllocTensor<float>();
            __ubuf__ float* src = (__ubuf__ float*)input.GetPhyAddr();
            __ubuf__ float* dst = (__ubuf__ float*)output.GetPhyAddr();
            __ubuf__ float* tmp = (__ubuf__ float*)scratch.GetPhyAddr();
            // These raw sort stages match the measured V175 R4096 component.
            vbitsort(dst, src, (__ubuf__ uint32_t*)(src + V179_RUN), 128);
            pipe_barrier(PIPE_V);
            __ubuf__ float* current = dst;
            __ubuf__ float* next = tmp;
            for (uint64_t factor = 32; factor <= 512; factor *= 4) {
                __ubuf__ float* lists[4] = {current, current + 2 * factor,
                    current + 4 * factor, current + 6 * factor};
                const uint64_t lengths = factor | (factor << 16) | (factor << 32) | (factor << 48);
                const uint64_t repeats = V179_RUN / (factor * 4);
                vmrgsort4(next, lists, lengths, uint64_t(0xf00) | repeats);
                pipe_barrier(PIPE_V);
                __ubuf__ float* swap = current; current = next; next = swap;
            }
            __ubuf__ float* finalLists[4] = {current, current + 4096, current, current};
            vmrgsort4(dst, finalLists, uint64_t(2048) | (uint64_t(2048) << 16), uint64_t(0x301));
            pipe_barrier(PIPE_V);
            scratch_.FreeTensor(scratch);
            mask_.FreeTensor(mask);
            out_.EnQue(output);
            in_.FreeTensor(input);
            output = out_.DeQue<float>();
            AscendC::GlobalTensor<float> destination;
            destination.SetGlobalBuffer((__gm__ float*)output_ + uint64_t(2) * base,
                uint64_t(2) * count);
            // Real equal sentinels precede padding. Discard all invalid payloads.
            AscendC::DataCopyPad(destination, output,
                {1, static_cast<uint16_t>(count * 8), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            out_.FreeTensor(output);
            ++tasks;
        }
        V179Record(status_, blocks_, 0, tasks, 0, 0xffffffffu);
    }
private:
    GM_ADDR input_;
    GM_ADDR output_;
    GM_ADDR status_;
    uint32_t rows_, n_, descending_, blocks_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_, scratch_, mask_;
};

class V190Initial {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR status,
        uint32_t rows, uint32_t n, uint32_t descending, uint32_t blocks, uint32_t kind,
        AscendC::TPipe* pipe)
    {
        input_ = input;
        output_ = output;
        status_ = status;
        rows_ = rows; n_ = n; descending_ = descending; blocks_ = blocks; kind_ = kind;
        // Values and generated indices share one 32 KiB input allocation.
        pipe->InitBuffer(in_, 1, V179_RUN * 8);
        pipe->InitBuffer(out_, 1, V179_RUN * 8);
        pipe->InitBuffer(scratch_, 1, V179_RUN * 8);
        pipe->InitBuffer(mask_, 1, V179_RUN / 8);
        pipe->InitBuffer(narrow_, 1, V179_RUN * 2);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tiles = (n_ + V179_RUN - 1) / V179_RUN;
        const uint32_t total = rows_ * tiles;
        uint32_t tasks = 0;
        for (uint32_t task = AscendC::GetBlockIdx(); task < total; task += blocks_) {
            const uint32_t row = task / tiles;
            const uint32_t start = (task % tiles) * V179_RUN;
            const uint32_t count = V179Min(V179_RUN, n_ - start);
            const uint64_t base = uint64_t(row) * uint64_t(n_) + uint64_t(start);
            AscendC::GlobalTensor<uint16_t> source;
            source.SetGlobalBuffer((__gm__ uint16_t*)input_ + base, count);
            auto narrow = narrow_.AllocTensor<uint16_t>();
            AscendC::DataCopyPad(narrow, source,
                {1, static_cast<uint16_t>(count * 2), 0, 0}, {false, 0, 0, 0});
            narrow_.EnQue(narrow);
            narrow = narrow_.DeQue<uint16_t>();
            auto input = in_.AllocTensor<float>();
            // Real SDK count Cast; raw input remains 16-bit until this point.
            if (kind_ == 0) {
                auto typed = narrow.ReinterpretCast<half>();
                AscendC::Cast(input, typed, AscendC::RoundMode::CAST_NONE, count);
            } else {
                auto typed = narrow.ReinterpretCast<bfloat16_t>();
                AscendC::Cast(input, typed, AscendC::RoundMode::CAST_NONE, count);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            narrow_.FreeTensor(narrow);
            in_.EnQue(input);
            input = in_.DeQue<float>();
            AscendC::PipeBarrier<PIPE_ALL>();

            // Set every invalid UB lane explicitly after DMA. No assumption is
            // made about DataCopyPad's bytes beyond an unaligned true length.
            auto raw = input.ReinterpretCast<uint32_t>();
            const uint32_t pad = descending_ ? 0xff800000u : 0x7fc00000u;
            for (uint32_t lane = count; lane < V179_RUN; ++lane) raw.SetValue(lane, pad);
            V179ScalarToVector();
            auto mask = mask_.AllocTensor<uint8_t>();
            AscendC::Compare(mask, input, input, AscendC::CMPMODE::EQ, V179_RUN);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(input, mask, input, V179FloatBits(0x7fc00000u),
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, V179_RUN);
            AscendC::PipeBarrier<PIPE_V>();
            auto indices = input.ReinterpretCast<int32_t>()[V179_RUN];
            AscendC::CreateVecIndex(indices, static_cast<int32_t>(start), V179_RUN);
            AscendC::PipeBarrier<PIPE_V>();
            if (!descending_) {
                auto scoreBits = input.ReinterpretCast<int32_t>();
                // Integer Adds wraps the sign bit exactly, as in Sort1D.cpp.
                AscendC::Adds(scoreBits, scoreBits, int32_t(-2147483647 - 1), int32_t(V179_RUN));
                AscendC::PipeBarrier<PIPE_V>();
            }

            auto output = out_.AllocTensor<float>();
            auto scratch = scratch_.AllocTensor<float>();
            __ubuf__ float* src = (__ubuf__ float*)input.GetPhyAddr();
            __ubuf__ float* dst = (__ubuf__ float*)output.GetPhyAddr();
            __ubuf__ float* tmp = (__ubuf__ float*)scratch.GetPhyAddr();
            // These raw sort stages match the measured V175 R4096 component.
            vbitsort(dst, src, (__ubuf__ uint32_t*)(src + V179_RUN), 128);
            pipe_barrier(PIPE_V);
            __ubuf__ float* current = dst;
            __ubuf__ float* next = tmp;
            for (uint64_t factor = 32; factor <= 512; factor *= 4) {
                __ubuf__ float* lists[4] = {current, current + 2 * factor,
                    current + 4 * factor, current + 6 * factor};
                const uint64_t lengths = factor | (factor << 16) | (factor << 32) | (factor << 48);
                const uint64_t repeats = V179_RUN / (factor * 4);
                vmrgsort4(next, lists, lengths, uint64_t(0xf00) | repeats);
                pipe_barrier(PIPE_V);
                __ubuf__ float* swap = current; current = next; next = swap;
            }
            __ubuf__ float* finalLists[4] = {current, current + 4096, current, current};
            vmrgsort4(dst, finalLists, uint64_t(2048) | (uint64_t(2048) << 16), uint64_t(0x301));
            pipe_barrier(PIPE_V);
            scratch_.FreeTensor(scratch);
            mask_.FreeTensor(mask);
            out_.EnQue(output);
            in_.FreeTensor(input);
            output = out_.DeQue<float>();
            AscendC::GlobalTensor<float> destination;
            destination.SetGlobalBuffer((__gm__ float*)output_ + uint64_t(2) * base,
                uint64_t(2) * count);
            // Real equal sentinels precede padding. Discard all invalid payloads.
            AscendC::DataCopyPad(destination, output,
                {1, static_cast<uint16_t>(count * 8), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            out_.FreeTensor(output);
            ++tasks;
        }
        V179Record(status_, blocks_, 0, tasks, 0, 0xffffffffu);
    }
private:
    GM_ADDR input_;
    GM_ADDR output_;
    GM_ADDR status_;
    uint32_t rows_, n_, descending_, blocks_, kind_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_, narrow_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_, scratch_, mask_;
};

class V187StreamingMerge4 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR status,
        uint32_t rows, uint32_t n, uint32_t run, uint32_t blocks, uint32_t finalOutput,
        AscendC::TPipe* pipe)
    {
        input_ = input;
        output_ = output;
        status_ = status;
        rows_ = rows; n_ = n; run_ = run; blocks_ = blocks; finalOutput_ = finalOutput;
        for (uint32_t slot = 0; slot < 4; ++slot)
            pipe->InitBuffer(inputs_[slot], 1, V179_WINDOW * 8 + 32);
        pipe->InitBuffer(merged_, 1, 4 * V179_WINDOW * 8 + 32);
        pipe->InitBuffer(decoded_, 1, V179_WINDOW * 8);
        pipe->InitBuffer(offsets_, 1, V179_WINDOW * 8);
        // Explicit UB: 4*(16384+32)+(65536+32)+16384+16384 = 164000 B.
    }

    __aicore__ inline void Emit(AscendC::LocalTensor<float>& proposals,
        uint32_t count, uint64_t base, const AscendC::LocalTensor<uint32_t>& offsets)
    {
        // Only called after all comparisons for this window have completed.
        auto words = proposals.ReinterpretCast<uint32_t>();
        for (uint32_t start = 0; start < count; start += V179_WINDOW) {
            const uint32_t chunk = V179Min(V179_WINDOW, count - start);
            AscendC::GlobalTensor<uint32_t> destination;
            destination.SetGlobalBuffer((__gm__ uint32_t*)output_ + uint64_t(2) * (base + start),
                uint64_t(2) * chunk);
            auto window = words[2 * start];
            if (finalOutput_) {
                // The next chunk's score word is dead: later gathers read only
                // payload words. The last full chunk uses the extra 32 B slot.
                // Never make this write when producing a non-final sorted run.
                AscendC::PipeBarrier<PIPE_ALL>();
                window.SetValue(2 * V179_WINDOW, uint32_t(0));
                V179ScalarToVector();
                auto decoded = decoded_.AllocTensor<uint32_t>();
                AscendC::Gather(decoded, window, offsets, uint32_t(0), uint32_t(2 * chunk));
                AscendC::PipeBarrier<PIPE_V>();
                decoded_.EnQue(decoded);
                decoded = decoded_.DeQue<uint32_t>();
                AscendC::DataCopyPad(destination, decoded,
                    {1, static_cast<uint16_t>(chunk * 8), 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
                decoded_.FreeTensor(decoded);
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyPad(destination, window,
                    {1, static_cast<uint16_t>(chunk * 8), 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t groups = (n_ + 4 * run_ - 1) / (4 * run_);
        const uint32_t totalTasks = rows_ * groups;
        const uint32_t block = AscendC::GetBlockIdx();
        if (block >= totalTasks) {
            V179Record(status_, blocks_, 0, 0, 0, 0xffffffffu);
            return;
        }
        auto offsets = offsets_.AllocTensor<uint32_t>();
        if (finalOutput_) {
            for (uint32_t lane = 0; lane < V179_WINDOW; ++lane) {
                offsets.SetValue(2 * lane, 8 * lane + 4);
                offsets.SetValue(2 * lane + 1, V179_WINDOW * 8);
            }
            V179ScalarToVector();
        }
        uint32_t status = 0, tasks = 0, totalRounds = 0, badTask = 0xffffffffu;
        for (uint32_t task = block; task < totalTasks; task += blocks_) {
            const uint32_t row = task / groups;
            const uint32_t start = (task % groups) * (4 * run_);
            const uint64_t base = uint64_t(row) * uint64_t(n_) + uint64_t(start);
            uint32_t lengths[4] = {0, 0, 0, 0};
            uint32_t position[4] = {0, 0, 0, 0};
            uint32_t totalLength = 0;
            for (uint32_t original = 0; original < 4; ++original) {
                const uint32_t relativeStart = original * run_;
                if (relativeStart < n_ - start)
                    lengths[original] = V179Min(run_, n_ - start - relativeStart);
                totalLength += lengths[original];
            }
            uint32_t written = 0, rounds = 0;
            while (status == 0 && written < totalLength) {
                if (rounds >= totalLength) { status = 8; break; }
                uint32_t consumedTotal = 0;
                for (uint32_t original = 0; original < 4; ++original) {
                    if (position[original] > lengths[original]) status |= 4;
                    consumedTotal += position[original];
                }
                if (consumedTotal != written) status |= 4;
                if (status != 0) break;

                // The hardware mask is a contiguous prefix (3, 7, or 15).
                // Preserve original run order when compacting live pointers.
                uint32_t live[4] = {0, 0, 0, 0};
                uint32_t loaded[4] = {0, 0, 0, 0};
                uint32_t active = 0, totalLoaded = 0;
                for (uint32_t original = 0; original < 4; ++original) {
                    const uint32_t remaining = lengths[original] - position[original];
                    if (remaining != 0) {
                        live[active] = original;
                        loaded[active] = V179Min(V179_WINDOW, remaining);
                        totalLoaded += loaded[active];
                        ++active;
                    }
                }
                if (active == 0) { status = 2; break; }
                AscendC::LocalTensor<float> local[4];
                for (uint32_t slot = 0; slot < active; ++slot) {
                    const uint32_t original = live[slot];
                    const uint64_t sourceIndex = base + uint64_t(original) * uint64_t(run_) +
                        uint64_t(position[original]);
                    AscendC::GlobalTensor<float> source;
                    source.SetGlobalBuffer((__gm__ float*)input_ + uint64_t(2) * sourceIndex,
                        uint64_t(2) * loaded[slot]);
                    local[slot] = inputs_[slot].AllocTensor<float>();
                    AscendC::DataCopyPad(local[slot], source,
                        {1, static_cast<uint16_t>(loaded[slot] * 8), 0, 0}, {false, 0, 0, 0});
                    inputs_[slot].EnQue(local[slot]);
                }
                for (uint32_t slot = 0; slot < active; ++slot)
                    local[slot] = inputs_[slot].DeQue<float>();
                AscendC::PipeBarrier<PIPE_ALL>();

                uint16_t consumed[4] = {0, 0, 0, 0};
                uint32_t count = 0;
                if (active == 1) {
                    // All other GLOBAL runs are exhausted. No further compare.
                    count = loaded[0];
                    consumed[0] = static_cast<uint16_t>(count);
                    Emit(local[0], count, base + written, offsets);
                } else {
                    auto output = merged_.AllocTensor<float>();
                    __ubuf__ float* first = (__ubuf__ float*)local[0].GetPhyAddr();
                    __ubuf__ float* lists[4] = {first, first, first, first};
                    uint64_t packedLengths = 0;
                    for (uint32_t slot = 0; slot < active; ++slot) {
                        lists[slot] = (__ubuf__ float*)local[slot].GetPhyAddr();
                        packedLengths |= uint64_t(loaded[slot]) << (16 * slot);
                    }
                    const uint64_t config = active == 4 ? uint64_t(0x1f01) :
                        (active == 3 ? uint64_t(0x1701) : uint64_t(0x1301));
                    vmrgsort4((__ubuf__ float*)output.GetPhyAddr(), lists, packedLengths, config);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    consumed[0] = consumed[1] = consumed[2] = consumed[3] = 0xffff;
                    AscendC::GetMrgSortResult(consumed[0], consumed[1], consumed[2], consumed[3]);
                    bool exhausted = false;
                    for (uint32_t slot = 0; slot < 4; ++slot) {
                        if (slot < active) {
                            if (consumed[slot] > loaded[slot]) status |= 1;
                            count += consumed[slot];
                            exhausted = exhausted || consumed[slot] == loaded[slot];
                        } else if (consumed[slot] != 0) status |= 1;
                    }
                    if (count == 0) status |= 2;
                    if (count > totalLoaded || count > totalLength - written) status |= 4;
                    if (!exhausted) status |= 32;
                    merged_.EnQue(output);
                    output = merged_.DeQue<float>();
                    if (status == 0) Emit(output, count, base + written, offsets);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    merged_.FreeTensor(output);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t slot = 0; slot < active; ++slot)
                    inputs_[slot].FreeTensor(local[slot]);
                ++rounds;
                if (status != 0) break;
                for (uint32_t slot = 0; slot < active; ++slot)
                    position[live[slot]] += consumed[slot];
                written += count;
            }
            totalRounds += rounds;
            if (status == 0) {
                if (written != totalLength) status = 4;
                for (uint32_t original = 0; original < 4; ++original)
                    if (position[original] != lengths[original]) status |= 4;
            }
            if (status != 0) { badTask = task; break; }
            ++tasks;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        offsets_.FreeTensor(offsets);
        V179Record(status_, blocks_, status, tasks, totalRounds, badTask);
    }
private:
    GM_ADDR input_;
    GM_ADDR output_;
    GM_ADDR status_;
    uint32_t rows_, n_, run_, blocks_, finalOutput_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputs_[4];
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> merged_, decoded_, offsets_;
};

extern "C" __global__ __vector__ void v202_fp32_initial_a(
    GM_ADDR ffts,
    GM_ADDR lock,
    GM_ADDR workspace,
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR status,
    uint32_t rows, uint32_t n, uint32_t descending, uint32_t blocks,
    uint32_t gridX, uint32_t gridY, uint32_t gridZ)
{
    (void)ffts; (void)lock; (void)workspace;
    if (gridX != blocks || gridY != 1 || gridZ != 1) return;
    AscendC::TPipe pipe;
    V179Initial op;
    op.Init(input, output, status, rows, n, descending, blocks, &pipe);
    op.Process();
}

extern "C" __global__ __vector__ void v202_raw16_initial_a(
    GM_ADDR ffts,
    GM_ADDR lock,
    GM_ADDR workspace,
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR status,
    uint32_t rows, uint32_t n, uint32_t descending, uint32_t blocks, uint32_t kind,
    uint32_t gridX, uint32_t gridY, uint32_t gridZ)
{
    (void)ffts; (void)lock; (void)workspace;
    if (gridX != blocks || gridY != 1 || gridZ != 1) return;
    AscendC::TPipe pipe;
    V190Initial op;
    op.Init(input, output, status, rows, n, descending, blocks, kind, &pipe);
    op.Process();
}

extern "C" __global__ __vector__ void v202_merge4_a(
    GM_ADDR ffts,
    GM_ADDR lock,
    GM_ADDR workspace,
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR status,
    uint32_t rows, uint32_t n, uint32_t run, uint32_t blocks, uint32_t finalOutput,
    uint32_t gridX, uint32_t gridY, uint32_t gridZ)
{
    (void)ffts; (void)lock; (void)workspace;
    if (gridX != blocks || gridY != 1 || gridZ != 1) return;
    AscendC::TPipe pipe;
    V187StreamingMerge4 op;
    op.Init(input, output, status, rows, n, run, blocks, finalOutput, &pipe);
    op.Process();
}

// Compile-time references only. Python uses the extracted device ELF and the
// original installed Triton loader/launcher; main never launches these stubs.
extern "C" void ascend_argsort_compile_anchor(aclrtStream stream, uint8_t* ffts,
    uint8_t* lock, uint8_t* workspace, uint8_t* input, uint8_t* output, uint8_t* status)
{
    v202_fp32_initial_a<<<1, nullptr, stream>>>(
        ffts, lock, workspace, input, output, status, 1, 8193, 0, 1, 1, 1, 1);
    v202_raw16_initial_a<<<1, nullptr, stream>>>(
        ffts, lock, workspace, input, output, status, 1, 8193, 0, 1, 0, 1, 1, 1);
    v202_merge4_a<<<1, nullptr, stream>>>(
        ffts, lock, workspace, input, output, status, 1, 8193, 4096, 1, 1, 1, 1, 1);
}

int main()
{
    std::fputs("Ascend argsort: build artifact only.\n", stderr);
    return 2;
}
"""
_ASC_SORT_CMAKE = r"""cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(ascend_argsort_runtime LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(ascend_argsort_runtime op_host/sort.asc)
target_include_directories(ascend_argsort_runtime PRIVATE
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
target_link_directories(ascend_argsort_runtime PRIVATE
    "$ENV{ASCEND_HOME_PATH}/lib64"
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64")
target_link_libraries(ascend_argsort_runtime PRIVATE
    tiling_api register platform unified_dlog dl m graph_base ascendcl ascendc_runtime)
target_compile_options(ascend_argsort_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
"""
_ASC_SORT_SIGNATURES = {
    "v202_fp32_initial_a": ("*fp32", "*fp32", "*u32", "u32", "u32", "u32", "u32"),
    "v202_raw16_initial_a": (
        "*u16",
        "*fp32",
        "*u32",
        "u32",
        "u32",
        "u32",
        "u32",
        "u32",
    ),
    "v202_merge4_a": ("*fp32", "*u32", "*u32", "u32", "u32", "u32", "u32", "u32"),
}
_ASC_SORT_RUNTIMES = {}
_ASC_SORT_LOCK = threading.RLock()
_ASC_SORT_ENV = (
    "ASCEND_HOME_PATH",
    "TRITON_ASCEND_ARCH",
    "TRITON_BACKEND",
    "TRITON_DISABLE_FFTS",
    "TRITON_ENABLE_TASKQUEUE",
    "TRITON_COMPILE_ONLY",
    "TRITON_DEVICE_PRINT",
    "TRITON_REGISTER_TENSOR_MSPROF",
    "TRITON_ENABLE_LIBDEVICE_SIMT",
    "TRITON_ALL_BLOCKS_PARALLEL",
)


def _asc_sort_require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _asc_sort_metadata(name, source_hash):
    return SimpleNamespace(
        arch="Ascend910B1",
        debug=False,
        mix_mode="aiv",
        parallel_mode="simd",
        force_simt_only=False,
        compile_on_910_95=False,
        shared=1,
        shared_mem_dynamic_size=221184,
        bs_task_type=10,
        enable_auto_blockify=None,
        name=name,
        kernel_name=name,
        hash=source_hash,
        workspace_size=0,
        lock_num=0,
        lock_init_value=0,
    )


def _asc_sort_check_launcher(source, signature):
    # Only the actual argument contract is checked; no source dump or package
    # hash whitelist is needed. Unknown layouts remain on the Triton route.
    bodies = re.findall(
        r"struct __attribute__\(\(packed\)\)\s*\{(.*?)\}\s*args\s*=\s*\{",
        source,
        flags=re.S,
    )
    _asc_sort_require(len(bodies) == 1, "unrecognized launcher argument struct")
    pattern = (
        r"(void\s*\*|int32_t|uint32_t)\s+(\w+)\s+__attribute__\(\(aligned\((\d+)\)\)\);"
    )
    fields = [
        (ty.replace(" ", ""), name, int(align))
        for ty, name, align in re.findall(pattern, bodies[0])
    ]
    expected = [
        ("void*", name, 8) for name in ("ffts_addr", "syncBlockLock", "workspace_addr")
    ]
    expected += [
        (
            "void*" if ty.startswith("*") else "uint32_t",
            f"arg{i}",
            8 if ty.startswith("*") else 4,
        )
        for i, ty in enumerate(signature)
    ]
    expected += [("int32_t", name, 4) for name in ("gridX", "gridY", "gridZ")]
    _asc_sort_require(fields == expected, "unsupported launcher argument ABI")
    _asc_sort_require(
        not re.sub(pattern, "", bodies[0]).strip(), "extra launcher fields"
    )
    compact = re.sub(r"\s+", "", source)
    _asc_sort_require(
        "rtGetC2cCtrlAddr" in source
        and "rtKernelLaunch(func,blockNum,static_cast<void*>(&args),sizeof(args),NULL,stream);"
        in compact
        and "ret=rtKernelLaunchWithFlagV2(" not in compact,
        "unsupported runtime launch contract",
    )


def _asc_sort_device_binary(fat_object):
    # Only extract the linked device section from the compiler's ELF64 fat
    # object. The diagnostic symbol/metadata disassembler is intentionally absent.
    _asc_sort_require(
        len(fat_object) >= 64 and fat_object[:6] == b"\x7fELF\x02\x01",
        "ASC did not produce an ELF64 little-endian object",
    )
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", fat_object)
    offset, entry_size, count, names_index = h[6], h[11], h[12], h[13]
    _asc_sort_require(
        h[1:3] == (1, 183)
        and entry_size == 64
        and 0 < names_index < count
        and offset + count * entry_size <= len(fat_object),
        "unsupported ASC fat object section table",
    )
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", fat_object, offset + i * entry_size)
        for i in range(count)
    ]
    names = sections[names_index]
    _asc_sort_require(names[4] + names[5] <= len(fat_object), "truncated ELF names")
    strings = fat_object[names[4] : names[4] + names[5]]
    matches = []
    for section in sections:
        start = section[0]
        end = strings.find(b"\0", start)
        _asc_sort_require(
            0 <= start < len(strings) and end >= start, "invalid ELF name"
        )
        if strings[start:end] == b".aicore_binary":
            _asc_sort_require(
                section[1] != 8 and section[4] + section[5] <= len(fat_object),
                "truncated device binary",
            )
            matches.append(fat_object[section[4] : section[4] + section[5]])
    _asc_sort_require(len(matches) == 1, "missing or ambiguous device binary")
    data = matches[0]
    _asc_sort_require(
        len(data) >= 64
        and data[:6] == b"\x7fELF\x02\x01"
        and struct.unpack_from("<HH", data, 16) == (2, 4137),
        "unsupported linked AIV ELF",
    )
    return data


def _asc_sort_build(identity, home, bisheng):
    import fcntl

    from triton.runtime.cache import get_cache_manager

    from flag_gems.utils.code_cache import code_cache_dir
    from flag_gems.utils.code_utils import write_atomic

    identity_text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(identity_text.encode()).hexdigest()
    cache = get_cache_manager(key)
    cached = cache.get_file("ascend_argsort.elf")
    if cached is not None:
        return Path(cached).read_bytes()
    build_root = code_cache_dir() / "ascend_argsort" / key
    build_root.mkdir(parents=True, exist_ok=True)
    with (build_root / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached = cache.get_file("ascend_argsort.elf")
            if cached is not None:
                return Path(cached).read_bytes()
            write_atomic(str(build_root / "identity.json"), identity_text)
            with tempfile.TemporaryDirectory(prefix="build-", dir=build_root) as temp:
                source = Path(temp)
                (source / "op_host").mkdir()
                write_atomic(str(source / "op_host/sort.asc"), _ASC_SORT_SOURCE)
                write_atomic(str(source / "CMakeLists.txt"), _ASC_SORT_CMAKE)
                build = source / "build"
                env = dict(os.environ, ASCEND_HOME_PATH=str(home))
                commands = (
                    [
                        identity["cmake"],
                        "-S",
                        str(source),
                        "-B",
                        str(build),
                        "-DCMAKE_ASC_COMPILER=" + str(bisheng),
                        "-DASC_DIR=" + str(home / "aarch64-linux/lib64/cmake"),
                    ],
                    [identity["cmake"], "--build", str(build), "-j2"],
                )
                for command in commands:
                    result = subprocess.run(
                        command,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=180,
                    )
                    if result.returncode:
                        raise RuntimeError("ASC build failed: " + result.stdout[-4000:])
                fat = build / "CMakeFiles/ascend_argsort_runtime.dir/op_host/sort.asc.o"
                binary = _asc_sort_device_binary(fat.read_bytes())
                # Publish only a complete, structurally checked successful build.
                cache.put(binary, "ascend_argsort.elf", binary=True)
                return binary
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class _AscSortRuntime:
    def __init__(self, device):
        from triton.backends.ascend import driver, utils

        _asc_sort_require(
            platform.machine() == "aarch64",
            "requires the validated AArch64 ASC toolchain",
        )
        _asc_sort_require(
            torch.npu.get_device_name(device) == "Ascend910B1", "unvalidated NPU model"
        )
        _asc_sort_require(
            triton.__version__.split("+", 1)[0] == "3.5.1"
            and torch.__version__.split("+", 1)[0].startswith("2.10."),
            "unvalidated torch/Triton release",
        )
        arch = utils.get_ascend_arch_from_env()
        _asc_sort_require(arch in ("", "Ascend910B1"), "different compiler target")
        _asc_sort_require(
            utils.is_ffts_supported(arch) and not utils.force_disable_ffts(),
            "FFTS is unavailable",
        )
        _asc_sort_require(
            os.getenv("TRITON_BACKEND", "torch_npu") == "torch_npu",
            "different runtime backend",
        )
        for name in (
            "TRITON_COMPILE_ONLY",
            "TRITON_DEVICE_PRINT",
            "TRITON_REGISTER_TENSOR_MSPROF",
        ):
            _asc_sort_require(
                os.getenv(name, "false").lower() not in ("true", "1"),
                "unsupported " + name,
            )
        _asc_sort_require(
            not os.getenv("TRITON_ENABLE_LIBDEVICE_SIMT", ""),
            "SIMT override is enabled",
        )
        _asc_sort_require(
            os.getenv("TRITON_ENABLE_TASKQUEUE", "true").lower() in ("true", "1"),
            "taskqueue is disabled",
        )
        home_value = os.getenv("ASCEND_HOME_PATH", "")
        _asc_sort_require(bool(home_value), "ASCEND_HOME_PATH is unset")
        home = Path(home_value).resolve(strict=True)
        info_path = home / "aarch64-linux/ascend_toolkit_install.info"
        if not info_path.is_file():
            info_path = home / "aarch64-linux/ascend_all_cann_install.info"
        info = info_path.read_text()
        fields = dict(
            (key.strip(), value.strip())
            for line in info.splitlines()
            if "=" in line
            for key, value in [line.split("=", 1)]
        )
        _asc_sort_require(
            fields.get("version") == "9.0.0"
            and fields.get("innerversion") == "V100R001C10SPC001B250"
            and fields.get("arch") == "aarch64",
            "unvalidated CANN SDK build",
        )
        bisheng = (home / "bin/bisheng").resolve(strict=True)
        _asc_sort_require(
            bisheng.is_file() and os.access(bisheng, os.X_OK), "missing ASC compiler"
        )
        cmake = shutil.which("cmake")
        _asc_sort_require(cmake is not None, "cmake is unavailable")
        self.driver = driver.NPUDriver()
        self.loader = self.driver.utils
        _asc_sort_require(
            self.driver.get_current_target().arch == "Ascend910B1",
            "runtime target mismatch",
        )
        self.aiv = int(self.loader.get_aivector_core_num())
        _asc_sort_require(0 < self.aiv < 65536, "invalid vector core count")
        source_hash = hashlib.sha256(_ASC_SORT_SOURCE.encode()).hexdigest()
        specifications = {}
        wrapper_hashes = {}
        for name, signature in _ASC_SORT_SIGNATURES.items():
            metadata = _asc_sort_metadata(name, source_hash)
            src = SimpleNamespace(signature=dict(enumerate(signature)), constants={})
            wrapper = driver.make_launcher(src.constants, src.signature, metadata)
            _asc_sort_check_launcher(wrapper, signature)
            specifications[name] = (src, metadata)
            wrapper_hashes[name] = hashlib.sha256(wrapper.encode()).hexdigest()
        identity = dict(
            schema="ascend-argsort-aiv-ffts-80-v1",
            source=source_hash,
            cmake_source=hashlib.sha256(_ASC_SORT_CMAKE.encode()).hexdigest(),
            arch="dav-2201",
            sdk_home=str(home),
            sdk_info=hashlib.sha256(info.encode()).hexdigest(),
            compiler=str(bisheng),
            compiler_sha256=utils.get_file_hash256(str(bisheng)),
            cmake=str(Path(cmake).resolve()),
            torch=torch.__version__,
            triton=triton.__version__,
            launcher_abi=wrapper_hashes,
        )
        self.device_bytes = _asc_sort_build(identity, home, bisheng)
        binary_hash = hashlib.sha256(self.device_bytes).hexdigest()
        self.entries = {}
        for name, (src, metadata) in specifications.items():
            metadata.hash = binary_hash
            launcher = driver.NPULauncher(src, metadata)
            handles = self.loader.load_binary(name, self.device_bytes, 0, device, "aiv")
            _asc_sort_require(
                isinstance(handles, tuple) and len(handles) == 5,
                "unexpected loader return",
            )
            module, function = handles[:2]
            _asc_sort_require(
                all(
                    isinstance(handle, int) and handle > 0
                    for handle in (module, function)
                ),
                "AIV kernel registration failed: " + name,
            )
            # Retain binary bytes, both handles and original launcher for the
            # lifetime of this process/device cache; no tensor scratch is cached.
            self.entries[name] = (module, function, launcher)

    def launch(self, name, blocks, stream, args):
        _, function, launcher = self.entries[name]
        launcher(
            blocks,
            1,
            1,
            stream,
            function,
            {"kernel_name": name, "tensor_kinds": [0, 1, 1]},
            None,
            None,
            None,
            *args,
        )


def _asc_sort_get_runtime(device):
    key = (os.getpid(), device, tuple(os.getenv(name) for name in _ASC_SORT_ENV))
    with _ASC_SORT_LOCK:
        if key not in _ASC_SORT_RUNTIMES:
            try:
                runtime = _AscSortRuntime(device)
            except Exception as error:
                # This function cannot submit kernels. Never fall back after a
                # launch has been submitted: execution errors must propagate.
                logger.warning(
                    "Ascend argsort ASC route unavailable; using Triton: %s", error
                )
                runtime = None
            _ASC_SORT_RUNTIMES[key] = runtime
        return _ASC_SORT_RUNTIMES[key]


def _argsort_asc_runtime(inp, descending):
    n = inp.shape[-1]
    rows = inp.numel() // n
    initial_tasks = rows * triton.cdiv(n, 4096)
    if initial_tasks > 0x7FFFFFFF:
        return None
    with torch_device_fn.device(inp.device):
        runtime = _asc_sort_get_runtime(inp.device.index)
        if runtime is None:
            return None
        stages = [(4096, initial_tasks, False)]
        run = 4096
        while run < n:
            tasks = rows * triton.cdiv(n, 4 * run)
            stages.append((run, tasks, 4 * run >= n))
            run *= 4
        blocks = [min(runtime.aiv, tasks) for _, tasks, _ in stages]
        out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        current = torch.empty(
            (2 * inp.numel() + 8,), dtype=torch.float32, device=inp.device
        )
        scratch = torch.empty(
            (2 * inp.numel() + 8,), dtype=torch.float32, device=inp.device
        )
        status = [
            torch.empty((count * 64,), dtype=torch.int32, device=inp.device)
            for count in blocks
        ]
        stream_object = torch.npu.current_stream()
        stream = runtime.driver.get_current_stream(inp.device.index)
        _asc_sort_require(isinstance(stream, int), "unexpected NPU stream handle")
        for tensor in (inp, out, current, scratch, *status):
            tensor.record_stream(stream_object)
        name = (
            "v202_fp32_initial_a"
            if inp.dtype == torch.float32
            else "v202_raw16_initial_a"
        )
        args = (inp, current, status[0], rows, n, int(descending), blocks[0])
        if inp.dtype != torch.float32:
            args += (0 if inp.dtype == torch.float16 else 1,)
        runtime.launch(name, blocks[0], stream, args)
        for i, (run, _, final) in enumerate(stages[1:], 1):
            runtime.launch(
                "v202_merge4_a",
                blocks[i],
                stream,
                (
                    current,
                    out if final else scratch,
                    status[i],
                    rows,
                    n,
                    run,
                    blocks[i],
                    int(final),
                ),
            )
            if not final:
                current, scratch = scratch, current
        return out


def argsort(inp, dim=-1, descending=False):
    logger.debug("GEMS_ASCEND ARGSORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    n = inp.shape[dim] if rank else 1
    if inp.numel() == 0:
        return torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    if n == 1:
        out = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        with torch_device_fn.device(inp.device):
            _argsort_integer_zero[(triton.cdiv(inp.numel(), 256),)](
                out, inp.numel(), num_warps=4
            )
        return out
    if (
        rank > 0
        and inp.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and 4096 < n <= 65536
        and dim % rank == rank - 1
        and inp.is_contiguous()
    ):
        result = _argsort_asc_runtime(inp, descending)
        if result is not None:
            return result
    if (
        rank > 0
        and n <= 256
        and inp.dtype
        in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int16,
            torch.int8,
            torch.uint8,
        )
        and any(size > 1 for size in inp.shape[dim % rank + 1 :])
    ):
        return _argsort_short_output(inp, dim, descending)
    if inp.dtype in (torch.int32, torch.int64) and n <= 4096:
        return _argsort_integer_paired(inp, dim, descending)
    if (
        inp.dtype in (torch.int32, torch.float16, torch.bfloat16, torch.int16)
        and 4096 < n <= 262144
    ):
        return _argsort_ascend_fp32_merge(inp, dim, descending)
    if (
        inp.dtype in (torch.int8, torch.uint8)
        and 4096 < n <= 262144
        and inp.stride(dim) == 1
    ):
        return _argsort_ascend_fp32_merge(inp, dim, descending)
    if inp.dtype == torch.int64 and 4096 < n <= 262144:
        return _argsort_int64_merge(inp, dim, descending)
    if inp.dtype == torch.float32:
        if rank > 0 and 256 < n <= 4096:
            return _argsort_paired(inp, dim, descending)
        if 256 < n <= 1024:
            return _argsort_medium_lsd(inp, dim, descending)
        if 1024 < n <= 4096:
            return _argsort_row_lsd(inp, dim, descending)
        if 4096 < n <= 262144:
            return _argsort_ascend_fp32_merge(inp, dim, descending)
    if (
        rank > 0
        and n <= 4096
        and (inp.dtype in (torch.int8, torch.uint8))
        and (inp.stride(dim) != 1)
    ):
        return _argsort_normal_narrow(inp, dim, descending)
    if 256 < n <= 4096 and inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        return _argsort_normal_narrow(inp, dim, descending)
    if n <= 256 and inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        return _argsort_short_narrow(inp, dim, descending)
    if inp.dtype == torch.float32 and n <= 256:
        return _argsort_short_lsd(inp, dim, descending)
    return _argsort_merge_entry(inp, dim, descending)
