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

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

_MAX_SINGLE_STAGE_WINDOW = 8192
_MAX_SMALL_WINDOW = 64
_SMALL_WINDOW_CHUNK_SIZE = 8
_SMALL_WINDOW_BLOCK_OUT = 256
_WINDOW_CHUNK_SIZE = 2048
_MAX_STAGE2_CHUNKS = 8192
_DEVICE_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit
def adaptive_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    d_bounds_ptr,
    h_bounds_ptr,
    w_bounds_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    BLOCK_WINDOW: tl.constexpr,
):
    nd_idx = tl.program_id(0)
    h_out = tl.program_id(1)
    w_out = tl.program_id(2)
    nc_idx = nd_idx // out_d
    d_out = nd_idx % out_d
    out_offset = (
        nc_idx * out_d * out_h * out_w + d_out * out_h * out_w + h_out * out_w + w_out
    )

    d_start = tl.load(d_bounds_ptr + d_out)
    h_start = tl.load(h_bounds_ptr + h_out)
    w_start = tl.load(w_bounds_ptr + w_out)
    d_end = tl.load(d_bounds_ptr + out_d + d_out)
    h_end = tl.load(h_bounds_ptr + out_h + h_out)
    w_end = tl.load(w_bounds_ptr + out_w + w_out)
    window_d = d_end - d_start
    window_h = h_end - h_start
    window_w = w_end - w_start
    window_offset = tl.arange(0, BLOCK_WINDOW)
    max_window_hw = MAX_WINDOW_H * MAX_WINDOW_W
    d_delta = window_offset // max_window_hw
    window_remainder = window_offset % max_window_hw
    h_delta = window_remainder // MAX_WINDOW_W
    w_delta = window_remainder % MAX_WINDOW_W
    d_in = d_start + d_delta
    h_in = h_start + h_delta
    w_in = w_start + w_delta

    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    input_offset = (
        n_idx * in_stride_n
        + c_idx * in_stride_c
        + d_in * in_stride_d
        + h_in * in_stride_h
        + w_in * in_stride_w
    )
    valid = (d_delta < window_d) & (h_delta < window_h) & (w_delta < window_w)
    min_value = get_dtype_min(input_ptr.type.element_ty)
    values = tl.load(input_ptr + input_offset, mask=valid, other=min_value)

    values_fp32 = values.to(tl.float32)
    value_bits = values_fp32.to(tl.uint32, bitcast=True)
    sign_bit = value_bits & 0x80000000
    ordered_bits = tl.where(sign_bit != 0, ~value_bits, value_bits ^ 0x80000000)
    ordered_bits = tl.where(values_fp32 == 0.0, 0x80000000, ordered_bits)
    is_nan = values_fp32 != values_fp32
    nan_rank = 0x100000000
    value_rank = tl.where(is_nan, nan_rank, ordered_bits.to(tl.int64))
    tie_rank = tl.where(is_nan, window_offset, KEY_STRIDE - 1 - window_offset).to(
        tl.int64
    )
    packed_key = value_rank * KEY_STRIDE + tie_rank
    packed_key = tl.where(valid, packed_key, 0)
    best_key = tl.max(packed_key, axis=0)
    best_value_rank = best_key // KEY_STRIDE
    best_tie_rank = best_key % KEY_STRIDE
    selected_offset = tl.where(
        best_value_rank == nan_rank,
        best_tie_rank,
        KEY_STRIDE - 1 - best_tie_rank,
    )

    selected_d = d_start + selected_offset // max_window_hw
    selected_remainder = selected_offset % max_window_hw
    selected_h = h_start + selected_remainder // MAX_WINDOW_W
    selected_w = w_start + selected_remainder % MAX_WINDOW_W
    selected_input_offset = (
        n_idx * in_stride_n
        + c_idx * in_stride_c
        + selected_d * in_stride_d
        + selected_h * in_stride_h
        + selected_w * in_stride_w
    )
    selected_value = tl.load(input_ptr + selected_input_offset)
    selected_index = selected_d * in_h * in_w + selected_h * in_w + selected_w
    tl.store(output_ptr + out_offset, selected_value)
    tl.store(indices_ptr + out_offset, selected_index)


@libentry()
@triton.jit
def adaptive_max_pool3d_forward_small_window_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    d_bounds_ptr,
    h_bounds_ptr,
    w_bounds_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    num_outputs,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_WINDOW: tl.constexpr,
):
    output_offset = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    output_valid = output_offset < num_outputs
    out_spatial = out_d * out_h * out_w
    nc_idx = output_offset // out_spatial
    output_remainder = output_offset % out_spatial
    d_out = output_remainder // (out_h * out_w)
    output_remainder = output_remainder % (out_h * out_w)
    h_out = output_remainder // out_w
    w_out = output_remainder % out_w

    d_start = tl.load(d_bounds_ptr + d_out, mask=output_valid, other=0)
    h_start = tl.load(h_bounds_ptr + h_out, mask=output_valid, other=0)
    w_start = tl.load(w_bounds_ptr + w_out, mask=output_valid, other=0)
    d_end = tl.load(d_bounds_ptr + out_d + d_out, mask=output_valid, other=0)
    h_end = tl.load(h_bounds_ptr + out_h + h_out, mask=output_valid, other=0)
    w_end = tl.load(w_bounds_ptr + out_w + w_out, mask=output_valid, other=0)
    window_d = d_end - d_start
    window_h = h_end - h_start
    window_w = w_end - w_start
    max_window_hw = MAX_WINDOW_H * MAX_WINDOW_W
    best_key = tl.zeros((BLOCK_OUT,), tl.int64)

    for window_offset in tl.static_range(0, BLOCK_WINDOW):
        d_delta = window_offset // max_window_hw
        window_remainder = window_offset % max_window_hw
        h_delta = window_remainder // MAX_WINDOW_W
        w_delta = window_remainder % MAX_WINDOW_W
        d_in = d_start + d_delta
        h_in = h_start + h_delta
        w_in = w_start + w_delta
        n_idx = nc_idx // in_c
        c_idx = nc_idx % in_c
        input_offset = (
            n_idx * in_stride_n
            + c_idx * in_stride_c
            + d_in * in_stride_d
            + h_in * in_stride_h
            + w_in * in_stride_w
        )
        valid = (
            output_valid
            & (d_delta < window_d)
            & (h_delta < window_h)
            & (w_delta < window_w)
        )
        min_value = get_dtype_min(input_ptr.type.element_ty)
        values = tl.load(input_ptr + input_offset, mask=valid, other=min_value)
        values_fp32 = values.to(tl.float32)
        value_bits = values_fp32.to(tl.uint32, bitcast=True)
        sign_bit = value_bits & 0x80000000
        ordered_bits = tl.where(sign_bit != 0, ~value_bits, value_bits ^ 0x80000000)
        ordered_bits = tl.where(values_fp32 == 0.0, 0x80000000, ordered_bits)
        is_nan = values_fp32 != values_fp32
        value_rank = tl.where(is_nan, 0x100000000, ordered_bits.to(tl.int64))
        tie_rank = tl.where(is_nan, window_offset, KEY_STRIDE - 1 - window_offset).to(
            tl.int64
        )
        packed_key = value_rank * KEY_STRIDE + tie_rank
        packed_key = tl.where(valid, packed_key, 0)
        best_key = tl.where(packed_key > best_key, packed_key, best_key)

    best_value_rank = best_key // KEY_STRIDE
    best_tie_rank = best_key % KEY_STRIDE
    selected_offset = tl.where(
        best_value_rank == 0x100000000,
        best_tie_rank,
        KEY_STRIDE - 1 - best_tie_rank,
    )
    selected_d = d_start + selected_offset // max_window_hw
    selected_remainder = selected_offset % max_window_hw
    selected_h = h_start + selected_remainder // MAX_WINDOW_W
    selected_w = w_start + selected_remainder % MAX_WINDOW_W
    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    selected_input_offset = (
        n_idx * in_stride_n
        + c_idx * in_stride_c
        + selected_d * in_stride_d
        + selected_h * in_stride_h
        + selected_w * in_stride_w
    )
    selected_value = tl.load(
        input_ptr + selected_input_offset, mask=output_valid, other=0.0
    )
    selected_index = selected_d * in_h * in_w + selected_h * in_w + selected_w
    tl.store(output_ptr + output_offset, selected_value, mask=output_valid)
    tl.store(indices_ptr + output_offset, selected_index, mask=output_valid)


@libentry()
@triton.jit
def adaptive_max_pool3d_forward_small_stage1_kernel(
    input_ptr,
    partial_key_ptr,
    d_bounds_ptr,
    h_bounds_ptr,
    w_bounds_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    in_c,
    out_d,
    out_h,
    out_w,
    num_outputs,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_WINDOW: tl.constexpr,
):
    output_offset = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    chunk_idx = tl.program_id(1)
    output_valid = output_offset < num_outputs
    out_spatial = out_d * out_h * out_w
    nc_idx = output_offset // out_spatial
    output_remainder = output_offset % out_spatial
    d_out = output_remainder // (out_h * out_w)
    output_remainder = output_remainder % (out_h * out_w)
    h_out = output_remainder // out_w
    w_out = output_remainder % out_w

    d_start = tl.load(d_bounds_ptr + d_out, mask=output_valid, other=0)
    h_start = tl.load(h_bounds_ptr + h_out, mask=output_valid, other=0)
    w_start = tl.load(w_bounds_ptr + w_out, mask=output_valid, other=0)
    d_end = tl.load(d_bounds_ptr + out_d + d_out, mask=output_valid, other=0)
    h_end = tl.load(h_bounds_ptr + out_h + h_out, mask=output_valid, other=0)
    w_end = tl.load(w_bounds_ptr + out_w + w_out, mask=output_valid, other=0)
    window_d = d_end - d_start
    window_h = h_end - h_start
    window_w = w_end - w_start
    max_window_hw = MAX_WINDOW_H * MAX_WINDOW_W
    best_key = tl.zeros((BLOCK_OUT,), tl.int64)

    for window_delta in tl.static_range(0, BLOCK_WINDOW):
        window_offset = chunk_idx * BLOCK_WINDOW + window_delta
        d_delta = window_offset // max_window_hw
        window_remainder = window_offset % max_window_hw
        h_delta = window_remainder // MAX_WINDOW_W
        w_delta = window_remainder % MAX_WINDOW_W
        d_in = d_start + d_delta
        h_in = h_start + h_delta
        w_in = w_start + w_delta
        n_idx = nc_idx // in_c
        c_idx = nc_idx % in_c
        input_offset = (
            n_idx * in_stride_n
            + c_idx * in_stride_c
            + d_in * in_stride_d
            + h_in * in_stride_h
            + w_in * in_stride_w
        )
        valid = (
            output_valid
            & (d_delta < window_d)
            & (h_delta < window_h)
            & (w_delta < window_w)
        )
        min_value = get_dtype_min(input_ptr.type.element_ty)
        values = tl.load(input_ptr + input_offset, mask=valid, other=min_value)
        values_fp32 = values.to(tl.float32)
        value_bits = values_fp32.to(tl.uint32, bitcast=True)
        sign_bit = value_bits & 0x80000000
        ordered_bits = tl.where(sign_bit != 0, ~value_bits, value_bits ^ 0x80000000)
        ordered_bits = tl.where(values_fp32 == 0.0, 0x80000000, ordered_bits)
        is_nan = values_fp32 != values_fp32
        value_rank = tl.where(is_nan, 0x100000000, ordered_bits.to(tl.int64))
        tie_rank = tl.where(is_nan, window_offset, KEY_STRIDE - 1 - window_offset).to(
            tl.int64
        )
        packed_key = value_rank * KEY_STRIDE + tie_rank
        packed_key = tl.where(valid, packed_key, 0)
        best_key = tl.where(packed_key > best_key, packed_key, best_key)

    partial_offset = output_offset * NUM_CHUNKS + chunk_idx
    tl.store(partial_key_ptr + partial_offset, best_key, mask=output_valid)


@libentry()
@triton.jit
def adaptive_max_pool3d_forward_small_stage2_kernel(
    input_ptr,
    partial_key_ptr,
    output_ptr,
    indices_ptr,
    d_bounds_ptr,
    h_bounds_ptr,
    w_bounds_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    num_outputs,
    KEY_STRIDE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    output_offset = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    output_valid = output_offset < num_outputs
    best_key = tl.zeros((BLOCK_OUT,), tl.int64)
    for chunk_idx in tl.static_range(0, NUM_CHUNKS):
        partial_key = tl.load(
            partial_key_ptr + output_offset * NUM_CHUNKS + chunk_idx,
            mask=output_valid,
            other=0,
        )
        best_key = tl.where(partial_key > best_key, partial_key, best_key)

    best_value_rank = best_key // KEY_STRIDE
    best_tie_rank = best_key % KEY_STRIDE
    selected_offset = tl.where(
        best_value_rank == 0x100000000,
        best_tie_rank,
        KEY_STRIDE - 1 - best_tie_rank,
    )
    out_spatial = out_d * out_h * out_w
    nc_idx = output_offset // out_spatial
    output_remainder = output_offset % out_spatial
    d_out = output_remainder // (out_h * out_w)
    output_remainder = output_remainder % (out_h * out_w)
    h_out = output_remainder // out_w
    w_out = output_remainder % out_w
    d_start = tl.load(d_bounds_ptr + d_out, mask=output_valid, other=0)
    h_start = tl.load(h_bounds_ptr + h_out, mask=output_valid, other=0)
    w_start = tl.load(w_bounds_ptr + w_out, mask=output_valid, other=0)
    max_window_hw = MAX_WINDOW_H * MAX_WINDOW_W
    selected_d = d_start + selected_offset // max_window_hw
    selected_remainder = selected_offset % max_window_hw
    selected_h = h_start + selected_remainder // MAX_WINDOW_W
    selected_w = w_start + selected_remainder % MAX_WINDOW_W
    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    selected_input_offset = (
        n_idx * in_stride_n
        + c_idx * in_stride_c
        + selected_d * in_stride_d
        + selected_h * in_stride_h
        + selected_w * in_stride_w
    )
    selected_value = tl.load(
        input_ptr + selected_input_offset, mask=output_valid, other=0.0
    )
    selected_index = selected_d * in_h * in_w + selected_h * in_w + selected_w
    tl.store(output_ptr + output_offset, selected_value, mask=output_valid)
    tl.store(indices_ptr + output_offset, selected_index, mask=output_valid)


@libentry()
@triton.jit
def adaptive_max_pool3d_forward_stage1_kernel(
    input_ptr,
    partial_key_ptr,
    d_bounds_ptr,
    h_bounds_ptr,
    w_bounds_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    in_c,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    BLOCK_WINDOW: tl.constexpr,
):
    ndh_idx = tl.program_id(0)
    w_out = tl.program_id(1)
    chunk_idx = tl.program_id(2)
    h_out = ndh_idx % out_h
    nd_idx = ndh_idx // out_h
    d_out = nd_idx % out_d
    nc_idx = nd_idx // out_d
    out_offset = (
        nc_idx * out_d * out_h * out_w + d_out * out_h * out_w + h_out * out_w + w_out
    )

    d_start = tl.load(d_bounds_ptr + d_out)
    h_start = tl.load(h_bounds_ptr + h_out)
    w_start = tl.load(w_bounds_ptr + w_out)
    d_end = tl.load(d_bounds_ptr + out_d + d_out)
    h_end = tl.load(h_bounds_ptr + out_h + h_out)
    w_end = tl.load(w_bounds_ptr + out_w + w_out)
    window_d = d_end - d_start
    window_h = h_end - h_start
    window_w = w_end - w_start

    window_offset = chunk_idx * BLOCK_WINDOW + tl.arange(0, BLOCK_WINDOW)
    max_window_hw = MAX_WINDOW_H * MAX_WINDOW_W
    d_delta = window_offset // max_window_hw
    window_remainder = window_offset % max_window_hw
    h_delta = window_remainder // MAX_WINDOW_W
    w_delta = window_remainder % MAX_WINDOW_W
    d_in = d_start + d_delta
    h_in = h_start + h_delta
    w_in = w_start + w_delta

    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    input_offset = (
        n_idx * in_stride_n
        + c_idx * in_stride_c
        + d_in * in_stride_d
        + h_in * in_stride_h
        + w_in * in_stride_w
    )
    valid = (d_delta < window_d) & (h_delta < window_h) & (w_delta < window_w)
    min_value = get_dtype_min(input_ptr.type.element_ty)
    values = tl.load(input_ptr + input_offset, mask=valid, other=min_value)
    values_fp32 = values.to(tl.float32)
    value_bits = values_fp32.to(tl.uint32, bitcast=True)
    sign_bit = value_bits & 0x80000000
    ordered_bits = tl.where(sign_bit != 0, ~value_bits, value_bits ^ 0x80000000)
    ordered_bits = tl.where(values_fp32 == 0.0, 0x80000000, ordered_bits)
    is_nan = values_fp32 != values_fp32
    value_rank = tl.where(is_nan, 0x100000000, ordered_bits.to(tl.int64))
    tie_rank = tl.where(is_nan, window_offset, KEY_STRIDE - 1 - window_offset).to(
        tl.int64
    )
    packed_key = value_rank * KEY_STRIDE + tie_rank
    packed_key = tl.where(valid, packed_key, 0)
    partial_key = tl.max(packed_key, axis=0)
    tl.store(partial_key_ptr + out_offset * NUM_CHUNKS + chunk_idx, partial_key)


@libentry()
@triton.jit
def adaptive_max_pool3d_forward_stage2_kernel(
    input_ptr,
    partial_key_ptr,
    output_ptr,
    indices_ptr,
    d_bounds_ptr,
    h_bounds_ptr,
    w_bounds_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    nd_idx = tl.program_id(0)
    h_out = tl.program_id(1)
    w_out = tl.program_id(2)
    nc_idx = nd_idx // out_d
    d_out = nd_idx % out_d
    out_offset = (
        nc_idx * out_d * out_h * out_w + d_out * out_h * out_w + h_out * out_w + w_out
    )

    chunk_offset = tl.arange(0, BLOCK_CHUNKS)
    partial_keys = tl.load(
        partial_key_ptr + out_offset * NUM_CHUNKS + chunk_offset,
        mask=chunk_offset < NUM_CHUNKS,
        other=0,
    )
    best_key = tl.max(partial_keys, axis=0)
    best_value_rank = best_key // KEY_STRIDE
    best_tie_rank = best_key % KEY_STRIDE
    selected_offset = tl.where(
        best_value_rank == 0x100000000,
        best_tie_rank,
        KEY_STRIDE - 1 - best_tie_rank,
    )

    d_start = tl.load(d_bounds_ptr + d_out)
    h_start = tl.load(h_bounds_ptr + h_out)
    w_start = tl.load(w_bounds_ptr + w_out)
    max_window_hw = MAX_WINDOW_H * MAX_WINDOW_W
    selected_d = d_start + selected_offset // max_window_hw
    selected_remainder = selected_offset % max_window_hw
    selected_h = h_start + selected_remainder // MAX_WINDOW_W
    selected_w = w_start + selected_remainder % MAX_WINDOW_W
    n_idx = nc_idx // in_c
    c_idx = nc_idx % in_c
    selected_input_offset = (
        n_idx * in_stride_n
        + c_idx * in_stride_c
        + selected_d * in_stride_d
        + selected_h * in_stride_h
        + selected_w * in_stride_w
    )
    selected_value = tl.load(input_ptr + selected_input_offset)
    selected_index = selected_d * in_h * in_w + selected_h * in_w + selected_w
    tl.store(output_ptr + out_offset, selected_value)
    tl.store(indices_ptr + out_offset, selected_index)


def _normalized_shape_and_strides(input):
    if input.ndim == 5:
        return input.shape, input.stride()
    if input.ndim == 4:
        c, d, h, w = input.shape
        stride_c, stride_d, stride_h, stride_w = input.stride()
        return (1, c, d, h, w), (0, stride_c, stride_d, stride_h, stride_w)
    raise RuntimeError(f"expected 4D or 5D input, got {input.ndim}D")


def _max_window_size(input_size, output_size):
    return max(
        ((out_idx + 1) * input_size + output_size - 1) // output_size
        - out_idx * input_size // output_size
        for out_idx in range(output_size)
    )


def _pool_bounds(input_size, output_size, device):
    starts = [out_idx * input_size // output_size for out_idx in range(output_size)]
    ends = [
        ((out_idx + 1) * input_size + output_size - 1) // output_size
        for out_idx in range(output_size)
    ]
    return torch.tensor(starts + ends, device=device, dtype=torch.int32)


def _cpu_fallback(input, output_size):
    cpu_output, cpu_indices = torch.ops.aten.adaptive_max_pool3d.default(
        input.cpu(), output_size
    )
    return cpu_output.to(input.device), cpu_indices.to(input.device)


def adaptive_max_pool3d(input, output_size):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D")

    output_size = tuple(output_size)
    if len(output_size) != 3:
        raise RuntimeError("adaptive_max_pool3d output_size must contain three values")
    if input.dtype not in _DEVICE_DTYPES:
        return _cpu_fallback(input, output_size)
    normalized_shape, strides = _normalized_shape_and_strides(input)
    n, c, in_d, in_h, in_w = normalized_shape
    out_d, out_h, out_w = output_size
    if any(size == 0 for size in (in_d, in_h, in_w)) or (input.ndim == 5 and c == 0):
        raise RuntimeError("adaptive_max_pool3d input has an empty non-batch dimension")

    output_shape = (*input.shape[:-3], out_d, out_h, out_w)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    indices = torch.empty(output_shape, device=input.device, dtype=torch.int64)
    if output.numel() == 0:
        return output, indices

    max_window_d = _max_window_size(in_d, out_d)
    max_window_h = _max_window_size(in_h, out_h)
    max_window_w = _max_window_size(in_w, out_w)
    max_window_volume = max_window_d * max_window_h * max_window_w
    key_stride = triton.next_power_of_2(max_window_volume)
    block_window = key_stride
    d_bounds = _pool_bounds(in_d, out_d, input.device)
    h_bounds = _pool_bounds(in_h, out_h, input.device)
    w_bounds = _pool_bounds(in_w, out_w, input.device)

    if block_window <= _SMALL_WINDOW_CHUNK_SIZE:
        adaptive_max_pool3d_forward_small_window_kernel[
            (triton.cdiv(output.numel(), _SMALL_WINDOW_BLOCK_OUT),)
        ](
            input,
            output,
            indices,
            d_bounds,
            h_bounds,
            w_bounds,
            *strides,
            c,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            output.numel(),
            MAX_WINDOW_H=max_window_h,
            MAX_WINDOW_W=max_window_w,
            KEY_STRIDE=key_stride,
            BLOCK_OUT=_SMALL_WINDOW_BLOCK_OUT,
            BLOCK_WINDOW=block_window,
            num_warps=4,
            num_stages=3,
        )
        return output, indices

    if block_window <= _MAX_SMALL_WINDOW:
        num_chunks = block_window // _SMALL_WINDOW_CHUNK_SIZE
        partial_keys = torch.empty(
            (output.numel(), num_chunks), device=input.device, dtype=torch.int64
        )
        adaptive_max_pool3d_forward_small_stage1_kernel[
            (
                triton.cdiv(output.numel(), _SMALL_WINDOW_BLOCK_OUT),
                num_chunks,
            )
        ](
            input,
            partial_keys,
            d_bounds,
            h_bounds,
            w_bounds,
            *strides,
            c,
            out_d,
            out_h,
            out_w,
            output.numel(),
            MAX_WINDOW_H=max_window_h,
            MAX_WINDOW_W=max_window_w,
            KEY_STRIDE=key_stride,
            NUM_CHUNKS=num_chunks,
            BLOCK_OUT=_SMALL_WINDOW_BLOCK_OUT,
            BLOCK_WINDOW=_SMALL_WINDOW_CHUNK_SIZE,
            num_warps=4,
            num_stages=3,
        )
        adaptive_max_pool3d_forward_small_stage2_kernel[
            (triton.cdiv(output.numel(), _SMALL_WINDOW_BLOCK_OUT),)
        ](
            input,
            partial_keys,
            output,
            indices,
            d_bounds,
            h_bounds,
            w_bounds,
            *strides,
            c,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            output.numel(),
            KEY_STRIDE=key_stride,
            NUM_CHUNKS=num_chunks,
            MAX_WINDOW_H=max_window_h,
            MAX_WINDOW_W=max_window_w,
            BLOCK_OUT=_SMALL_WINDOW_BLOCK_OUT,
            num_warps=4,
            num_stages=3,
        )
        return output, indices

    block_window = max(block_window, 256)
    if block_window <= _MAX_SINGLE_STAGE_WINDOW:
        adaptive_max_pool3d_forward_kernel[(n * c * out_d, out_h, out_w)](
            input,
            output,
            indices,
            d_bounds,
            h_bounds,
            w_bounds,
            *strides,
            c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            MAX_WINDOW_H=max_window_h,
            MAX_WINDOW_W=max_window_w,
            KEY_STRIDE=key_stride,
            BLOCK_WINDOW=block_window,
            num_warps=4,
            num_stages=3,
        )
        return output, indices

    num_chunks = triton.cdiv(max_window_volume, _WINDOW_CHUNK_SIZE)
    if num_chunks > _MAX_STAGE2_CHUNKS:
        logger.debug(
            "GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D_CPU_FALLBACK window=%d",
            max_window_volume,
        )
        return _cpu_fallback(input, output_size)

    partial_keys = torch.empty(
        (output.numel(), num_chunks), device=input.device, dtype=torch.int64
    )
    adaptive_max_pool3d_forward_stage1_kernel[
        (n * c * out_d * out_h, out_w, num_chunks)
    ](
        input,
        partial_keys,
        d_bounds,
        h_bounds,
        w_bounds,
        *strides,
        c,
        out_d,
        out_h,
        out_w,
        MAX_WINDOW_H=max_window_h,
        MAX_WINDOW_W=max_window_w,
        NUM_CHUNKS=num_chunks,
        KEY_STRIDE=key_stride,
        BLOCK_WINDOW=_WINDOW_CHUNK_SIZE,
        num_warps=4,
        num_stages=3,
    )
    adaptive_max_pool3d_forward_stage2_kernel[(n * c * out_d, out_h, out_w)](
        input,
        partial_keys,
        output,
        indices,
        d_bounds,
        h_bounds,
        w_bounds,
        *strides,
        c,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        MAX_WINDOW_H=max_window_h,
        MAX_WINDOW_W=max_window_w,
        NUM_CHUNKS=num_chunks,
        KEY_STRIDE=key_stride,
        BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
        num_warps=4,
        num_stages=3,
    )
    return output, indices


# Indices can be produced before entering use_gems(). Install this vendor-scoped
# CUDA override at import time and keep the Library alive.
_ATEN_LIB = torch.library.Library("aten", "IMPL")
try:
    _ATEN_LIB.impl(
        "adaptive_max_pool3d",
        adaptive_max_pool3d,
        "CUDA",
        allow_override=True,
    )
except TypeError:
    _ATEN_LIB.impl("adaptive_max_pool3d", adaptive_max_pool3d, "CUDA")
