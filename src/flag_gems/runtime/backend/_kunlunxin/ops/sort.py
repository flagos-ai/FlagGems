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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.tensor_wrapper import StridedBuffer

from .cat import copy_func
from .topk import _get_finfo_val, _get_iinfo_val, argsort

logger = logging.getLogger(__name__)


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    if x.dtype.is_floating():
        if x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


@triton.jit
def count_kernel(
    x_ptr,
    counts_ptr,  # Output: [M, R_PAD] int32, bin-major: bin * GRID_N + block
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
    GRID_N: tl.constexpr,
    R_PAD: tl.constexpr,
):

    pid = tl.program_id(0)

    row_idx = pid // GRID_N
    block_idx = pid % GRID_N

    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N

    # Masked loads on XPU can still issue the tail address.  Clamp inactive
    # lanes to the last real element and use the mask only for the histogram
    # predicate.
    n_offset_safe = tl.minimum(n_offset, N - 1)
    val = tl.load(x_ptr + row_start + n_offset_safe)
    val_u = convert_to_uint_preverse_order(val, descending)

    bfe_mask = num_bins - 1
    key = (val_u >> bit_offset) & bfe_mask

    counts_row = counts_ptr + row_idx * R_PAD + block_idx
    for i in range(num_bins):
        bin_mask = (key == i) & mask
        count = tl.sum(bin_mask.to(tl.int32))
        tl.store(counts_row + i * GRID_N, count)


@libentry()
@triton.jit
def bin_prefix_kernel(
    counts_ptr,  # [M, R_PAD] int32 (bin-major histogram)
    offsets_ptr,  # [M, R_PAD] int32 (exclusive prefix sums)
    R: tl.constexpr,  # num_bins * GRID_N valid entries per row
    R_PAD: tl.constexpr,  # padded row pitch, multiple of TILE
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * R_PAD
    carry = tl.zeros([TILE], tl.int32)
    for start in range(0, R_PAD, TILE):
        offs = start + tl.arange(0, TILE)
        v = tl.load(counts_ptr + base + offs)
        v = tl.where(offs < R, v, 0)
        inclusive = tl.cumsum(v, axis=0)
        tl.store(offsets_ptr + base + offs, inclusive - v + carry)
        carry += tl.sum(v, axis=0)


@triton.jit
def scatter_kernel(
    x_ptr,
    x_out_ptr,
    idx_in_ptr,
    idx_out_ptr,
    global_offsets_ptr,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
    GRID_N: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // GRID_N
    block_idx = pid % GRID_N

    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N

    n_offset_safe = tl.minimum(n_offset, N - 1)
    val = tl.load(x_ptr + row_start + n_offset_safe)
    val_u = convert_to_uint_preverse_order(val, descending)

    idx = tl.load(idx_in_ptr + row_start + n_offset_safe)

    bfe_mask = num_bins - 1
    key = (val_u >> bit_offset) & bfe_mask

    lane = tl.arange(0, BLOCK_N)
    dest_idx = (lane - BLOCK_N).to(tl.int64)

    offsets_row = global_offsets_ptr + row_idx * R_PAD + block_idx
    for i in range(num_bins):
        bin_mask = (key == i) & mask
        local_rank = tl.cumsum(tl.where(bin_mask, 1, 0), axis=0) - 1

        global_start = tl.load(offsets_row + i * GRID_N)

        dest_idx = tl.where(
            bin_mask,
            (row_start + global_start + local_rank).to(tl.int64),
            dest_idx,
        )

    tl.store(x_out_ptr + dest_idx, val)
    tl.store(idx_out_ptr + dest_idx, idx)


@libentry()
@triton.jit
def init_indices_kernel(indices, total, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(indices + offsets, offsets % N, mask=offsets < total)


@libentry()
@triton.jit
def init_sort_buffers_kernel(
    source, values, indices, total, N, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    tl.store(values + offsets, tl.load(source + offsets, mask=mask), mask=mask)
    tl.store(indices + offsets, offsets % N, mask=mask)


def radix_sort_low_mem(arr, k_bits=4, descending=False):
    original_shape = arr.shape
    N = arr.shape[-1]
    arr = arr.reshape(-1, N)
    M = arr.shape[0]

    _env_block_n = os.environ.get("GEMS_XPU_RADIX_BLOCK_N")
    if _env_block_n:
        BLOCK_N = int(_env_block_n)
    else:
        BLOCK_N = min(4096, max(64, triton.next_power_of_2(N)))
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (M * grid_n,)

    _HEAD_PAD = BLOCK_N
    _TAIL_PAD = 256
    _keepalive = []

    def _padded(dtype):
        buf = torch.empty(_HEAD_PAD + M * N + _TAIL_PAD, device=arr.device, dtype=dtype)
        _keepalive.append(buf)
        return buf[_HEAD_PAD : _HEAD_PAD + M * N].view(M, N)

    arr_in = _padded(arr.dtype)
    arr_out = _padded(arr.dtype)
    idx_in = _padded(torch.int64)
    idx_out = _padded(torch.int64)

    index_block = 256
    with torch_device_fn.device(arr.device):
        init_sort_buffers_kernel[(triton.cdiv(M * N, index_block),)](
            arr, arr_in, idx_in, M * N, N, BLOCK_SIZE=index_block
        )

    dtype = arr.dtype
    num_bits = 1
    if dtype == torch.bool:
        pass
    elif dtype == torch.bfloat16:
        num_bits = 4 * 8
    else:
        num_bits = arr.element_size() * 8
    num_passes = (num_bits + k_bits - 1) // k_bits
    num_bins = 2**k_bits

    r = num_bins * grid_n
    tile_r = max(64, min(4096, triton.next_power_of_2(r)))
    r_pad = triton.cdiv(r, tile_r) * tile_r

    with torch_device_fn.device(arr.device):
        counts = torch.empty(M * r_pad, device=arr.device, dtype=torch.int32)
        global_offsets = torch.empty(M * r_pad, device=arr.device, dtype=torch.int32)

        for p in range(num_passes):
            bit_offset = p * k_bits
            count_kernel[grid](
                arr_in,
                counts,
                M,
                N,
                bit_offset,
                num_bins,
                BLOCK_N,
                descending,
                GRID_N=grid_n,
                R_PAD=r_pad,
            )

            bin_prefix_kernel[(M,)](
                counts,
                global_offsets,
                R=r,
                R_PAD=r_pad,
                TILE=tile_r,
            )

            scatter_kernel[grid](
                arr_in,
                arr_out,
                idx_in,
                idx_out,
                global_offsets,
                M,
                N,
                bit_offset,
                num_bins,
                BLOCK_N,
                descending,
                GRID_N=grid_n,
                R_PAD=r_pad,
            )

            arr_in, arr_out = arr_out, arr_in
            idx_in, idx_out = idx_out, idx_in

    return arr_in.reshape(original_shape), idx_in.reshape(original_shape)


def _copy_bitview(t):
    # NOTE(kunlunxin): the tuned copy codegen hits an illegal memory access for
    # int32 on the strided-read path at some shapes (deterministic repro:
    # permute-read of a (65536, 4) / (32768, 8) int32 tensor, i.e.
    # M * N = 256 Ki; smaller shapes are clean).  int32 and float32 share
    # itemsize, so bit-cast both sides and reuse the proven fp32 copy path;
    # the copy is bit-exact.  Same workaround as the cat family.
    if t.dtype == torch.int32:
        return t.view(torch.float32)
    return t


def _permute_copy_to_last(inp, dim):
    """Materialize a dim-last view without routing a strided copy through copy_."""
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    shape = tuple(inp.shape[i] for i in order)
    strides = tuple(inp.stride()[i] for i in order)
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    src = _copy_bitview(inp)
    dst = _copy_bitview(out)
    in_view = StridedBuffer(src, shape, strides)
    out_view = StridedBuffer(dst, shape, dst.stride())
    copy_func.instantiate(inp.ndim)(in_view, out0=out_view)
    return out, order


def _permute_copy_from_last(inp, out_shape, order):
    """Copy a dim-last result into the original contiguous dimension order."""
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    shape = tuple(out_shape[i] for i in order)
    strides = tuple(out.stride()[i] for i in order)
    src = _copy_bitview(inp)
    dst = _copy_bitview(out)
    in_view = StridedBuffer(src, shape, src.stride())
    out_view = StridedBuffer(dst, shape, strides)
    copy_func.instantiate(len(order))(in_view, out0=out_view)
    return out


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
        in_val = tl.where(in_val.dtype.is_fp64(), in_val, in_val.to(tl.float32))
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val).to(tl.int32)
    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN SORT")
    if inp.ndim == 0:
        return inp.clone(), torch.zeros_like(inp, dtype=torch.int64)
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0 or inp.numel() == 0:
        return inp, torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        indices = torch.empty_like(inp, dtype=torch.int64)
        with torch_device_fn.device(inp.device):
            init_indices_kernel[(triton.cdiv(inp.numel(), 256),)](
                indices, inp.numel(), 1, BLOCK_SIZE=256
            )
        return inp, indices

    return sort_stable(inp, stable=True, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN SORT_STABLE")
    # We only implement stable radix sort here
    _ = stable
    if inp.ndim == 0:
        return inp.clone(), torch.zeros_like(inp, dtype=torch.int64)
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0 or inp.numel() == 0:
        return inp, torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        indices = torch.empty_like(inp, dtype=torch.int64)
        with torch_device_fn.device(inp.device):
            init_indices_kernel[(triton.cdiv(inp.numel(), 256),)](
                indices, inp.numel(), 1, BLOCK_SIZE=256
            )
        return inp, indices

    if dim < 0:
        dim = dim + inp.ndim
    original_shape = inp.shape
    if dim != inp.ndim - 1:
        inp, order = _permute_copy_to_last(inp, dim)
    else:
        order = list(range(inp.ndim))
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort_low_mem(inp, num_bits_per_pass, descending)

    if dim != len(order) - 1:
        out = _permute_copy_from_last(out, original_shape, order)
        out_index = _permute_copy_from_last(out_index, original_shape, order)
    return out, out_index
