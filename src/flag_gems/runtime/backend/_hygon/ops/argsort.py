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

from flag_gems.ops.argsort import (
    _argsort_merge,
    _argsort_tiles,
    _packed_merge,
    _radix_bucket_prefix,
    _radix_counts_only,
    _radix_fused_local_scatter,
    _radix_local_sort,
    _radix_prefix,
    _radix_scatter,
    _radix_tile_prefix,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _argsort_merge_entry(inp, dim=-1, descending=False):
    """Stable indices using bounded tiles and launch-separated merge passes."""
    logger.debug("GEMS_HYGON ARGSORT")
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
    block = min(triton.next_power_of_2(n), 256)
    row_block = max(1, 128 // block)
    index_bits = (n - 1).bit_length()
    use_packed = inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
    )
    if use_packed:
        block = min(triton.next_power_of_2(n), 1024)
    else:
        block = min(triton.next_power_of_2(n), 1024)
    row_block = max(1, 128 // block)
    merge_block = min(block, 1024)
    tile_warps = 4
    merge_warps = 4
    packed_warps = 1 if n >= 131072 and inp.dtype in (torch.float32, torch.int32) else 4
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
            if n >= 131072 and inp.dtype in (torch.float32, torch.int32):
                packed_merge_kernel = _packed_merge.fn
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
            values = torch.empty((rows, n), dtype=inp.dtype, device=inp.device)
            indices = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
            values_tmp = torch.empty_like(values)
            indices_tmp = torch.empty_like(indices)
            _argsort_tiles[rows * triton.cdiv(n, block),](
                inp,
                values,
                indices,
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


def _argsort_radix_fused_narrow(inp, dim, descending):
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
    block = 1024
    log_block = block.bit_length() - 1
    tiles = triton.cdiv(n, block)
    tile_block = triton.next_power_of_2(tiles)
    bucket_block = min(256, 16384 // tile_block)
    passes = inp.element_size()
    assert passes in (1, 2)
    counts = torch.empty((rows, 256, tiles), dtype=torch.int32, device=inp.device)
    offsets = torch.empty((rows, 256, tiles), dtype=torch.int32, device=inp.device)
    split_prefix = False
    bucket_totals = (
        torch.empty((rows, 256), dtype=torch.int32, device=inp.device)
        if split_prefix
        else offsets
    )
    if passes == 2:
        keys_current = torch.empty(
            (rows, n),
            dtype=torch.int16 if inp.dtype == torch.bfloat16 else torch.int32,
            device=inp.device,
        )
        indices_current = torch.empty((rows, n), dtype=torch.int32, device=inp.device)
    else:
        keys_current = out
        indices_current = out
    counts_warps = 4
    prefix_warps = 4
    fused_warps = 8 if split_prefix and passes == 2 else 4
    with torch_device_fn.device(inp.device):
        for pass_index in range(passes):
            first = pass_index == 0
            last = pass_index + 1 == passes
            _radix_counts_only[rows * tiles,](
                inp,
                keys_current,
                counts,
                n,
                tiles,
                shape,
                strides,
                axis_stride,
                block,
                pass_index * 8,
                descending,
                first,
                counts_warps,
                num_warps=counts_warps,
            )
            if split_prefix:
                _radix_tile_prefix[rows * 256,](
                    counts, offsets, bucket_totals, tiles, tile_block, num_warps=4
                )
                _radix_bucket_prefix[rows, triton.cdiv(tiles, 16)](
                    counts, offsets, bucket_totals, tiles, 16, num_warps=4
                )
            else:
                _radix_prefix[rows,](
                    counts,
                    offsets,
                    tiles,
                    tile_block,
                    bucket_block,
                    prefix_warps,
                    num_warps=prefix_warps,
                )
            _radix_fused_local_scatter[rows * tiles,](
                inp,
                keys_current,
                indices_current,
                offsets,
                out,
                n,
                tiles,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                log_block,
                pass_index * 8,
                descending,
                first,
                last,
                fused_warps,
                num_warps=fused_warps,
            )
    return out


def _argsort_radix(inp, dim, descending):
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
    block = 1024
    log_block = 10
    tiles = triton.cdiv(n, block)
    tile_block = triton.next_power_of_2(tiles)
    bucket_block = min(256, 4096 // tile_block)
    value_bits = inp.element_size() * 8
    key_bits = 64 if value_bits == 64 else 32
    key_dtype = torch.int64 if key_bits == 64 else torch.int32
    passes = value_bits // 8
    byte_only = passes == 1 and inp.dtype in (torch.int8, torch.uint8)
    scratch_shape = (rows, n)
    indices_local = torch.empty(scratch_shape, dtype=torch.int32, device=inp.device)
    if byte_only:
        digits_local = indices_local
    else:
        digits_local = torch.empty(scratch_shape, dtype=torch.uint8, device=inp.device)
    counts = torch.empty((rows, 256, tiles), dtype=torch.int32, device=inp.device)
    offsets = torch.empty((rows, 256, tiles), dtype=torch.int32, device=inp.device)
    split_prefix = False
    bucket_totals = (
        torch.empty((rows, 256), dtype=torch.int32, device=inp.device)
        if split_prefix
        else offsets
    )
    if passes > 1:
        keys_current = torch.empty(scratch_shape, dtype=key_dtype, device=inp.device)
        keys_local = torch.empty(scratch_shape, dtype=key_dtype, device=inp.device)
        indices_current = torch.empty(
            scratch_shape, dtype=torch.int32, device=inp.device
        )
    else:
        keys_current = indices_local
        keys_local = indices_local
        indices_current = indices_local
    warps = 4
    local_warps = warps
    with torch_device_fn.device(inp.device):
        for pass_index in range(passes):
            last = pass_index + 1 == passes
            _radix_local_sort[rows * tiles,](
                inp,
                keys_current,
                indices_current,
                keys_local,
                indices_local,
                digits_local,
                counts,
                n,
                tiles,
                shape,
                strides,
                axis_stride,
                block,
                log_block,
                key_bits,
                pass_index * 8,
                descending,
                pass_index == 0,
                last,
                byte_only,
                local_warps,
                num_warps=local_warps,
            )
            if split_prefix:
                _radix_tile_prefix[rows * 256,](
                    counts, offsets, bucket_totals, tiles, tile_block, num_warps=4
                )
                _radix_bucket_prefix[rows, triton.cdiv(tiles, 16)](
                    counts, offsets, bucket_totals, tiles, 16, num_warps=4
                )
            else:
                _radix_prefix[rows,](
                    counts,
                    offsets,
                    tiles,
                    tile_block,
                    bucket_block,
                    warps,
                    num_warps=warps,
                )
            _radix_scatter[rows * tiles,](
                keys_local,
                indices_local,
                digits_local,
                offsets,
                keys_current,
                indices_current,
                out,
                n,
                tiles,
                block,
                shape,
                out_strides,
                out_axis_stride,
                last,
                byte_only,
                warps,
                num_warps=warps,
            )
    return out


def argsort(inp, dim=-1, descending=False):
    logger.debug("GEMS_HYGON ARGSORT")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    n = inp.shape[dim] if rank else 1
    if (
        inp.numel() > 0
        and 131072 <= n <= 262144
        and (inp.dtype in (torch.float16, torch.bfloat16, torch.int16))
    ):
        return _argsort_radix_fused_narrow(inp, dim, descending)
    if (
        inp.numel() > 0
        and 2048 <= n <= 262144
        and (inp.dtype in (torch.int8, torch.uint8))
    ):
        return _argsort_radix(inp, dim, descending)
    return _argsort_merge_entry(inp, dim, descending)
