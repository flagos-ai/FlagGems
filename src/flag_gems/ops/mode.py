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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.sort import convert_to_uint_preverse_order
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def _mode_sort_histogram(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    # arr_ptr: (m, n)
    # out_ptr: (m, n_passes, r), where r = 2 ** k_bits is the number of bins
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1  # a.k.a. 2 ** k_bits - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):  # parallel
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):  # parallel
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):  # sequantial
                n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask, other=0)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                matches = (n_offsets[None, :] < n) & (
                    bin_indices[:, None] == key[None, :]
                )  # (TILE_R, TILE_N)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )


@libentry()
@triton.jit
def _mode_sort_histogram_by_bucket(
    arr_ptr,
    out_ptr,
    N: tl.constexpr,
    PASSES: tl.constexpr,
    BINS: tl.constexpr,
    BITS: tl.constexpr,
    DESCENDING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    bucket = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    for p in range(PASSES):
        count = tl.full((), 0, tl.int32)
        for start in range(tl.cdiv(N, BLOCK)):
            pos = start * BLOCK + offsets
            arr = tl.load(arr_ptr + row * N + pos, pos < N, other=0)
            key = convert_to_uint_preverse_order(arr, DESCENDING)
            digit = (key >> (p * BITS)) & (BINS - 1)
            match = (pos < N) & (digit == bucket)
            count += tl.sum(match.to(tl.int32), 0)
        tl.store(out_ptr + (row * PASSES + p) * BINS + bucket, count)


@triton.jit
def _mode_sort_sweep(
    arr_ptr,
    associate_arr_ptr,  # inputs: (key & value)
    out_ptr,
    associate_out_ptr,  # outputs: (key & value)
    excumsum_bins_ptr,
    status_ptr,  # aux input and status
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    # r: num_bins = 2 ** k_bits
    # OUT_N: grid_n = cdiv(N, )

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # excumsum_bins_ptr: (m, n_passes, r)
    # flag_ptr: (m, r, OUT_N)

    # grid: (m, grid_r, grid_n)

    # load data
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # cumsum for a bin_index
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)  # (TILE_N, )
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask, other=0)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )
    if associate_arr_ptr is not None:
        associate_arr = tl.load(
            associate_arr_ptr + pid_m * N + n_offsets, mask=mask, other=0
        )
    # since triton can only use scalar as condition, loop by bin_index
    # status must be pre zero-initialized, or else we have to initialize it
    for bin_index in range(cta_r_start, cta_r_end):
        matches = (n_offsets < N) & (key == bin_index)  # (TILE_N, ) bool
        # cta level cumsum per bin
        # CAUTION: tl.sum in triton 3.2 does not promote type
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        # decoupled lookback
        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)  # uin32
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += pack1 & v_mask
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

        local_ex_cumsum = (
            tl.cumsum(matches.to(tl.uint32), axis=0) - matches
        )  # (TILE_N, )
        ex_cumsum_in_bin = (
            exclusive_prefix + local_ex_cumsum
        )  # global ex_cumsum_in_bin (TILE_N, )

        # ex_cumsum_bins (m, n_passes, r)
        ex_cumsum_bins = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
        )  # scalar
        pos = ex_cumsum_bins + ex_cumsum_in_bin  # (TILE_N, )

        # scatter
        tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
        if associate_arr_ptr is not None:
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)


def _mode_radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (m * grid_n, 1, 1)

    with torch_device_fn.device(arr.device):
        global_hist = torch.zeros(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )
        if runtime.device.vendor_name == "ascend":
            # Independent buckets avoid large UB tiles and inter-program atomics.
            _mode_sort_histogram_by_bucket[(m, num_bins)](
                arr, global_hist, n, n_passes, num_bins, k_bits, descending, 512
            )
        else:
            _mode_sort_histogram[grid_for_global_hist](
                arr,
                global_hist,
                n_passes,
                m,
                n,
                tiles_n_per_cta,
                TILE_N,
                TILE_R,
                k_bits,
                descending,
            )
        ex_cumsum_bins = torch.cumsum(global_hist, -1) - global_hist
        ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

        # sort
        arr_in = torch.clone(arr)
        indices_in = (
            torch.arange(0, n, dtype=torch.int64, device=arr_in.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        TILE_R = 8
        grid_r = triton.cdiv(num_bins, TILE_R)
        TILE_N = 2048
        grid_n = triton.cdiv(n, TILE_N)
        grid_for_sweep = (m * grid_n, grid_r)

        status = torch.empty(
            (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
        )

        for i in range(0, n_passes):
            bit_offset = i * k_bits
            if runtime.device.vendor_name == "ascend":
                # torch-npu does not implement zero_ for the uint32 status buffer.
                from flag_gems import zero_

                zero_(status)
            else:
                status.zero_()
            _mode_sort_sweep[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                ex_cumsum_bins,
                status,
                n_passes,
                i,
                bit_offset,
                m,
                n,
                grid_n,
                TILE_N,
                TILE_R,
                k_bits,
                descending,
            )
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


def _mode_sort(inp, dim=-1, descending=False):
    # Keep mode's radix sorting fixes independent of the public sort operator.
    if inp.shape[dim] == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)
    dim %= inp.ndim
    x = inp.movedim(dim, -1).contiguous()
    bits = 1 if inp.dtype == torch.bool else 4
    values, indices = _mode_radix_sort(x, bits, descending)
    return values.movedim(-1, dim), indices.movedim(-1, dim)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def mode_kernel(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M

    # Load first element
    n0 = tl.arange(0, 1)
    offset0 = m_offset[:, None] * N + n0[None, :]
    mask0 = mask_m[:, None] & (n0[None, :] < N)
    val0 = tl.load(sorted_inp + offset0, mask=mask0)
    idx0 = tl.load(sorted_indices + offset0, mask=mask0)

    # Squeeze the n-dimension (size 1)
    cur_value = val0.reshape([BLOCK_M])
    cur_index = idx0.reshape([BLOCK_M])
    cur_count = tl.full([BLOCK_M], 1, dtype=tl.int64)
    best_value = cur_value
    best_index = cur_index
    best_count = tl.full([BLOCK_M], 1, dtype=tl.int64)

    # Scan remaining elements one by one
    for i in range(1, N):
        ni = tl.full([1], i, dtype=tl.int32)
        offset_i = m_offset[:, None] * N + ni[None, :]
        mask_i = mask_m[:, None] & (ni[None, :] < N)
        val_i = tl.load(sorted_inp + offset_i, mask=mask_i).reshape([BLOCK_M])
        idx_i = tl.load(sorted_indices + offset_i, mask=mask_i).reshape([BLOCK_M])

        same = val_i == cur_value
        cur_count = tl.where(same, cur_count + 1, tl.full([BLOCK_M], 1, dtype=tl.int64))
        cur_value = tl.where(same, cur_value, val_i)
        cur_index = idx_i  # Always track the latest index in the run

        better = cur_count > best_count
        best_count = tl.where(better, cur_count, best_count)
        best_value = tl.where(better, cur_value, best_value)
        best_index = tl.where(better, cur_index, best_index)

    tl.store(out_value + m_offset, best_value, mask=mask_m)
    tl.store(out_index + m_offset, best_index, mask=mask_m)


@libentry()
@triton.jit
def _mode_byte_count(
    inp, counts, indices, N: tl.constexpr, LOW: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    bucket = tl.program_id(1)
    value = bucket + LOW
    i = tl.arange(0, B)
    count = tl.full((), 0, tl.int32)
    index = tl.full((), 0, tl.int32)
    for start in range(tl.cdiv(N, B)):
        pos = start * B + i
        val = tl.load(inp + row * N + pos, pos < N, other=0).to(tl.int32)
        match = (pos < N) & (val == value)
        count += tl.sum(match.to(tl.int32), 0)
        index = tl.maximum(index, tl.max(tl.where(match, pos, 0), 0))
    tl.store(counts + row * 256 + bucket, count)
    tl.store(indices + row * 256 + bucket, index)


@libentry()
@triton.jit
def _mode_byte_select(counts, indices, values, out_indices, LOW: tl.constexpr):
    row = tl.program_id(0)
    bucket = tl.arange(0, 256)
    count = tl.load(counts + row * 256 + bucket)
    largest = tl.max(count, 0)
    winner = tl.min(tl.where(count == largest, bucket, 256), 0)
    index = tl.load(indices + row * 256 + winner)
    tl.store(values + row, winner + LOW)
    tl.store(out_indices + row, index)


def _mode_byte(inp, dim, keepdim):
    dim %= inp.ndim
    x = inp.movedim(dim, -1).contiguous()
    n = x.shape[-1]
    rows = x.numel() // n
    counts = torch.empty((rows, 256), dtype=torch.int32, device=x.device)
    indices = torch.empty_like(counts)
    values = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
    out_indices = torch.empty_like(values, dtype=torch.int64)
    if rows:
        low = torch.iinfo(x.dtype).min
        with torch_device_fn.device(inp.device):
            _mode_byte_count[(rows, 256)](x, counts, indices, n, low, 512)
            _mode_byte_select[(rows,)](counts, indices, values, out_indices, low)
    if keepdim:
        values = values.unsqueeze(dim)
        out_indices = out_indices.unsqueeze(dim)
    return namedtuple("mode", ["values", "indices"])(values, out_indices)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS MODE")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if (
        inp.dtype in (torch.int8, torch.uint8)
        and inp.shape[dim] > 0
        and runtime.device.vendor_name in ("ascend", "hygon", "mthreads")
    ):
        return _mode_byte(inp, dim, keepdim)
    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    M = inp.numel() // N

    if runtime.device.vendor_name == "hygon":
        from flag_gems import sort as gems_sort
    else:
        gems_sort = _mode_sort

    sorted_inp, sorted_indices = gems_sort(inp, dim=dim)

    # Move dim to last for 2D processing
    sorted_inp = torch.movedim(sorted_inp, dim, -1).contiguous()
    sorted_indices = torch.movedim(sorted_indices, dim, -1).contiguous()

    sorted_flat = sorted_inp.reshape(M, N)
    indices_flat = sorted_indices.reshape(M, N)

    out_value = torch.empty(M, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(M, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        mode_kernel[grid](sorted_flat, indices_flat, out_value, out_index, M, N)

    out_shape = list(shape)
    out_shape[dim] = 1
    out_value = out_value.reshape(out_shape)
    out_index = out_index.reshape(out_shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    Mode_out = namedtuple("mode", ["values", "indices"])
    return Mode_out(values=out_value, indices=out_index)
