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

from flag_gems.ops.mode import _mode_sort_histogram, _mode_sort_sweep
from flag_gems.ops.topk import _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

ModeOut = namedtuple("mode", ["values", "indices"])


@libentry()
@triton.jit
def _mode_fused(X, V, out_indices, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    if X.dtype.element_ty.is_floating():
        limit = float("inf")
    else:
        limit = _get_iinfo_val(X.dtype.element_ty, return_max=True)
    x = tl.load(X + row * N + c, c < N, other=limit)
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        x = x.to(tl.float32)
    if B == 1:
        ordered = x
    else:
        ordered = tl.sort(x, descending=False)
    ones = tl.full((B,), 1, tl.int32)
    _, count, _ = tl.associative_scan((ordered, ones, ones), 0, _run_count)
    count = tl.where(c < N, count, 0)
    most = tl.max(count, 0)
    at = tl.min(tl.where(count == most, c, B), 0)
    value = tl.sum(tl.where(c == at, ordered, 0), 0)
    index = tl.min(
        tl.where((c < N) & ((x == value) | ((x != x) & (value != value))), c, B), 0
    )
    tl.store(V + row, value)
    tl.store(out_indices + row, index)


@triton.jit
def _run_count(av, ac, al, bv, bc, bl):
    # A concatenation extends the left run only if the right segment is uniform.
    count = tl.where((bc == bl) & (av == bv), ac + bc, bc)
    return bv, count, al + bl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.jit
def _mode_sorted(X, IX, V, out_indices, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    carry = 0
    best_count = 0
    best_pos = 0
    for base in range(tl.cdiv(N, B)):
        p = base * B + c
        x = tl.load(X + row * N + p, p < N, other=0)
        prev_p = tl.minimum(tl.maximum(p - 1, 0), N - 1)
        prev = tl.load(X + row * N + prev_p)
        starts = tl.where((p == 0) | (x != prev), p, carry)
        starts = tl.associative_scan(starts, 0, _maximum)
        count = tl.where(p < N, p - starts + 1, 0)
        most = tl.max(count, 0)
        at = tl.min(tl.where(count == most, p, N), 0)
        better = most > best_count
        best_pos = tl.where(better, at, best_pos)
        best_count = tl.maximum(best_count, most)
        carry = tl.max(tl.where(p < N, starts, 0), 0)
    value = tl.load(X + row * N + best_pos)
    index = tl.load(IX + row * N + best_pos)
    tl.store(V + row, value)
    tl.store(out_indices + row, index)


def _radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < 1 << 30, "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else arr.itemsize * 8
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
        # MACA lacks UInt32 copy kernels; counts fit in signed 32-bit storage.
        ex_cumsum_bins = ex_cumsum_bins.to(torch.int32).view(torch.uint32)
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
            # Preserve the unsigned state flags while using MACA's INT32 fill.
            status.view(torch.int32).zero_()
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
            arr_in, arr_out = (arr_out, arr_in)
            indices_in, indices_out = (indices_out, indices_in)
    return (arr_in, indices_in)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_METAX MODE")
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    if inp.dtype in (torch.int8, torch.uint8) and inp.shape[dim] > 0:
        from flag_gems.ops.mode import _mode_byte

        return _mode_byte(inp, dim, keepdim)
    dim %= inp.ndim
    x = inp.movedim(dim, -1).contiguous()
    n = x.shape[-1]
    rows = x.numel() // n
    values = torch.empty(x.shape[:-1], dtype=inp.dtype, device=inp.device)
    indices = torch.empty(x.shape[:-1], dtype=torch.int64, device=inp.device)
    if rows:
        with torch_device_fn.device(inp.device):
            if n <= 4096:
                _mode_fused[(rows,)](x, values, indices, n, triton.next_power_of_2(n))
            else:
                sorted_values, sorted_indices = _radix_sort(
                    x, k_bits=1 if inp.dtype == torch.bool else 4
                )
                _mode_sorted[(rows,)](
                    sorted_values, sorted_indices, values, indices, n, 1024
                )
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return ModeOut(values, indices)


__all__ = ["mode"]
