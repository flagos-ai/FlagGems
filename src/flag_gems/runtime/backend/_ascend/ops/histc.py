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

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _init_minmax_kernel(minmax_ptr):
    offsets = tl.arange(0, 2)
    values = tl.where(offsets == 0, float("inf"), -float("inf"))
    tl.store(minmax_ptr + offsets, values)


@triton.jit
def _minmax_kernel(inp_ptr, minmax_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    local_min = tl.min(tl.where(mask, values, float("inf")), axis=0)
    local_max = tl.max(tl.where(mask, values, -float("inf")), axis=0)
    tl.atomic_min(minmax_ptr, local_min)
    tl.atomic_max(minmax_ptr + 1, local_max)


@triton.jit
def _partial_hist_kernel(
    inp_ptr,
    partial_ptr,
    minmax_ptr,
    n_elements: tl.constexpr,
    bins: tl.constexpr,
    min_value: tl.constexpr,
    max_value: tl.constexpr,
    INFER_RANGE: tl.constexpr,
    BLOCK: tl.constexpr,
    HIST_BINS: tl.constexpr,
):
    pid = tl.program_id(0)
    nprograms = tl.num_programs(0)
    local_hist = tl.zeros((HIST_BINS,), dtype=tl.int32)

    if INFER_RANGE:
        lo = tl.load(minmax_ptr)
        hi = tl.load(minmax_ptr + 1)
        equal = lo == hi
        lo = tl.where(equal, lo - 1.0, lo)
        hi = tl.where(equal, hi + 1.0, hi)
    else:
        lo = min_value
        hi = max_value

    for block_start in tl.range(pid * BLOCK, n_elements, nprograms * BLOCK):
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scaled = (values - lo) * bins / (hi - lo)
        bin_idx = scaled.to(tl.int32)
        bin_idx = tl.where(values == hi, bins - 1, bin_idx)
        valid = (
            mask & (values >= lo) & (values <= hi) & (bin_idx >= 0) & (bin_idx < bins)
        )
        histogram_input = tl.where(valid, bin_idx, HIST_BINS - 1)
        local_hist += tl.histogram(histogram_input, HIST_BINS)

    bin_offsets = tl.arange(0, HIST_BINS)
    tl.store(
        partial_ptr + pid * bins + bin_offsets, local_hist, mask=bin_offsets < bins
    )


@triton.jit
def _reduce_hist_kernel(
    partial_ptr,
    out_ptr,
    nprograms: tl.constexpr,
    bins: tl.constexpr,
    PROGRAM_BLOCK: tl.constexpr,
    BIN_BLOCK: tl.constexpr,
):
    program_offsets = tl.arange(0, PROGRAM_BLOCK)[:, None]
    bin_offsets = tl.arange(0, BIN_BLOCK)[None, :]
    mask = (program_offsets < nprograms) & (bin_offsets < bins)
    values = tl.load(
        partial_ptr + program_offsets * bins + bin_offsets, mask=mask, other=0
    )
    totals = tl.sum(values, axis=0)
    store_offsets = tl.arange(0, BIN_BLOCK)
    tl.store(out_ptr + store_offsets, totals.to(tl.float32), mask=store_offsets < bins)


def histc(inp, bins=100, min=0, max=0):
    out = torch.empty((bins,), dtype=inp.dtype, device=inp.device)
    n_elements = inp.numel()
    block = 128 if n_elements <= 4096 else 1024
    max_programs = driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )["num_vectorcore"]
    nprograms = triton.cdiv(n_elements, block)
    if nprograms > max_programs:
        nprograms = max_programs

    infer_range = min == 0 and max == 0
    if infer_range:
        minmax = torch.empty((2,), dtype=torch.float32, device=inp.device)
        _init_minmax_kernel[(1,)](minmax)
        _minmax_kernel[(triton.cdiv(n_elements, block),)](
            inp, minmax, n_elements=n_elements, BLOCK=block
        )
    else:
        minmax = out

    partial = torch.empty((nprograms, bins), dtype=torch.int32, device=inp.device)
    hist_bins = triton.next_power_of_2(bins + 1)
    _partial_hist_kernel[(nprograms,)](
        inp,
        partial,
        minmax,
        n_elements=n_elements,
        bins=bins,
        min_value=min,
        max_value=max,
        INFER_RANGE=infer_range,
        BLOCK=block,
        HIST_BINS=hist_bins,
    )
    _reduce_hist_kernel[(1,)](
        partial,
        out,
        nprograms=nprograms,
        bins=bins,
        PROGRAM_BLOCK=triton.next_power_of_2(nprograms),
        BIN_BLOCK=triton.next_power_of_2(bins),
    )
    return out
