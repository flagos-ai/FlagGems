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

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def histc_kernel_simple(
    inp_ptr,
    out_ptr,
    N_total,
    bins,
    min_val,
    max_val,
    inv_bin_width,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        inp_val = tl.load(inp_ptr + off, mask=mask, other=float("nan")).to(tl.float32)
        bin_idx = ((inp_val - min_val) * inv_bin_width).to(tl.int64)
        bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
        in_range = (inp_val >= min_val) & (inp_val <= max_val)
        bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
        bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)
        valid_mask = mask & in_range
        tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid_mask, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def histc_kernel_local(
    inp_ptr,
    out_ptr,
    N_total,
    bins,
    min_val,
    max_val,
    inv_bin_width,
    BLOCK: tl.constexpr,
    NBINS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    bin_range = tl.arange(0, NBINS)
    hist = tl.zeros((NBINS,), dtype=tl.int32)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        inp_val = tl.load(inp_ptr + off, mask=mask, other=float("nan")).to(tl.float32)
        bin_idx = ((inp_val - min_val) * inv_bin_width).to(tl.int32)
        in_range = (inp_val >= min_val) & (inp_val <= max_val)
        bin_idx = tl.minimum(bin_idx, bins - 1)
        valid = mask & in_range
        key = tl.where(valid, bin_idx, NBINS - 1)
        hist += tl.histogram(key, NBINS)
    tl.atomic_add(
        out_ptr + bin_range,
        hist.to(out_ptr.dtype.element_ty),
        mask=bin_range < bins,
        sem="relaxed",
    )


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def histc(inp, bins=100, min=0, max=0):
    logger.debug("GEMS_ENFLAME HISTC")
    inp = inp.contiguous()

    min_val = float(min)
    max_val = float(max)
    if min_val == 0 and max_val == 0:
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())

    if min_val == max_val:
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out

    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
    N = inp.numel()
    if N == 0:
        return out

    NBINS = _next_pow2(bins + 1)
    if NBINS <= 2048:
        if N <= 8192:
            BLOCK = 1024
        elif N <= 98304:
            BLOCK = 2048
        else:
            BLOCK = 4096
        grid = builtins.min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
        with torch_device_fn.device(inp.device):
            histc_kernel_local[(grid,)](
                inp,
                out,
                N,
                bins,
                min_val,
                max_val,
                bins / (max_val - min_val),
                BLOCK=BLOCK,
                NBINS=NBINS,
                num_warps=4,
            )
    else:
        BLOCK = 64
        grid = builtins.min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
        with torch_device_fn.device(inp.device):
            histc_kernel_simple[(grid,)](
                inp,
                out,
                N,
                bins,
                min_val,
                max_val,
                bins / (max_val - min_val),
                BLOCK=BLOCK,
                num_warps=4,
            )
    return out
