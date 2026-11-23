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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metax-specific constants.
# Metax hardware limits:
#   - max 512 threads per block → num_warps ≤ 16
#   - ~4 KB private memory per thread
#
# Strategy:
#   - Small N (≤ 4K):   single block, power-of-2 BLOCK_SIZE
#   - Medium N (≤ 64K): BLOCK_SIZE = 512,  moderate block count
#   - Large N (> 64K):  BLOCK_SIZE = 2048, keep block count low
#   - Cross-CTA scan uses chunked vector ops (chunk ≤ 512 int32s = 2 KB)
# ---------------------------------------------------------------------------

_MAX_BLOCK_SIZE = 2048
_MIN_BLOCK_SIZE = 64
_MAX_WARPS = 8  # 256 threads — well within 512 limit
_SCAN_CHUNK = 512  # 512 × 4 bytes = 2 KB < 4 KB private memory limit


@libentry()
@triton.jit(do_not_specialize=["N"])
def _msb_count_kernel(
    mask_ptr,
    counts_ptr,
    counter_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_vals = tl.load(mask_ptr + offsets, mask=offsets < N, other=0)
    count = tl.sum(mask_vals.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, count)

    # ---- cross-CTA exclusive scan (run by the last CTA) ----
    barrier = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    np = ext.num_programs(axis=0)
    if barrier == np - 1:
        # Chunked vector scan: process SCAN_CHUNK counts at a time to stay
        # within the 4 KB/thread private memory budget on Metax.
        # SCAN_CHUNK=512 → 512 int32s = 2 KB — well under the limit.
        running_sum = 0
        SCAN_CHUNK: tl.constexpr = 512
        for chunk_start in range(0, np, SCAN_CHUNK):
            chunk_off = tl.arange(0, SCAN_CHUNK)
            abs_off = chunk_start + chunk_off
            chunk_mask = abs_off < np
            counts = tl.load(counts_ptr + abs_off, mask=chunk_mask, other=0)
            pre_sums = running_sum + tl.cumsum(counts, axis=0) - counts
            tl.store(counts_ptr + abs_off, pre_sums, mask=chunk_mask)
            running_sum += tl.sum(counts, axis=0)
        tl.store(counts_ptr + np, running_sum)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _msb_write_kernel(
    grad_ptr,
    mask_ptr,
    counts_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_val = tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=offsets < N, other=0)

    global_offset = tl.load(counts_ptr + pid)

    mask_ints = mask_val.to(tl.int32)
    local_pos = tl.cumsum(mask_ints, axis=0) - 1  # inclusive → exclusive

    pos = global_offset + local_pos
    tl.store(out_ptr + pos, grad_val, mask=(offsets < N) & mask_val)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _get_launch_params(N):
    """Choose BLOCK_SIZE, n_blocks, num_warps for Metax hardware.

    Tiered strategy:
      - N ≤ 4096:    single block, power-of-2 BLOCK_SIZE (fastest)
      - N ≤ 65536:   BLOCK_SIZE = 512, moderate parallelism
      - N > 65536:   BLOCK_SIZE = 2048, fewer blocks → less overhead
    """
    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(max(N, _MIN_BLOCK_SIZE))
        n_blocks = 1
    elif N <= 65536:
        BLOCK_SIZE = 512
        n_blocks = triton.cdiv(N, BLOCK_SIZE)
    else:
        BLOCK_SIZE = _MAX_BLOCK_SIZE  # 2048
        n_blocks = triton.cdiv(N, BLOCK_SIZE)
    num_warps = min(_MAX_WARPS, max(4, BLOCK_SIZE // 32))
    return BLOCK_SIZE, n_blocks, num_warps


def _masked_scatter_backward_impl(grad_output, mask, numel):
    """Custom multi-block path adapted for Metax hardware."""
    N = mask.numel()

    if N == 0:
        return torch.zeros(numel, dtype=grad_output.dtype, device=grad_output.device)

    BLOCK_SIZE, n_blocks, num_warps = _get_launch_params(N)

    out = torch.zeros(numel, dtype=grad_output.dtype, device=grad_output.device)

    with torch_device_fn.device(grad_output.device):
        counts = torch.empty(n_blocks + 1, dtype=torch.int32, device=grad_output.device)
        barrier = torch.zeros([], dtype=torch.int32, device=grad_output.device)

        _msb_count_kernel[(n_blocks,)](
            mask,
            counts,
            barrier,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        _msb_write_kernel[(n_blocks,)](
            grad_output,
            mask,
            counts,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    logger.debug("GEMS_METAX MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    if N == 0:
        return torch.zeros(
            numel, dtype=grad_output.dtype, device=grad_output.device
        ).view(sizes)

    out = _masked_scatter_backward_impl(grad_output, mask, numel)
    return out.view(sizes)
