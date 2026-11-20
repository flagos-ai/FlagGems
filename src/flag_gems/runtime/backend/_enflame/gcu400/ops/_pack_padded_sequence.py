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

"""GCU400-specific ``_pack_padded_sequence``: reduce host metadata and H2D overhead.

Packed rows are ordered time-step major, and the ``row -> (t, b)`` mapping is derived from lengths. The original
implementation builds two equal-length index tables on the host with one H2D each (synchronous copy), whose fixed
overhead dominates for small shapes. This implementation offers three switchable paths: the original 2D grid with two
tables, a 1D grid with merged tables (one H2D), and a small table carrying only ``batch_sizes`` prefix sums whose
row indices are reconstructed inside the kernel. Small shapes use the merged / small-table path; large shapes keep
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)

# the vectorized host construction.
# Cache the read-only cumulative row-count table by lengths contents; the table depends entirely on CPU lengths,
_TABLE_CACHE = {}
_TABLE_CACHE_CAP = 64


def _get_cumsum_table(lengths_cpu, cumsum, device):
    """A cache hit saves one synchronous H2D; the small table is built and uploaded only on a miss."""
    cache_key = (str(device), tuple(lengths_cpu.tolist()))
    cached_table = _TABLE_CACHE.get(cache_key)
    if cached_table is None:
        cached_table = cumsum.to(device)
        if len(_TABLE_CACHE) >= _TABLE_CACHE_CAP:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[cache_key] = cached_table
    return cached_table


# --- TUNING PARAMS START ---
# Compile-time / launch-time config, adjusted per optimization round; semantics unchanged.
META_MODE = 2
GRID_MODE = 1
BLOCK_FEAT_CAP = 1024
NUM_WARPS = 1
NUM_STAGES = 1
FAST_HOST_META = 0
FAST_META_LIMIT = 2048
TABLE_CAP = 64
NON_BLOCKING = 0
# --- TUNING PARAMS END ---


@triton.jit
def _pack_copy_kernel(
    input_ptr,
    data_ptr,
    t_ptr,
    b_ptr,
    stride_t,
    stride_b,
    feat_size,
    BLOCK_FEAT: tl.constexpr,
):
    """Original 2D-grid implementation: one program per row, reading (t, b) from two index tables."""
    row = tl.program_id(0)
    feat_block = tl.program_id(1)

    t = tl.load(t_ptr + row)
    b = tl.load(b_ptr + row)

    offsets = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = offsets < feat_size

    src = t * stride_t + b * stride_b + offsets
    dst = row * feat_size + offsets

    vals = tl.load(input_ptr + src, mask=feat_mask, other=0.0)
    tl.store(data_ptr + dst, vals, mask=feat_mask)


@triton.jit
def _pack_copy_kernel_1d(
    input_ptr,
    data_ptr,
    meta_ptr,
    stride_t,
    stride_b,
    feat_size,
    NUM_FEAT_BLOCKS: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
):
    """1D grid: one H2D of the merged [total_len, 2] index table; pid is linearized over (row, feat_block)."""
    pid = tl.program_id(0)
    row = pid // NUM_FEAT_BLOCKS
    feat_block = pid % NUM_FEAT_BLOCKS

    t = tl.load(meta_ptr + row * 2)
    b = tl.load(meta_ptr + row * 2 + 1)

    offsets = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = offsets < feat_size
    src = t * stride_t + b * stride_b + offsets
    dst = row * feat_size + offsets

    vals = tl.load(input_ptr + src, mask=feat_mask, other=0.0)
    tl.store(data_ptr + dst, vals, mask=feat_mask)


@triton.jit
def _pack_copy_kernel_table(
    input_ptr,
    data_ptr,
    cumsum_ptr,
    max_len,
    stride_t,
    stride_b,
    feat_size,
    NUM_FEAT_BLOCKS: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
    LOG_STEPS: tl.constexpr,
):
    """1D grid: pass only per-time-step cumulative row counts and reconstruct (t, b) from row inside the kernel.

    ``cumsum[t]`` is the total number of packed rows in the first t+1 time steps; the smallest t satisfying
    ``cumsum[t]`` is the total number of packed rows in the first t+1 time steps; the smallest t satisfying
    ``row < cumsum[t]`` is the time step, and the within-row index is ``row - cumsum[t-1]``. The table length is
    """
    pid = tl.program_id(0)
    row = pid // NUM_FEAT_BLOCKS
    feat_block = pid % NUM_FEAT_BLOCKS

    # only max_len, far below total_len, so small shapes avoid the H2D sync of a large index table.
    low_step = 0
    high_step = max_len
    for _ in tl.static_range(LOG_STEPS):
        mid_step = (low_step + high_step) // 2
        end_at_mid = tl.load(
            cumsum_ptr + mid_step, mask=mid_step < max_len, other=2147483647
        )
        active_at_mid = row >= end_at_mid
        low_step = tl.where(active_at_mid, mid_step + 1, low_step)
        high_step = tl.where(active_at_mid, high_step, mid_step)
    t = low_step
    prev_end = tl.load(cumsum_ptr + t - 1, mask=t > 0, other=0)
    b = row - prev_end

    offsets = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = offsets < feat_size
    src = t * stride_t + b * stride_b + offsets
    dst = row * feat_size + offsets

    vals = tl.load(input_ptr + src, mask=feat_mask, other=0.0)
    tl.store(data_ptr + dst, vals, mask=feat_mask)


def _pack_padded_sequence(input, lengths, batch_first):
    logger.debug("GEMS_ENFLAME_GCU400 _PACK_PADDED_SEQUENCE")

    if batch_first:
        batch_size = input.shape[0]
        stride_b, stride_t = input.stride(0), input.stride(1)
    else:
        batch_size = input.shape[1]
        stride_t, stride_b = input.stride(0), input.stride(1)

    feat_shape = input.shape[2:]
    feat_size = 1
    for size in feat_shape:
        feat_size *= size

    # Since cumsum is strictly increasing, binary search for the time step uses fewer instructions than a
    # BLOCK_STEPS-element vector reduction.
    # ATen guarantees lengths is CPU int64 and non-increasing; all metadata computation stays on the host,
    lengths_cpu = lengths.to(torch.int64).cpu()
    if batch_size == 0:
        data = torch.empty(0, *feat_shape, dtype=input.dtype, device=input.device)
        return data, torch.empty(0, dtype=torch.int64)

    max_len = int(lengths_cpu[0].item())
    total_len = int(lengths_cpu.sum().item())

    steps = torch.arange(max_len, dtype=torch.int64).unsqueeze(1)
    active = lengths_cpu.unsqueeze(0) > steps
    batch_sizes = active.sum(dim=1)
    cumsum = flag_gems.cumsum(batch_sizes, dim=0)

    data = torch.empty(total_len, *feat_shape, dtype=input.dtype, device=input.device)

    if total_len > 0 and feat_size > 0:
        block_feat = min(triton.next_power_of_2(feat_size), BLOCK_FEAT_CAP)
        num_feat_blocks = triton.cdiv(feat_size, block_feat)
        meta_mode = META_MODE
        # avoiding an extra device-side query per packed row.
        if meta_mode == 2 and max_len > TABLE_CAP:
            meta_mode = 1

        if meta_mode == 0:
            # The small-table path needs one reduction over max_len in the kernel, so a very large max_len is not
            # worthwhile.
            step_ids = torch.arange(max_len, dtype=torch.int64)
            time_index = torch.repeat_interleave(step_ids, batch_sizes)
            run_starts = torch.repeat_interleave(cumsum - batch_sizes, batch_sizes)
            batch_index = torch.arange(total_len, dtype=torch.int64) - run_starts
            row2t = time_index.to(input.device)
            row2b = batch_index.to(input.device)
            grid = (total_len, num_feat_blocks)
            _pack_copy_kernel[grid](
                input,
                data,
                row2t,
                row2b,
                stride_t,
                stride_b,
                feat_size,
                BLOCK_FEAT=block_feat,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
        elif meta_mode == 1:
            # Original path: two independent index tables + 2D grid.
            if FAST_HOST_META and total_len <= FAST_META_LIMIT:
                pairs = []
                for time_step, count in enumerate(batch_sizes.tolist()):
                    pairs.extend(
                        (time_step, batch_index) for batch_index in range(int(count))
                    )
                meta_cpu = torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)
            else:
                step_ids = torch.arange(max_len, dtype=torch.int64)
                time_index = torch.repeat_interleave(step_ids, batch_sizes)
                run_starts = torch.repeat_interleave(cumsum - batch_sizes, batch_sizes)
                batch_index = torch.arange(total_len, dtype=torch.int64) - run_starts
                meta_cpu = flag_gems.stack(
                    (time_index, batch_index), dim=1
                ).contiguous()
            meta = meta_cpu.to(input.device, non_blocking=bool(NON_BLOCKING))
            if GRID_MODE:
                grid = (total_len * num_feat_blocks,)
                _pack_copy_kernel_1d[grid](
                    input,
                    data,
                    meta,
                    stride_t,
                    stride_b,
                    feat_size,
                    num_feat_blocks,
                    block_feat,
                    num_warps=NUM_WARPS,
                    num_stages=NUM_STAGES,
                )
            else:
                # Merged index table: one H2D for both (t, b) columns.
                row2t = meta[:, 0]
                row2b = meta[:, 1]
                grid = (total_len, num_feat_blocks)
                _pack_copy_kernel[grid](
                    input,
                    data,
                    row2t,
                    row2b,
                    stride_t,
                    stride_b,
                    feat_size,
                    BLOCK_FEAT=block_feat,
                    num_warps=NUM_WARPS,
                    num_stages=NUM_STAGES,
                )
        else:
            # The merged table can also be read as two independent views under a 2D grid.
            table = _get_cumsum_table(lengths_cpu, cumsum, input.device)
            grid = (total_len * num_feat_blocks,)
            _pack_copy_kernel_table[grid](
                input,
                data,
                table,
                max_len,
                stride_t,
                stride_b,
                feat_size,
                num_feat_blocks,
                block_feat,
                max(1, (max_len + 1).bit_length()),
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )

    return data, batch_sizes
