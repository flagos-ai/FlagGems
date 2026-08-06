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

from flag_gems.utils import triton_lang_extension as ext


_BLOCK = 32


@triton.jit
def _ordered_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits >> 31
    return tl.where(sign != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)


@triton.jit
def _init_kernel(
    indices_ptr,
    prefix_ptr,
    rem_ptr,
    row_starts,
    row_ends,
    num_rows,
    top_k,
    TOP_K: tl.constexpr,
):
    row = ext.program_id(0)
    if row < num_rows:
        start = tl.load(row_starts + row).to(tl.int32)
        end = tl.load(row_ends + row).to(tl.int32)
        tl.store(prefix_ptr + row, 0)
        tl.store(rem_ptr + row, tl.minimum(top_k, end - start))
        offsets = tl.arange(0, TOP_K)
        output_offset = row.to(tl.int64) * top_k + offsets
        tl.store(indices_ptr + output_offset, -1, mask=offsets < top_k)


@triton.jit
def _select_threshold_bit_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    prefix_ptr,
    rem_ptr,
    num_rows,
    num_chunks,
    stride0,
    stride1,
    PREFIX_MASK: tl.constexpr,
    BIT: tl.constexpr,
):
    BLOCK: tl.constexpr = 32
    row = ext.program_id(0)
    if row < num_rows:
        start = tl.load(row_starts + row).to(tl.int32)
        end = tl.load(row_ends + row).to(tl.int32)
        row_len = end - start
        prefix = tl.load(prefix_ptr + row).to(tl.uint32)
        rem = tl.load(rem_ptr + row)
        row_offset = row.to(tl.int64) * stride0
        count_one = 0
        for chunk in tl.range(0, num_chunks):
            for lane in tl.static_range(0, BLOCK):
                position = chunk * BLOCK + lane
                if position < row_len:
                    column = (start + position).to(tl.int64) * stride1
                    value = tl.load(logits_ptr + row_offset + column)
                    key = _ordered_key(value)
                    active = (key & PREFIX_MASK) == prefix
                    count_one += (active & ((key & BIT) != 0)).to(tl.int32)

        choose_one = count_one >= rem
        tl.store(prefix_ptr + row, tl.where(choose_one, prefix | BIT, prefix))
        tl.store(rem_ptr + row, tl.where(choose_one, rem, rem - count_one))


@triton.jit
def _chunk_counts_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    threshold_ptr,
    counts_ptr,
    masks_ptr,
    num_rows,
    num_chunks,
    stride0,
    stride1,
):
    BLOCK: tl.constexpr = 32
    pid = ext.program_id(0)
    row = pid // num_chunks
    chunk = pid - row * num_chunks
    if row < num_rows:
        start = tl.load(row_starts + row).to(tl.int32)
        row_len = tl.load(row_ends + row).to(tl.int32) - start
        threshold = tl.load(threshold_ptr + row).to(tl.uint32)
        logits_row_offset = row.to(tl.int64) * stride0
        lanes = tl.arange(0, BLOCK)
        position = chunk * BLOCK + lanes
        valid = position < row_len
        column = (start + position).to(tl.int64) * stride1
        key = _ordered_key(
            tl.load(logits_ptr + logits_row_offset + column, mask=valid, other=0.0)
        )
        greater = valid & (key > threshold)
        equal = valid & (key == threshold)
        count_gt = tl.sum(greater.to(tl.int32), axis=0)
        count_eq = tl.sum(equal.to(tl.int32), axis=0)
        mask_gt = tl.sum(greater.to(tl.uint32) << lanes, axis=0)
        mask_eq = tl.sum(equal.to(tl.uint32) << lanes, axis=0)

        row_offset = row.to(tl.int64) * num_chunks * 2
        chunk_offset = chunk.to(tl.int64)
        equal_offset = num_chunks + chunk_offset
        tl.store(counts_ptr + row_offset + chunk_offset, count_gt)
        tl.store(masks_ptr + row_offset + chunk_offset, mask_gt)
        tl.store(counts_ptr + row_offset + equal_offset, count_eq)
        tl.store(masks_ptr + row_offset + equal_offset, mask_eq)


@triton.jit
def _chunk_prefix_kernel(counts_ptr, prefix_ptr, num_rows, total_chunks):
    row = ext.program_id(0)
    if row < num_rows:
        row_offset = row.to(tl.int64) * total_chunks
        prefix = 0
        for chunk in tl.range(0, total_chunks):
            count = tl.load(counts_ptr + row_offset + chunk)
            tl.store(prefix_ptr + row_offset + chunk, prefix)
            prefix += count


@triton.jit
def _binary_search_step_kernel(
    row_starts,
    row_ends,
    counts_ptr,
    prefix_ptr,
    lo_ptr,
    hi_ptr,
    num_rows,
    total_chunks,
    top_k,
):
    pid = ext.program_id(0)
    row = pid // top_k
    rank = pid % top_k
    if row < num_rows:
        row_len = tl.load(row_ends + row).to(tl.int32) - tl.load(
            row_starts + row
        ).to(tl.int32)
        if rank < tl.minimum(top_k, row_len):
            output_offset = row.to(tl.int64) * top_k + rank
            lo = tl.load(lo_ptr + output_offset)
            hi = tl.load(hi_ptr + output_offset)
            mid = (lo + hi) // 2
            row_offset = row.to(tl.int64) * total_chunks
            prefix = tl.load(prefix_ptr + row_offset + mid)
            count = tl.load(counts_ptr + row_offset + mid)
            move_right = rank >= prefix + count
            tl.store(lo_ptr + output_offset, tl.where(move_right, mid + 1, lo))
            tl.store(hi_ptr + output_offset, tl.where(move_right, hi, mid))


@triton.jit
def _scatter_by_rank_kernel(
    row_starts,
    row_ends,
    prefix_ptr,
    masks_ptr,
    chunks_ptr,
    indices_ptr,
    num_rows,
    num_chunks,
    top_k,
):
    BLOCK: tl.constexpr = 32
    pid = ext.program_id(0)
    row = pid // top_k
    rank = pid % top_k
    if row < num_rows:
        row_len = tl.load(row_ends + row).to(tl.int32) - tl.load(
            row_starts + row
        ).to(tl.int32)
        if rank < tl.minimum(top_k, row_len):
            output_offset = row.to(tl.int64) * top_k + rank
            total_chunks = num_chunks * 2
            row_offset = row.to(tl.int64) * total_chunks
            chunk = tl.load(chunks_ptr + output_offset)
            chunk_prefix = tl.load(prefix_ptr + row_offset + chunk)
            target = rank - chunk_prefix
            mask = tl.load(masks_ptr + row_offset + chunk).to(tl.uint32)
            seen = 0
            lane_result = 0
            for lane in tl.static_range(0, BLOCK):
                selected = ((mask >> lane) & 1) != 0
                lane_result = tl.where(selected & (seen == target), lane, lane_result)
                seen += selected.to(tl.int32)
            physical_chunk = tl.where(chunk >= num_chunks, chunk - num_chunks, chunk)
            tl.store(
                indices_ptr + output_offset,
                physical_chunk * BLOCK + lane_result,
            )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    assert num_rows == logits.shape[0]
    max_rows_per_launch = 256
    if num_rows > max_rows_per_launch:
        for row_start in range(0, num_rows, max_rows_per_launch):
            row_end = min(row_start + max_rows_per_launch, num_rows)
            top_k_per_row_prefill(
                logits[row_start:row_end],
                row_starts[row_start:row_end],
                row_ends[row_start:row_end],
                indices[row_start:row_end],
                row_end - row_start,
                stride0,
                stride1,
                top_k,
            )
        return

    num_chunks = triton.cdiv(logits.shape[1], _BLOCK)
    total_chunks = num_chunks * 2
    prefix = torch.empty((num_rows,), dtype=torch.int32, device=logits.device)
    rem = torch.empty((num_rows,), dtype=torch.int32, device=logits.device)

    _init_kernel[(num_rows,)](
        indices,
        prefix,
        rem,
        row_starts,
        row_ends,
        num_rows,
        top_k,
        TOP_K=triton.next_power_of_2(top_k),
    )
    for shift in range(31, -1, -1):
        prefix_mask = (0xFFFFFFFF << (shift + 1)) & 0xFFFFFFFF
        _select_threshold_bit_kernel[(num_rows,)](
            logits,
            row_starts,
            row_ends,
            prefix,
            rem,
            num_rows,
            num_chunks,
            stride0,
            stride1,
            PREFIX_MASK=prefix_mask,
            BIT=1 << shift,
            num_warps=1,
            isCloseUnrollControl=True,
        )

    counts = torch.empty(
        (num_rows, total_chunks), dtype=torch.int32, device=logits.device
    )
    masks = torch.empty_like(counts)
    chunk_prefix = torch.empty_like(counts)
    chunk_count_batch_rows = 256
    for row_start in range(0, num_rows, chunk_count_batch_rows):
        row_end = min(row_start + chunk_count_batch_rows, num_rows)
        batch_rows = row_end - row_start
        _chunk_counts_kernel[(batch_rows * num_chunks,)](
            logits[row_start:row_end],
            row_starts[row_start:row_end],
            row_ends[row_start:row_end],
            prefix[row_start:row_end],
            counts[row_start:row_end],
            masks[row_start:row_end],
            batch_rows,
            num_chunks,
            stride0,
            stride1,
            num_warps=1,
            isCloseUnrollControl=True,
        )
    _chunk_prefix_kernel[(num_rows,)](
        counts,
        chunk_prefix,
        num_rows,
        total_chunks,
        num_warps=1,
        isCloseUnrollControl=True,
    )

    search_steps = max(1, (total_chunks - 1).bit_length())
    search_batch_rows = 256
    for row_start in range(0, num_rows, search_batch_rows):
        row_end = min(row_start + search_batch_rows, num_rows)
        batch_rows = row_end - row_start
        search_size = batch_rows * top_k
        lo = torch.zeros((batch_rows, top_k), dtype=torch.int32, device=logits.device)
        hi = torch.full_like(lo, total_chunks)
        for _ in range(search_steps):
            _binary_search_step_kernel[(search_size,)](
                row_starts[row_start:row_end],
                row_ends[row_start:row_end],
                counts[row_start:row_end],
                chunk_prefix[row_start:row_end],
                lo,
                hi,
                batch_rows,
                total_chunks,
                top_k,
                num_warps=1,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
            torch.cuda.synchronize()
        _scatter_by_rank_kernel[(search_size,)](
            row_starts[row_start:row_end],
            row_ends[row_start:row_end],
            chunk_prefix[row_start:row_end],
            masks[row_start:row_end],
            lo,
            indices[row_start:row_end],
            batch_rows,
            num_chunks,
            top_k,
            SINGLE_ROW=search_batch_rows == 1,
            num_warps=1,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        torch.cuda.synchronize()
    return indices
