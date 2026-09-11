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

"""Histogram-based top_k_per_row_prefill for DeepSeek V4 sparse attention.

Algorithm (quantile-filter / single-level radix select):
  Stage A: per-row 12-bit histogram over monotonic-uint32 keys (4096 buckets).
  Stage B: per-row scan to find threshold bucket b s.t. sum(hist[b..]) >= K.
  Stage C: per-row compact pass — collect (key, idx) for elements whose bucket
           >= threshold into a fixed-size candidate buffer.
  Stage D: per-row sort the candidate buffer descending; write the top K indices.
"""

import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


@triton.jit
def _hist_kernel(
    LOGITS,
    STARTS,
    ENDS,
    HIST,
    N,
    stride_lm,
    stride_ln,
    stride_hm,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    start = tl.load(STARTS + pid_m).to(tl.int32)
    end = tl.load(ENDS + pid_m).to(tl.int32)

    base = pid_n.to(tl.int32) * BLOCK
    offs = base + tl.arange(0, BLOCK).to(tl.int32)
    in_range = (offs >= start) & (offs < end) & (offs < N)

    v = tl.load(
        LOGITS + pid_m * stride_lm + offs.to(tl.int64) * stride_ln,
        mask=in_range,
        other=float("-inf"),
    )

    bits_u = v.to(tl.uint32, bitcast=True)
    sign_bit = bits_u >> 31
    xor_mask = tl.where(
        sign_bit != 0,
        tl.full([BLOCK], 0xFFFFFFFF, tl.uint32),
        tl.full([BLOCK], 0x80000000, tl.uint32),
    )
    key_u = bits_u ^ xor_mask
    bucket = (key_u >> SHIFT).to(tl.int64)

    tl.atomic_add(HIST + pid_m * stride_hm + bucket, 1, mask=in_range)


@triton.jit
def _thr_kernel(HIST, THR, K, stride_hm, BUCKETS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BUCKETS).to(tl.int64)
    h = tl.load(HIST + pid * stride_hm + offs)

    cs = tl.cumsum(h, axis=0)
    total = tl.sum(h)
    cum_top = total - cs + h
    pass_mask = (cum_top >= K).to(tl.int32)
    n_pass = tl.sum(pass_mask)
    thr_bucket = tl.maximum(n_pass - 1, 0)
    tl.store(THR + pid, thr_bucket)


@triton.jit
def _compact_kernel(
    LOGITS,
    STARTS,
    ENDS,
    THR,
    CAND,
    CAND_CTR,
    N,
    stride_lm,
    stride_ln,
    stride_cm,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    start = tl.load(STARTS + pid_m).to(tl.int32)
    end = tl.load(ENDS + pid_m).to(tl.int32)
    thr = tl.load(THR + pid_m).to(tl.int32)

    base = pid_n.to(tl.int32) * BLOCK
    offs = base + tl.arange(0, BLOCK).to(tl.int32)
    in_range = (offs >= start) & (offs < end) & (offs < N)

    v = tl.load(
        LOGITS + pid_m * stride_lm + offs.to(tl.int64) * stride_ln,
        mask=in_range,
        other=float("-inf"),
    )
    bits_u = v.to(tl.uint32, bitcast=True)
    sign_bit = bits_u >> 31
    xor_mask = tl.where(
        sign_bit != 0,
        tl.full([BLOCK], 0xFFFFFFFF, tl.uint32),
        tl.full([BLOCK], 0x80000000, tl.uint32),
    )
    key_u = bits_u ^ xor_mask
    bucket = (key_u >> SHIFT).to(tl.int32)

    pass_mask = in_range & (bucket >= thr)
    pass_i32 = pass_mask.to(tl.int32)

    cs = tl.cumsum(pass_i32, axis=0)
    n_valid = tl.sum(pass_i32)
    block_base = tl.atomic_add(CAND_CTR + pid_m, n_valid)

    rel_idx = offs - start
    neg1 = tl.full([BLOCK], -1, tl.int32)
    rel_idx = tl.where(in_range, rel_idx, neg1)
    idx_u = rel_idx.to(tl.uint32, bitcast=True)
    pack = (key_u.to(tl.uint64) << 32) | idx_u.to(tl.uint64)

    write_offs = (block_base + cs - 1).to(tl.int64)
    tl.store(
        CAND + pid_m * stride_cm + write_offs,
        pack.to(tl.int64, bitcast=True),
        mask=pass_mask,
    )


@triton.jit
def _sort_cand_kernel(CAND, OUT, K_OUT, stride_cm, stride_om, CAND_PAD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CAND_PAD).to(tl.int64)
    pack_i64 = tl.load(CAND + pid * stride_cm + offs)
    pack_u = pack_i64.to(tl.uint64, bitcast=True)
    sorted_u = tl.sort(pack_u, descending=True)

    mask_low = tl.full([CAND_PAD], 0xFFFFFFFF, tl.uint64)
    low = (sorted_u & mask_low).to(tl.uint32)
    idx_i32 = low.to(tl.int32, bitcast=True)
    out_mask = offs < K_OUT
    tl.store(OUT + pid * stride_om + offs, idx_i32, mask=out_mask)


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Uses a histogram-based quantile-filter algorithm:
    1. Build 12-bit histogram of monotonic-uint32 keys per row
    2. Find threshold bucket via cumulative sum scan
    3. Compact candidates above threshold into a buffer
    4. Sort candidates and emit top-K indices

    Args:
        logits: [num_rows, vocab_size] float32 tensor.
        row_starts: [num_rows] int32 — start of valid range per row (inclusive).
        row_ends: [num_rows] int32 — end of valid range per row (exclusive).
        indices: [num_rows, top_k] int32 — output buffer, filled with 0-based
                 indices relative to row_starts[i]. Caller pre-allocates this.
        num_rows: number of rows.
        stride0: logits.stride(0).
        stride1: logits.stride(1).
        top_k: number of top elements to select per row.
    """
    M = num_rows
    N = logits.shape[1]

    if not logits.is_contiguous():
        logits = logits.contiguous()
    if not row_starts.is_contiguous():
        row_starts = row_starts.contiguous()
    if not row_ends.is_contiguous():
        row_ends = row_ends.contiguous()

    BUCKET_BITS = 12
    BUCKETS = 1 << BUCKET_BITS
    SHIFT = 32 - BUCKET_BITS

    CAND_PAD = _next_pow2(max(2 * top_k, 2048))

    BLOCK_N = 2048
    n_blocks = (N + BLOCK_N - 1) // BLOCK_N

    dev = logits.device
    hist = torch.zeros((M, BUCKETS), dtype=torch.int32, device=dev)
    thr = torch.empty((M,), dtype=torch.int32, device=dev)
    cand = torch.zeros((M, CAND_PAD), dtype=torch.int64, device=dev)
    cand_ctr = torch.zeros((M,), dtype=torch.int32, device=dev)

    _hist_kernel[(M, n_blocks)](
        logits,
        row_starts,
        row_ends,
        hist,
        N,
        logits.stride(0),
        logits.stride(1),
        hist.stride(0),
        BLOCK=BLOCK_N,
        SHIFT=SHIFT,
        num_warps=8,
        num_stages=2,
    )

    _thr_kernel[(M,)](
        hist,
        thr,
        top_k,
        hist.stride(0),
        BUCKETS=BUCKETS,
        num_warps=8,
    )

    _compact_kernel[(M, n_blocks)](
        logits,
        row_starts,
        row_ends,
        thr,
        cand,
        cand_ctr,
        N,
        logits.stride(0),
        logits.stride(1),
        cand.stride(0),
        BLOCK=BLOCK_N,
        SHIFT=SHIFT,
        num_warps=8,
        num_stages=2,
    )

    sort_warps = 16
    _sort_cand_kernel[(M,)](
        cand,
        indices,
        top_k,
        cand.stride(0),
        indices.stride(0),
        CAND_PAD=CAND_PAD,
        num_warps=sort_warps,
        num_stages=2,
    )
