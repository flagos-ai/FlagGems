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

logger = logging.getLogger(__name__)

# Encoding bit width: the low PREFIX_BITS bits hold the prefix value, the high bits hold the entry index.
# prefix upper bound = batch * max_ctx / split_kv, in practice far below 2^20.
PREFIX_BITS = 20

# Number of programs that can run concurrently in one wave on GCU400 (4 DIE × 6 SIP). Measurements show that once grid
# exceeds 24,
# program scheduling overhead rises linearly; below 24 the in-kernel serial SM loop becomes longer. Hence fixed at 24.
GRID_CAP = 24


@triton.jit
def _paged_mqa_logits_metadata_kernel(
    context_lens_ptr,
    context_lens_stride,
    schedule_metadata_ptr,
    batch_size,
    split_kv,
    num_sms,
    BLOCK_SIZE: tl.constexpr,
    GRID_DIM_X: tl.constexpr,
    PREFIX_BITS_C: tl.constexpr,
):
    """Each program owns the (q_idx, kv_split_idx) scheduling metadata for a batch of SMs.

    The old version used grid = num_sms + 1, reloaded the whole context_lens and recomputed the
    cumsum in every program, and needed 2 compares + 2 reduces per SM. Here:
      1) the prefix sum is hoisted out of the SM loop;
      2) (entry index, prefix value) are packed into one int32 and reduced with a single max,
         yielding both q_idx and the last segment boundary before this SM;
      3) the grid is fixed to one-wave width 24 and remaining SMs are covered by an in-kernel stride loop.
    """
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # 1. Semantic context length -> number of KV segments per sequence (out-of-bounds positions zeroed)
    ctx_lens = tl.load(
        context_lens_ptr + offsets * context_lens_stride, mask=mask, other=0
    )
    num_segs = (ctx_lens + split_kv - 1) // split_kv
    num_segs = tl.where(mask, num_segs, 0)

    # 2. The inclusive prefix sum of the whole batch is computed only once per program
    prefix_sum = tl.cumsum(num_segs, axis=0)
    total_segs = tl.max(prefix_sum)

    # 3. Set out-of-bounds lanes to total_segs + 1 so subsequent comparisons need no mask
    prefix_padded = tl.where(mask, prefix_sum, total_segs + 1)

    # 4. Starting segment number of each SM = sm * q + min(sm, r) (remainder distributed first)
    q = total_segs // num_sms
    r = total_segs % num_sms

    for sm_idx in tl.range(pid, num_sms + 1, GRID_DIM_X):
        seg_starts = sm_idx * q + tl.minimum(sm_idx, r)

        # 5. prefix_sum is monotonically non-decreasing: the last position <= seg_starts is q_idx - 1
        if BLOCK_SIZE <= 2048:
            # Fast path: entry index fits in the high int32 bits (index < 2^11).
            packed = tl.where(
                prefix_padded <= seg_starts,
                (offsets << PREFIX_BITS_C) | prefix_sum,
                -1,
            )
        else:
            # Large batch: index << PREFIX_BITS_C overflows int32, pack in int64.
            packed = tl.where(
                prefix_padded <= seg_starts,
                (offsets.to(tl.int64) << PREFIX_BITS_C) | prefix_sum,
                -1,
            )
        best = tl.max(packed)
        last_hit = best >> PREFIX_BITS_C
        q_idx = tl.where(best >= 0, last_hit + 1, 0)
        prev_prefix = tl.where(best >= 0, best & ((1 << PREFIX_BITS_C) - 1), 0)

        kv_split_idx = seg_starts - prev_prefix

        out_idx = sm_idx * 2
        tl.store(schedule_metadata_ptr + out_idx, q_idx.to(tl.int32))
        tl.store(schedule_metadata_ptr + out_idx + 1, kv_split_idx.to(tl.int32))


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME GET_PAGED_MQA_LOGITS_METADATA")
    # DeepGEMM cuts each sequence into KV blocks of `block_size` (its `block_kv`).
    # The caller passes the paged-cache block size, so a hardcoded 256 produces a
    # different partitioning from the reference whenever the caller does not use 256.
    split_kv = block_size
    device = context_lens.device

    # 1. Handle 1D / 2D input
    if context_lens.dim() == 2:
        batch_size, next_n = context_lens.shape
        effective_context_lens = context_lens[:, next_n - 1]
    else:
        batch_size = context_lens.shape[0]
        effective_context_lens = context_lens

    # Edge case: if batch is 0, return all zeros
    if batch_size == 0:
        return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=device)

    # 2. grid is fixed to the single-wave width; remaining SMs are covered by the in-kernel strided loop
    grid_dim_x = max(1, min(num_sms + 1, GRID_CAP))
    grid = (grid_dim_x,)

    # Auto-get next power of 2 as BLOCK_SIZE to cover entire batch
    BLOCK_SIZE = triton.next_power_of_2(max(16, batch_size))

    # 3. Launch kernel
    schedule_metadata = torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=device)

    _paged_mqa_logits_metadata_kernel[grid](
        effective_context_lens,
        effective_context_lens.stride(0),
        schedule_metadata,
        batch_size,
        split_kv,
        num_sms,
        BLOCK_SIZE=BLOCK_SIZE,
        GRID_DIM_X=grid_dim_x,
        PREFIX_BITS_C=PREFIX_BITS,
    )

    return schedule_metadata
