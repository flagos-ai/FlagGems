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


@triton.jit
def _paged_mqa_logits_metadata_kernel(
    context_lens_ptr,
    context_lens_stride,
    schedule_metadata_ptr,
    batch_size,
    split_kv,
    num_sms,
    BLOCK_SIZE: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    context_lens = tl.load(
        context_lens_ptr + offsets * context_lens_stride, mask=mask, other=0
    )
    num_segs = tl.where(mask, (context_lens + split_kv - 1) // split_kv, 0)
    prefix_sum = tl.cumsum(num_segs, axis=0)
    total_segs = tl.max(prefix_sum)

    segments_per_sm = total_segs // num_sms
    remainder = total_segs % num_sms
    segment_start = sm_idx * segments_per_sm + tl.minimum(sm_idx, remainder)

    q_idx = tl.sum(tl.where((prefix_sum <= segment_start) & mask, 1, 0))
    previous_prefix = tl.max(tl.where(offsets < q_idx, prefix_sum, 0))
    kv_split_idx = segment_start - previous_prefix

    output_offset = sm_idx * 2
    tl.store(schedule_metadata_ptr + output_offset, q_idx)
    tl.store(schedule_metadata_ptr + output_offset + 1, kv_split_idx)


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    logger.debug("KUNLUNXIN GET_PAGED_MQA_LOGITS_METADATA")
    del block_size

    if context_lens.dim() == 2:
        batch_size, next_n = context_lens.shape
        effective_context_lens = context_lens[:, next_n - 1]
    else:
        batch_size = context_lens.shape[0]
        effective_context_lens = context_lens

    schedule_metadata = torch.empty(
        (num_sms + 1, 2), dtype=torch.int32, device=context_lens.device
    )
    block_size = triton.next_power_of_2(max(16, batch_size))
    _paged_mqa_logits_metadata_kernel[(num_sms + 1,)](
        effective_context_lens,
        effective_context_lens.stride(0),
        schedule_metadata,
        batch_size,
        256,
        num_sms,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return schedule_metadata
