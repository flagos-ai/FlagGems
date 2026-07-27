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

logger = logging.getLogger(__name__)


def bf16_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_metadata,
    max_context_len,
    clean_logits=False,
    logits_dtype=torch.float32,
):
    logger.debug("GEMS BF16_PAGED_MQA_LOGITS")

    B, next_n, H, D = q.shape
    total_tokens = B * next_n

    logits = torch.empty(
        total_tokens,
        max_context_len,
        dtype=logits_dtype,
        device=q.device,
    )

    if total_tokens == 0 or max_context_len == 0:
        return logits

    if clean_logits:
        logits.zero_()
    block_size = kv_cache.shape[1]
    for batch_id in range(B):
        for token_id in range(next_n):
            row = batch_id * next_n + token_id
            context_len = int(context_lens[batch_id, token_id].item())
            positions = torch.arange(context_len, device=q.device)
            logical_blocks = positions // block_size
            block_offsets = positions % block_size
            physical_blocks = block_table[batch_id, logical_blocks].long()
            keys = kv_cache[physical_blocks, block_offsets, 0]
            scores = torch.matmul(keys, q[batch_id, token_id].transpose(0, 1))
            scores = torch.relu(scores)
            logits[row, :context_len] = (
                scores.float() * weights[row].float().unsqueeze(0)
            ).sum(dim=1)
    return logits
