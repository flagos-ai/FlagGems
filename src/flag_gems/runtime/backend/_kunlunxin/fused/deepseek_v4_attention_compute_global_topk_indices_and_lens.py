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

from typing import Optional, Tuple

import torch


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    if is_valid_token is None:
        is_valid_token = torch.ones(
            (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
        )
    num_tokens, topk = topk_indices.shape
    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp_min(0).to(torch.long)
    block_indices = torch.div(safe_indices, block_size, rounding_mode="floor")
    request_tables = block_table[token_to_req_indices.to(torch.long)]
    block_numbers = torch.gather(request_tables, 1, block_indices)
    offsets = block_numbers * block_size + safe_indices.remainder(block_size)
    global_indices = torch.where(valid, offsets, -1).to(torch.int32)
    lens = valid.sum(dim=1, dtype=torch.int32)
    lens = torch.where(is_valid_token != 0, lens, torch.zeros_like(lens))
    return global_indices, lens
