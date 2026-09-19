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
"""Kunlunxin moe_align_block_size -- host routing-metadata path.

The routing metadata is tiny (int32, roughly M*topk + E*block elements), so it
is computed on the host and copied back to the device. The computation uses
plain Python builtins only (no NumPy or Torch compute calls): this backend
family has repeatedly mis-compiled scalar read-modify-write counting, atomics
and wide cumsum tiles, which is exactly what the previous triton staging
kernels had to work around (see the git history of this file).

Semantics follow the previous contract:
  - sorted_token_ids: flat slot indices (m*topk + k) grouped by expert;
    unused slots are filled with numel (= M*topk)
  - expert_ids: the expert id of each aligned block; the unused tail is
    filled with -1
  - num_tokens_post_padded: total number of aligned slots

`moe_align_block_size_triton` is kept as an out-parameter-compatible shell for
callers of the previous triton implementation; the singleton / small_grouped
variants are host ports of the two generic fast paths.
"""

import logging
from collections import Counter
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _counts_and_offsets(flat, num_experts, block_size):
    """Per-expert route counts and BS-aligned region offsets (exclusive scan)."""
    counter = Counter(flat)
    counts = [counter[e] for e in range(num_experts)]
    aligned = [(c + block_size - 1) // block_size * block_size for c in counts]
    offsets = [0]
    for a in aligned:
        offsets.append(offsets[-1] + a)
    return counts, aligned, offsets


def _expert_ids_from_aligned(aligned, block_size, total):
    expert_ids = [-1] * total
    pos = 0
    for e, a in enumerate(aligned):
        n = a // block_size
        if n:
            expert_ids[pos : pos + n] = [e] * n
            pos += n
    return expert_ids


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    logger.debug("GEMS_KUNLUNXIN MOE_ALIGN_BLOCK_SIZE (host path)")
    device = topk_ids.device
    numel = topk_ids.numel()
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)

    flat = topk_ids.detach().reshape(-1).cpu().tolist()
    counts, aligned, offsets = _counts_and_offsets(flat, num_experts, block_size)
    ntp = offsets[-1]

    order = sorted(range(numel), key=flat.__getitem__)  # slots grouped by expert

    sorted_ids = [numel] * max_num_tokens_padded
    pos = 0
    for e in range(num_experts):
        c = counts[e]
        if c:
            s = offsets[e]
            sorted_ids[s : s + c] = order[pos : pos + c]
            pos += c

    expert_ids = _expert_ids_from_aligned(aligned, block_size, max_num_m_blocks)

    sorted_ids_t = torch.tensor(sorted_ids, dtype=torch.int32, device=device)
    expert_ids_t = torch.tensor(expert_ids, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.tensor([ntp], dtype=torch.int32, device=device)

    if expert_map is not None:
        expert_ids_t = expert_map[expert_ids_t]
    return sorted_ids_t, expert_ids_t, num_tokens_post_pad


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Out-parameter variant (same call convention as the previous triton
    implementation); writes the host-computed routing metadata into the
    caller-provided buffers. Needed by tests/test_moe_align_block_size.py."""
    sorted_ids, ids, ntp = moe_align_block_size(topk_ids, block_size, num_experts)
    sorted_token_ids.copy_(sorted_ids[: sorted_token_ids.numel()])
    expert_ids.copy_(ids[: expert_ids.numel()])
    num_tokens_post_pad.copy_(ntp)


def moe_align_block_size_singleton(
    topk_ids: torch.Tensor,
    block_size: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Host port of the singleton fast path (generic contract preserved):
    one aligned block per route; block head = route index, remainder =
    num_routes sentinel; expert_ids[r] = flattened topk_ids[r]; ntp = routes*BS.
    """
    num_routes = topk_ids.numel()
    flat = topk_ids.detach().reshape(-1).cpu().tolist()
    sorted_ids = [num_routes] * (num_routes * block_size)
    sorted_ids[::block_size] = list(range(num_routes))
    expert_ids = flat
    ntp = num_routes * block_size
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([ntp], dtype=torch.int32, device=dev),
    )


def moe_align_block_size_small_grouped(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Host port of the small_grouped fast path (generic contract preserved):
    per-expert BS-multiple regions packed from 0, routes in flat order within
    each expert, sentinel = num_routes, ntp = sum(aligned counts); expert_ids
    tail (beyond ntp/BS) filled with -1.
    """
    num_routes = topk_ids.numel()
    flat = topk_ids.detach().reshape(-1).cpu().tolist()
    counts, aligned, offsets = _counts_and_offsets(flat, num_experts, block_size)
    ntp = offsets[-1]

    sorted_ids = [num_routes] * (num_routes * block_size)
    order = sorted(range(num_routes), key=flat.__getitem__)
    pos = 0
    for e in range(num_experts):
        c = counts[e]
        if c:
            s = offsets[e]
            sorted_ids[s : s + c] = order[pos : pos + c]
            pos += c

    expert_ids = _expert_ids_from_aligned(aligned, block_size, num_routes)
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([ntp], dtype=torch.int32, device=dev),
    )
