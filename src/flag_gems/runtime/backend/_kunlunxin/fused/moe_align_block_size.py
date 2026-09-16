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
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


# TritonXPU(Kunlunxin) notes -- why this file does not mirror the generic
# staged implementation (all empirically established on XPU, 2026-08-31):
#
# * The generic / previous vendor stage1 counted with a scalar read-modify-write
#   on the same address (`c = load(cnt+idx); store(cnt+idx, c+1)`).  On this
#   backend that pattern loses and duplicates updates: for 512 experts and
#   10320 tokens the resulting histogram summed to 301996 instead of 10320
#   (29x over-count).  The garbage then overflows the int32 aligned cumsum
#   (num_tokens_post_pad read -1832370752) and stage4 stores far outside
#   `sorted_token_ids` -> `illegal memory access` (error 700).
# * `tl.atomic_add` is not a usable replacement either: its return value cannot
#   be lowered (`uni_sram` / `tt.atomic_rmw op ... still has uses`) and plain
#   accumulation silently drops updates whenever several lanes target the same
#   address, which is exactly the histogram case.
# * A masked discrete store is not usable: the mask is not honoured, so
#   inactive lanes still write, and they all collapse onto the same address.
#
# The design below therefore uses only primitives verified on this backend:
#   - one program per expert, so every store target is inside that expert's own
#     contiguous output range (any permutation inside a range is legal here,
#     see the invariants checked by tests/test_moe_align_block_size.py);
#   - counting by vector compare + `tl.sum` instead of scalar RMW / atomics;
#   - compaction with `tl.cumsum` on a 256 lane tile (>=512 lanes is broken);
#   - zero masks and zero `other=` in the scatter: inactive lanes are
#     redirected to a per (expert, lane) unique scratch sink, so the discrete
#     store never has duplicate targets;
#   - the flat token buffer is padded with an out-of-range expert id so the
#     token loads never need a mask.


_MOE_TILE = 256


@triton.jit(do_not_specialize=["n", "value"])
def moe_align_fill_i32(out_ptr, value, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * TILE + tl.arange(0, TILE)
    tl.store(out_ptr + offsets, value, mask=offsets < n)


@triton.jit(do_not_specialize=["numel_padded"])
def moe_align_count_per_expert(
    topk_ids_ptr,
    counts_ptr,
    numel_padded,
    TILE: tl.constexpr,
):
    expert_id = tl.program_id(0)
    lanes = tl.arange(0, TILE)
    total = 0
    for start in range(0, numel_padded, TILE):
        tokens = tl.load(topk_ids_ptr + start + lanes)
        total = total + tl.sum((tokens == expert_id).to(tl.int32), axis=0)
    tl.store(counts_ptr + expert_id, total)


@triton.jit
def moe_align_expert_offsets(
    counts_ptr,
    starts_ptr,
    total_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    running = 0
    for i in range(0, num_experts):
        count = tl.load(counts_ptr + i)
        tl.store(starts_ptr + i, running)
        running = running + tl.cdiv(count, block_size) * block_size
    tl.store(total_tokens_post_pad_ptr, running)


@triton.jit
def moe_align_write_expert_ids(
    counts_ptr,
    starts_ptr,
    expert_ids_ptr,
    block_size: tl.constexpr,
):
    expert_id = tl.program_id(0)
    start = tl.load(starts_ptr + expert_id)
    count = tl.load(counts_ptr + expert_id)
    num_blocks = tl.cdiv(count, block_size)
    first_block = start // block_size
    for b in range(0, num_blocks):
        tl.store(expert_ids_ptr + first_block + b, expert_id)


@triton.jit(do_not_specialize=["numel_padded"])
def moe_align_scatter_tokens(
    topk_ids_ptr,
    out_ptr,
    starts_ptr,
    numel_padded,
    SINK_OFFSET: tl.constexpr,
    TILE: tl.constexpr,
):
    expert_id = tl.program_id(0)
    base = tl.load(starts_ptr + expert_id)
    lanes = tl.arange(0, TILE)
    sink = SINK_OFFSET + expert_id * TILE + lanes
    carry = 0
    for start in range(0, numel_padded, TILE):
        tokens = tl.load(topk_ids_ptr + start + lanes)
        hit = tokens == expert_id
        hit_i32 = hit.to(tl.int32)
        rank = tl.cumsum(hit_i32, axis=0) - hit_i32
        target = tl.where(hit, base + carry + rank, sink)
        tl.store(out_ptr + target, start + lanes)
        carry = carry + tl.sum(hit_i32, axis=0)


@triton.jit(do_not_specialize=["n"])
def moe_align_copy_prefix_i32(src_ptr, dst_ptr, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * TILE + tl.arange(0, TILE)
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, values, mask=offsets < n)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    logger.debug("GEMS_KUNLUNXIN MOE_ALIGN_BLOCK_SIZE_TRITON")
    device = topk_ids.device
    numel = topk_ids.numel()
    numel_sorted = sorted_token_ids.numel()
    numel_expert_ids = expert_ids.numel()
    tile = _MOE_TILE

    # Fill the sentinels first: the golden implementation leaves `numel` in
    # every unused sorted-token slot and -1 in every unused expert-id block,
    # and the caller only hands us uninitialised buffers.
    grid_expert_fill = (ceil_div(numel_expert_ids, tile),)
    moe_align_fill_i32[grid_expert_fill](expert_ids, -1, numel_expert_ids, TILE=tile)

    if numel == 0:
        grid_sorted_fill = (ceil_div(numel_sorted, tile),)
        moe_align_fill_i32[grid_sorted_fill](
            sorted_token_ids, numel, numel_sorted, TILE=tile
        )
        moe_align_fill_i32[(1,)](num_tokens_post_pad, 0, 1, TILE=tile)
        return

    # Pad the flat token stream with an out-of-range expert id so that every
    # token load inside the kernels is a full unmasked tile.
    numel_padded = round_up(numel, tile)
    topk_flat = topk_ids.reshape(-1)
    if numel_padded == numel and topk_flat.is_contiguous():
        topk_padded = topk_flat
    else:
        topk_padded = torch.full(
            (numel_padded,), num_experts, dtype=torch.int32, device=device
        )
        topk_padded[:numel] = topk_flat

    counts = torch.empty((num_experts,), dtype=torch.int32, device=device)
    starts = torch.empty((num_experts,), dtype=torch.int32, device=device)
    grid = (num_experts,)

    moe_align_count_per_expert[grid](topk_padded, counts, numel_padded, TILE=tile)
    moe_align_expert_offsets[(1,)](
        counts, starts, num_tokens_post_pad, num_experts, block_size
    )
    moe_align_write_expert_ids[grid](counts, starts, expert_ids, block_size)

    # `moe_align_scatter_tokens` needs a per (expert, lane) unique sink for the
    # lanes that do not belong to the current expert, so it writes into an
    # over-allocated scratch buffer whose tail is the sink area.
    sink_offset = numel_sorted
    scratch = torch.empty(
        (sink_offset + num_experts * tile,), dtype=torch.int32, device=device
    )
    grid_scratch_fill = (ceil_div(scratch.numel(), tile),)
    moe_align_fill_i32[grid_scratch_fill](scratch, numel, scratch.numel(), TILE=tile)
    moe_align_scatter_tokens[grid](
        topk_padded,
        scratch,
        starts,
        numel_padded,
        SINK_OFFSET=sink_offset,
        TILE=tile,
    )
    moe_align_copy_prefix_i32[(ceil_div(numel_sorted, tile),)](
        scratch, sorted_token_ids, numel_sorted, TILE=tile
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    logger.debug("GEMS_KUNLUNXIN MOE_ALIGN_BLOCK_SIZE")
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad
