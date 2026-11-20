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

# Persistent grid: GCU CTA dispatch is expensive, so the program count is
# decoupled from the problem size and every program walks several tiles.
_NUM_PROGRAMS = 48
_TILE_ELEMS = 8192
_STATIC_TOPK_MAX = 8
_COPY_BYTES = 32768

# Expert-path tiling. A tile spans only the contiguous (topk, hidden) slab of
# one token, so larger tiles mean fewer and longer DMA transactions: 32 KiB per
# tile only pays off once the tensor is big enough to hide the launch tail.
# 64 KiB tiles measure fastest for >=256 MiB inputs (1615 GB/s at K=512).
_EXPERT_TILE_SMALL = 16384
_EXPERT_TILE_SMALL_BYTES = 32 << 10
_EXPERT_TILE_TARGET_BYTES = 128 << 10
_EXPERT_TILE_LARGE_BYTES = 256 << 20
_SPLIT_SERIAL_TILES = 128
_SPLIT_TARGET_WORK = 384
_PARTIAL_BYTES_MAX = 32 << 20


@triton.jit
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    total_work = tl.cdiv(num_tokens, BLOCK_M) * NUM_K
    offs_k = tl.arange(0, BLOCK_K)
    for work in range(pid, total_work, GRID):
        pid_m = work // NUM_K
        pid_k = work % NUM_K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_idx = pid_k * BLOCK_K + offs_k
        mask = (offs_m[:, None] < num_tokens) & (k_idx[None, :] < hidden_size)
        base = (
            input_ptr
            + offs_m[:, None] * input_stride_token
            + k_idx[None, :] * input_stride_hidden
        )
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        if TOPK > 0:
            for expert in tl.static_range(TOPK):
                acc += tl.load(base + expert * input_stride_topk, mask=mask, other=0.0)
        else:
            for expert in range(topk):
                acc += tl.load(base + expert * input_stride_topk, mask=mask, other=0.0)
        out_pos = (
            output_ptr
            + offs_m[:, None] * output_stride_token
            + k_idx[None, :] * output_stride_hidden
        )
        tl.store(out_pos, acc.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def moe_sum_copy_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    total = tl.cdiv(numel, BLOCK)
    for tile in range(tl.program_id(0), total, GRID):
        idx = tile * BLOCK + offs
        mask = idx < numel
        tl.store(output_ptr + idx, tl.load(input_ptr + idx, mask=mask), mask=mask)


@triton.jit
def moe_sum_expert_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    GRID: tl.constexpr,
):
    """Reduce the topk axis with contiguous tiles.

    For a fixed token the ``(topk, hidden)`` slab is contiguous, so a
    ``(BLOCK_E, BLOCK_K)`` tile is one long DMA instead of one short segment per
    row. The accumulator stays 2D and is reduced once at the end, which avoids a
    cross-lane reduction per tile.
    """
    pid = tl.program_id(0)
    total_work = num_tokens * NUM_K
    offs_k = tl.arange(0, BLOCK_K)
    offs_e = tl.arange(0, BLOCK_E)
    for work in range(pid, total_work, GRID):
        token_idx = work // NUM_K
        pid_k = work % NUM_K
        k_idx = pid_k * BLOCK_K + offs_k
        k_mask = k_idx < hidden_size
        base = input_ptr + token_idx * input_stride_token + k_idx * input_stride_hidden
        acc = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
        for expert_start in range(0, topk, BLOCK_E):
            e_idx = expert_start + offs_e
            mask = (e_idx[:, None] < topk) & k_mask[None, :]
            acc += tl.load(
                base + e_idx[:, None] * input_stride_topk, mask=mask, other=0.0
            )
        out_pos = (
            output_ptr + token_idx * output_stride_token + k_idx * output_stride_hidden
        )
        tl.store(
            out_pos, tl.sum(acc, axis=0).to(output_ptr.dtype.element_ty), mask=k_mask
        )


@triton.jit
def moe_sum_expert_split_kernel(
    input_ptr,
    partial_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    total_work = num_tokens * NUM_K * SPLIT
    offs_k = tl.arange(0, BLOCK_K)
    offs_e = tl.arange(0, BLOCK_E)
    for work in range(pid, total_work, GRID):
        split_id = work % SPLIT
        rest = work // SPLIT
        pid_k = rest % NUM_K
        token_idx = rest // NUM_K
        k_idx = pid_k * BLOCK_K + offs_k
        k_mask = k_idx < hidden_size
        expert_start = split_id * CHUNK
        expert_end = tl.minimum(expert_start + CHUNK, topk)
        base = (
            input_ptr
            + token_idx * input_stride_token
            + k_idx * input_stride_hidden
            + expert_start * input_stride_topk
        )
        acc = tl.zeros((BLOCK_E, BLOCK_K), dtype=tl.float32)
        for offset in range(0, expert_end - expert_start, BLOCK_E):
            e_idx = offset + offs_e
            mask = (e_idx[:, None] < expert_end - expert_start) & k_mask[None, :]
            acc += tl.load(
                base + e_idx[:, None] * input_stride_topk, mask=mask, other=0.0
            )
        slot = ((split_id * num_tokens + token_idx) * NUM_K + pid_k) * BLOCK_K
        tl.store(partial_ptr + slot + offs_k, tl.sum(acc, axis=0))


@triton.jit
def moe_sum_expert_reduce_kernel(
    partial_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    output_stride_token,
    output_stride_hidden,
    SPLIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // NUM_K
    pid_k = pid % NUM_K
    offs_k = tl.arange(0, BLOCK_K)
    k_idx = pid_k * BLOCK_K + offs_k
    k_mask = k_idx < hidden_size
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for split_id in tl.static_range(SPLIT):
        slot = ((split_id * num_tokens + token_idx) * NUM_K + pid_k) * BLOCK_K
        acc += tl.load(partial_ptr + slot + offs_k)
    out_pos = (
        output_ptr + token_idx * output_stride_token + k_idx * output_stride_hidden
    )
    tl.store(out_pos, acc.to(output_ptr.dtype.element_ty), mask=k_mask)


def _token_tile_shape(num_tokens, hidden_size):
    block_k = min(triton.next_power_of_2(hidden_size), _TILE_ELEMS)
    block_m = max(1, _TILE_ELEMS // block_k)
    block_m = min(block_m, triton.next_power_of_2(num_tokens), 32)
    return block_m, block_k


def _expert_tile_shape(num_tokens, topk, hidden_size, element_size):
    total_bytes = num_tokens * topk * hidden_size * element_size
    if total_bytes >= _EXPERT_TILE_LARGE_BYTES:
        # Budget the tile in bytes: 128 KiB is the sweet spot for fp16/bf16,
        # while a 256 KiB fp32 tile loses bandwidth.
        tile_elems = max(_EXPERT_TILE_SMALL, _EXPERT_TILE_TARGET_BYTES // element_size)
    else:
        tile_elems = _EXPERT_TILE_SMALL
    block_k = min(triton.next_power_of_2(hidden_size), 2048)
    block_e = max(1, tile_elems // block_k)
    block_e = min(block_e, triton.next_power_of_2(topk))
    return block_e, block_k


def _launch(
    input,
    output,
    block_m=None,
    block_k=None,
    block_e=None,
    num_programs=None,
    split=None,
    num_warps=None,
):
    num_tokens, topk, hidden_size = input.shape
    programs = num_programs or _NUM_PROGRAMS
    warps = num_warps or 4
    in_strides = input.stride()
    out_strides = output.stride()

    if topk == 1 and input.is_contiguous() and output.is_contiguous():
        numel = input.numel()
        block = min(
            triton.next_power_of_2(numel),
            max(1024, _COPY_BYTES // input.element_size()),
        )
        total = triton.cdiv(numel, block)
        nprog = min(programs, total)
        moe_sum_copy_kernel[(nprog,)](
            input, output, numel, BLOCK=block, GRID=nprog, num_warps=warps
        )
        return

    if topk > _STATIC_TOPK_MAX:
        if block_e is None or block_k is None:
            block_e, block_k = _expert_tile_shape(
                num_tokens, topk, hidden_size, input.element_size()
            )
        num_k = triton.cdiv(hidden_size, block_k)
        base_work = num_tokens * num_k
        if split is None:
            serial_tiles = triton.cdiv(topk, block_e)
            if serial_tiles >= _SPLIT_SERIAL_TILES:
                split = max(1, triton.cdiv(_SPLIT_TARGET_WORK, base_work))
            else:
                split = 1
            max_split = max(1, _PARTIAL_BYTES_MAX // (base_work * block_k * 4))
            split = min(split, max_split, topk)
        if split <= 1:
            nprog = min(programs, base_work)
            moe_sum_expert_kernel[(nprog,)](
                input,
                output,
                num_tokens,
                topk,
                hidden_size,
                in_strides[0],
                in_strides[1],
                in_strides[2],
                out_strides[0],
                out_strides[1],
                BLOCK_E=block_e,
                BLOCK_K=block_k,
                NUM_K=num_k,
                GRID=nprog,
                num_warps=warps,
            )
            return

        chunk = triton.cdiv(topk, split)
        partial = torch.empty(
            (split * base_work * block_k,), dtype=torch.float32, device=input.device
        )
        num_work = base_work * split
        nprog = min(programs, num_work)
        moe_sum_expert_split_kernel[(nprog,)](
            input,
            partial,
            num_tokens,
            topk,
            hidden_size,
            in_strides[0],
            in_strides[1],
            in_strides[2],
            CHUNK=chunk,
            SPLIT=split,
            BLOCK_E=block_e,
            BLOCK_K=block_k,
            NUM_K=num_k,
            GRID=nprog,
            num_warps=warps,
        )
        moe_sum_expert_reduce_kernel[(base_work,)](
            partial,
            output,
            num_tokens,
            hidden_size,
            out_strides[0],
            out_strides[1],
            SPLIT=split,
            BLOCK_K=block_k,
            NUM_K=num_k,
            num_warps=warps,
        )
        return

    if block_m is None or block_k is None:
        block_m, block_k = _token_tile_shape(num_tokens, hidden_size)
    num_k = triton.cdiv(hidden_size, block_k)
    num_work = triton.cdiv(num_tokens, block_m) * num_k
    nprog = min(programs, num_work)
    moe_sum_kernel[(nprog,)](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        in_strides[0],
        in_strides[1],
        in_strides[2],
        out_strides[0],
        out_strides[1],
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NUM_K=num_k,
        GRID=nprog,
        num_warps=warps,
    )


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    logger.debug("GEMS_ENFLAME GCU400 MOE_SUM")
    _launch(input, output)
