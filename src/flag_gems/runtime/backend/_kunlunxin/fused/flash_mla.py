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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

device = device.name
logger = logging.getLogger(__name__)


@triton.jit
def _flash_mla_score_kernel(
    q,
    kv_cache,
    block_table,
    cache_seqlens,
    scores,
    sm_scale,
    head_num,
    stride_q_batch,
    stride_q_head,
    stride_kv_token,
    stride_block_table_batch,
    max_seqlen_pad,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head_block = ext.program_id(0)
    token_block = ext.program_id(1)
    batch = ext.program_id(2)

    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    tokens = token_block * BLOCK_N + tl.arange(0, BLOCK_N)
    head_mask = heads < head_num
    seq_len = tl.load(cache_seqlens + batch)
    token_mask = tokens < seq_len

    pages = tl.load(
        block_table + batch * stride_block_table_batch + tokens // PAGE_SIZE,
        mask=token_mask,
        other=0,
    )
    kv_tokens = pages * PAGE_SIZE + tokens % PAGE_SIZE

    score = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
    for dim in tl.static_range(0, HEAD_DIM):
        q_values = tl.load(
            q + batch * stride_q_batch + heads * stride_q_head + dim,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        k_values = tl.load(
            kv_cache + kv_tokens * stride_kv_token + dim,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        score += q_values[:, None] * k_values[None, :]
    score *= sm_scale
    score = tl.where(token_mask[None, :], score, float("-inf"))

    score_offsets = (
        (batch * head_num + heads[:, None]) * max_seqlen_pad + tokens[None, :]
    )
    tl.store(
        scores + score_offsets,
        score,
        mask=head_mask[:, None] & (tokens[None, :] < max_seqlen_pad),
    )


@triton.jit
def _flash_mla_partial_stats_kernel(
    scores,
    cache_seqlens,
    partial_max,
    partial_sum,
    head_num,
    num_chunks,
    max_seqlen_pad,
    BLOCK_N: tl.constexpr,
):
    chunk = ext.program_id(0)
    head = ext.program_id(1)
    batch = ext.program_id(2)
    start = chunk * BLOCK_N
    seq_len = tl.load(cache_seqlens + batch)

    if start < seq_len:
        tokens = start + tl.arange(0, BLOCK_N)
        mask = tokens < seq_len
        safe_tokens = tl.where(mask, tokens, start)
        offsets = (batch * head_num + head) * max_seqlen_pad + safe_tokens
        values = tl.load(scores + offsets)
        values = tl.where(mask, values, float("-inf"))
        value_max = tl.max(values, axis=0)
        exp_values = tl.where(mask, tl.exp(values - value_max), 0.0)
        value_sum = tl.sum(exp_values, axis=0)
    else:
        value_max = float("-inf")
        value_sum = 0.0

    stat_offset = (batch * head_num + head) * num_chunks + chunk
    tl.store(partial_max + stat_offset, value_max)
    tl.store(partial_sum + stat_offset, value_sum)


@triton.jit
def _flash_mla_finalize_stats_kernel(
    partial_max,
    partial_sum,
    row_max,
    row_sum,
    head_num,
    NUM_CHUNKS: tl.constexpr,
):
    head = ext.program_id(0)
    batch = ext.program_id(1)
    base = (batch * head_num + head) * NUM_CHUNKS

    current_max = tl.full((), float("-inf"), tl.float32)
    current_sum = tl.zeros((), tl.float32)
    for chunk in tl.static_range(0, NUM_CHUNKS):
        chunk_max = tl.load(partial_max + base + chunk)
        chunk_sum = tl.load(partial_sum + base + chunk)
        new_max = tl.maximum(current_max, chunk_max)
        current_sum = current_sum * tl.exp(current_max - new_max) + chunk_sum * tl.exp(
            chunk_max - new_max
        )
        current_max = new_max

    row = batch * head_num + head
    tl.store(row_max + row, current_max)
    tl.store(row_sum + row, current_sum)


@triton.jit
def _flash_mla_value_partial_kernel(
    kv_cache,
    block_table,
    cache_seqlens,
    scores,
    row_max,
    row_sum,
    partial_output,
    head_num,
    stride_kv_token,
    stride_block_table_batch,
    num_value_chunks,
    max_seqlen_pad,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    VALUE_BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
):
    dim_block = tl.program_id(0)
    head_block = tl.program_id(1)
    combined = tl.program_id(2)
    value_chunk = combined % num_value_chunks
    batch = combined // num_value_chunks

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_mask = dims < HEAD_DIM_V
    head_mask = heads < head_num
    rows = batch * head_num + heads
    max_value = tl.load(row_max + rows, mask=head_mask, other=0.0)
    sum_value = tl.load(row_sum + rows, mask=head_mask, other=1.0)
    seq_len = tl.load(cache_seqlens + batch)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    start = value_chunk * VALUE_BLOCK_N

    for token_offset in tl.static_range(0, VALUE_BLOCK_N):
        token = start + token_offset
        if token < seq_len:
            score = tl.load(
                scores + rows * max_seqlen_pad + token,
                mask=head_mask,
                other=float("-inf"),
            )
            probability = tl.exp(score - max_value) / sum_value
            page = tl.load(
                block_table
                + batch * stride_block_table_batch
                + token // PAGE_SIZE
            )
            kv_token = page * PAGE_SIZE + token % PAGE_SIZE
            values = tl.load(
                kv_cache + kv_token * stride_kv_token + dims,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            acc += probability[:, None] * values[None, :]

    partial_offsets = (
        ((batch * head_num + heads[:, None]) * num_value_chunks + value_chunk)
        * HEAD_DIM_V
        + dims[None, :]
    )
    tl.store(
        partial_output + partial_offsets,
        acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _flash_mla_value_finalize_kernel(
    partial_output,
    output,
    head_num,
    stride_output_batch,
    stride_output_head,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_VALUE_CHUNKS: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
):
    dim_block = tl.program_id(0)
    head_block = tl.program_id(1)
    batch = tl.program_id(2)
    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_mask = dims < HEAD_DIM_V
    head_mask = heads < head_num
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    for value_chunk in tl.static_range(0, NUM_VALUE_CHUNKS):
        partial_offsets = (
            ((batch * head_num + heads[:, None]) * NUM_VALUE_CHUNKS + value_chunk)
            * HEAD_DIM_V
            + dims[None, :]
        )
        acc += tl.load(
            partial_output + partial_offsets,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

    output_offsets = (
        batch * stride_output_batch
        + heads[:, None] * stride_output_head
        + dims[None, :]
    )
    tl.store(
        output + output_offsets,
        acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
):
    logger.debug("GEMS_KUNLUNXIN FLASH_MLA")
    assert s_q == 1
    assert h_kv == 1
    assert d == 576
    assert dv == 512

    q = q.contiguous()
    block_table = block_table.contiguous()
    blocked_k = blocked_k.contiguous()
    cache_seqlens = cache_seqlens.contiguous()

    head_num = h_q
    sm_scale = 1 / math.sqrt(d)
    block_h = 16
    block_n = 64
    block_d = 32
    value_block_n = 256
    num_chunks = triton.cdiv(max_seqlen_pad, block_n)
    num_value_chunks = triton.cdiv(max_seqlen_pad, value_block_n)

    scores = torch.empty(
        (b, head_num, max_seqlen_pad),
        dtype=torch.float32,
        device=q.device,
    )
    partial_max = torch.empty(
        (b, head_num, num_chunks),
        dtype=torch.float32,
        device=q.device,
    )
    partial_sum = torch.empty_like(partial_max)
    row_max = torch.empty((b, head_num), dtype=torch.float32, device=q.device)
    row_sum = torch.empty_like(row_max)
    partial_output = torch.empty(
        (b, head_num, num_value_chunks, dv),
        dtype=torch.float32,
        device=q.device,
    )
    output = torch.empty((b * s_q, head_num, dv), dtype=q.dtype, device=q.device)

    with torch_device_fn.device(q.device):
        _flash_mla_score_kernel[
            (
                triton.cdiv(head_num, block_h),
                triton.cdiv(max_seqlen_pad, block_n),
                b,
            )
        ](
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            scores,
            sm_scale,
            head_num,
            q.stride(0),
            q.stride(2),
            blocked_k.stride(-2),
            block_table.stride(0),
            max_seqlen_pad,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            PAGE_SIZE=block_size,
            HEAD_DIM=d,
            isCloseVectorization=True,
            buffer_size_limit=2048,
            num_warps=8,
            num_stages=1,
        )
        _flash_mla_partial_stats_kernel[(num_chunks, head_num, b)](
            scores,
            cache_seqlens,
            partial_max,
            partial_sum,
            head_num,
            num_chunks,
            max_seqlen_pad,
            BLOCK_N=block_n,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        _flash_mla_finalize_stats_kernel[(head_num, b)](
            partial_max,
            partial_sum,
            row_max,
            row_sum,
            head_num,
            NUM_CHUNKS=num_chunks,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        _flash_mla_value_partial_kernel[
            (
                triton.cdiv(dv, block_d),
                triton.cdiv(head_num, block_h),
                num_value_chunks * b,
            )
        ](
            blocked_k,
            block_table,
            cache_seqlens,
            scores,
            row_max,
            row_sum,
            partial_output,
            head_num,
            blocked_k.stride(-2),
            block_table.stride(0),
            num_value_chunks,
            max_seqlen_pad,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            VALUE_BLOCK_N=value_block_n,
            PAGE_SIZE=block_size,
            HEAD_DIM_V=dv,
            isCloseVectorization=True,
            buffer_size_limit=2048,
            num_warps=4,
            num_stages=1,
        )
        _flash_mla_value_finalize_kernel[
            (
                triton.cdiv(dv, block_d),
                triton.cdiv(head_num, block_h),
                b,
            )
        ](
            partial_output,
            output,
            head_num,
            output.stride(0),
            output.stride(1),
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            NUM_VALUE_CHUNKS=num_value_chunks,
            HEAD_DIM_V=dv,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )

    return output.view((b, s_q, h_q, dv))
