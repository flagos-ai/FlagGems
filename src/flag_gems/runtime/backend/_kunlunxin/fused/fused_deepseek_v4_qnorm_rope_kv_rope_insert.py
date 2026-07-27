import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _q_partial_sum_kernel(
    q,
    partial_sums,
    stride_q_token,
    stride_q_head,
    num_heads: tl.constexpr,
    PARTIALS: tl.constexpr,
    PARTIAL_INDEX: tl.constexpr,
    Q_ITEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_item = tl.program_id(0)
    offsets = PARTIAL_INDEX * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(q + q_item * 512 + offsets).to(tl.float32)
    tl.store(
        partial_sums + PARTIAL_INDEX * Q_ITEMS + q_item,
        tl.sum(values * values, axis=0),
    )


@libentry()
@triton.jit
def _q_rstd_kernel(
    partial_sums,
    rstd,
    eps,
    PARTIALS: tl.constexpr,
    Q_ITEMS: tl.constexpr,
    num_heads: tl.constexpr,
):
    q_item = tl.program_id(0)
    square_sum = tl.zeros((), dtype=tl.float32)
    for partial in tl.static_range(0, PARTIALS):
        square_sum += tl.load(partial_sums + partial * Q_ITEMS + q_item)
    tl.store(rstd + q_item, tl.rsqrt(square_sum / 512.0 + eps))


@libentry()
@triton.jit
def _q_norm_rope_kernel(
    q,
    rstd,
    position_ids,
    cos_sin_cache,
    stride_q_token,
    stride_q_head,
    stride_cos_position,
    num_heads: tl.constexpr,
):
    q_item = tl.program_id(0)
    token = q_item // num_heads
    base = q_item * 512
    scale = tl.load(rstd + q_item)

    offsets = tl.arange(0, 512)
    values = tl.load(q + base + offsets).to(tl.float32)
    normalized = (values * scale).to(tl.bfloat16)

    pair = tl.arange(0, 32)
    even_offsets = 448 + pair * 2
    odd_offsets = even_offsets + 1
    even = (
        tl.load(q + base + even_offsets).to(tl.float32) * scale
    ).to(tl.bfloat16).to(tl.float32)
    odd = (
        tl.load(q + base + odd_offsets).to(tl.float32) * scale
    ).to(tl.bfloat16).to(tl.float32)
    position = tl.load(position_ids + token)
    cos = tl.load(cos_sin_cache + position * stride_cos_position + pair).to(
        tl.float32
    )
    sin = tl.load(cos_sin_cache + position * stride_cos_position + 32 + pair).to(
        tl.float32
    )

    tl.store(q + base + offsets, normalized)
    tl.store(q + base + even_offsets, even * cos - odd * sin)
    tl.store(q + base + odd_offsets, even * sin + odd * cos)


@libentry()
@triton.jit
def _q_norm_chunk_kernel(
    q,
    rstd,
    num_heads: tl.constexpr,
    CHUNK_INDEX: tl.constexpr,
):
    q_item = tl.program_id(0)
    offsets = CHUNK_INDEX * 32 + tl.arange(0, 32)
    values = tl.load(q + q_item * 512 + offsets).to(tl.float32)
    normalized = (values * tl.load(rstd + q_item)).to(tl.bfloat16)
    tl.store(q + q_item * 512 + offsets, normalized)


@libentry()
@triton.jit
def _q_norm_all_kernel(
    q,
    normalized_q,
    rstd,
    Q_ITEM: tl.constexpr,
):
    scale = tl.load(rstd + Q_ITEM)
    for chunk in tl.static_range(0, 32):
        offsets = chunk * 16 + tl.arange(0, 16)
        values = tl.load(q + Q_ITEM * 512 + offsets).to(tl.float32)
        tl.store(
            normalized_q + Q_ITEM * 512 + offsets,
            (values * scale).to(tl.bfloat16),
        )


@libentry()
@triton.jit
def _q_copy_kernel(src, dst, Q_ITEM: tl.constexpr):
    chunk = tl.program_id(0)
    offsets = chunk * 16 + tl.arange(0, 16)
    values = tl.load(src + Q_ITEM * 512 + offsets)
    tl.store(dst + Q_ITEM * 512 + offsets, values)


@libentry()
@triton.jit
def _q_norm_scalar_out_kernel(q, normalized_q, rstd):
    dimension = tl.program_id(0)
    q_item = tl.program_id(1)
    offset = q_item * 512 + dimension
    value = tl.load(q + offset).to(tl.float32)
    scale = tl.load(rstd + q_item)
    tl.store(normalized_q + offset, (value * scale).to(tl.bfloat16))


@libentry()
@triton.jit
def _q_copy_scalar_all_kernel(src, dst):
    dimension = tl.program_id(0)
    q_item = tl.program_id(1)
    offset = q_item * 512 + dimension
    tl.store(dst + offset, tl.load(src + offset))


@libentry()
@triton.jit
def _q_norm_scalar_kernel(
    q,
    rstd,
    Q_ITEM: tl.constexpr,
):
    dimension = tl.program_id(0)
    offset = Q_ITEM * 512 + dimension
    value = tl.load(q + offset).to(tl.float32)
    scale = tl.load(rstd + Q_ITEM)
    tl.store(q + offset, (value * scale).to(tl.bfloat16))


@libentry()
@triton.jit
def _q_rope_kernel(
    q,
    rope_q,
    position_ids,
    cos_sin_cache,
    stride_cos_position,
    num_heads: tl.constexpr,
):
    pair = tl.program_id(0)
    head = tl.program_id(1)
    token = tl.program_id(2)
    q_item = token * num_heads + head
    base = q_item * 512
    even_offset = 448 + pair * 2
    odd_offset = even_offset + 1
    even = tl.load(q + base + even_offset).to(tl.float32)
    odd = tl.load(q + base + odd_offset).to(tl.float32)
    position = tl.load(position_ids + token)
    cos = tl.load(cos_sin_cache + position * stride_cos_position + pair).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + position * stride_cos_position + 32 + pair
    ).to(tl.float32)
    tl.store(
        rope_q + base + even_offset,
        (even * cos - odd * sin).to(tl.bfloat16),
    )
    tl.store(
        rope_q + base + odd_offset,
        (even * sin + odd * cos).to(tl.bfloat16),
    )


@libentry()
@triton.jit
def _kv_copy_scalar_kernel(
    kv,
    k_cache,
    slot_mapping,
    stride_kv_token,
    stride_cache_block,
    stride_cache_token,
    cache_block_size: tl.constexpr,
    TOKEN: tl.constexpr,
):
    dimension = tl.program_id(0)
    slot = tl.load(slot_mapping + TOKEN)
    if slot >= 0:
        cache_base = (
            (slot // cache_block_size) * stride_cache_block
            + (slot % cache_block_size) * stride_cache_token
        )
        value = tl.load(kv + TOKEN * stride_kv_token + dimension)
        tl.store(k_cache + cache_base + dimension, value)


@libentry()
@triton.jit
def _kv_copy_chunk_kernel(
    kv,
    k_cache,
    slot_mapping,
    stride_kv_token,
    stride_cache_block,
    stride_cache_token,
    cache_block_size: tl.constexpr,
    TOKEN: tl.constexpr,
    CHUNK_INDEX: tl.constexpr,
):
    slot = tl.load(slot_mapping + TOKEN)
    if slot >= 0:
        cache_base = (
            (slot // cache_block_size) * stride_cache_block
            + (slot % cache_block_size) * stride_cache_token
        )
        offsets = CHUNK_INDEX * 16 + tl.arange(0, 16)
        values = tl.load(kv + TOKEN * stride_kv_token + offsets)
        tl.store(k_cache + cache_base + offsets, values)


@libentry()
@triton.jit
def _kv_rope_scalar_kernel(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    stride_kv_token,
    stride_cache_block,
    stride_cache_token,
    stride_cos_position,
    cache_block_size: tl.constexpr,
    TOKEN: tl.constexpr,
):
    pair = tl.program_id(0)
    slot = tl.load(slot_mapping + TOKEN)
    if slot >= 0:
        cache_base = (
            (slot // cache_block_size) * stride_cache_block
            + (slot % cache_block_size) * stride_cache_token
        )
        even_offset = 448 + pair * 2
        odd_offset = even_offset + 1
        even = tl.load(kv + TOKEN * stride_kv_token + even_offset).to(tl.float32)
        odd = tl.load(kv + TOKEN * stride_kv_token + odd_offset).to(tl.float32)
        position = tl.load(position_ids + TOKEN)
        cos = tl.load(
            cos_sin_cache + position * stride_cos_position + pair
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache + position * stride_cos_position + 32 + pair
        ).to(tl.float32)
        tl.store(
            k_cache + cache_base + even_offset,
            (even * cos - odd * sin).to(tl.bfloat16),
        )
        tl.store(
            k_cache + cache_base + odd_offset,
            (even * sin + odd * cos).to(tl.bfloat16),
        )


@libentry()
@triton.jit
def _kv_rope_insert_kernel(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    stride_kv_token,
    stride_cache_block,
    stride_cache_token,
    stride_cos_position,
    cache_block_size: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slot_mapping + token)
    if slot >= 0:
        offsets = tl.arange(0, 512)
        values = tl.load(kv + token * stride_kv_token + offsets)
        cache_base = (
            (slot // cache_block_size) * stride_cache_block
            + (slot % cache_block_size) * stride_cache_token
        )
        tl.store(k_cache + cache_base + offsets, values)

        pair = tl.arange(0, 32)
        even_offsets = 448 + pair * 2
        odd_offsets = even_offsets + 1
        even = tl.load(kv + token * stride_kv_token + even_offsets).to(tl.float32)
        odd = tl.load(kv + token * stride_kv_token + odd_offsets).to(tl.float32)
        position = tl.load(position_ids + token)
        cos = tl.load(cos_sin_cache + position * stride_cos_position + pair).to(
            tl.float32
        )
        sin = tl.load(
            cos_sin_cache + position * stride_cos_position + 32 + pair
        ).to(tl.float32)
        tl.store(k_cache + cache_base + even_offsets, even * cos - odd * sin)
        tl.store(k_cache + cache_base + odd_offsets, even * sin + odd * cos)


def fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    logger.debug("GEMS_KUNLUNXIN FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_INSERT")
    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == 512
    q_items = num_tokens * num_heads
    partials = 16
    partial_sums = torch.empty(
        (partials, q_items), device=q.device, dtype=torch.float32
    )
    rstd = torch.empty((q_items,), device=q.device, dtype=torch.float32)

    with torch_device_fn.device(q.device):
        for partial_index in range(partials):
            _q_partial_sum_kernel[(q_items,)](
                q,
                partial_sums,
                q.stride(0),
                q.stride(1),
                num_heads,
                PARTIALS=partials,
                PARTIAL_INDEX=partial_index,
                Q_ITEMS=q_items,
                BLOCK=32,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        _q_rstd_kernel[(q_items,)](
            partial_sums,
            rstd,
            eps,
            PARTIALS=partials,
            Q_ITEMS=q_items,
            num_heads=num_heads,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        normalized_q = torch.empty_like(q)
        _q_norm_scalar_out_kernel[(512, q_items)](
            q,
            normalized_q,
            rstd,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        rope_q = torch.empty_like(q)
        _q_copy_scalar_all_kernel[(512, q_items)](
            normalized_q,
            rope_q,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        _q_rope_kernel[(32, num_heads, num_tokens)](
            normalized_q,
            rope_q,
            position_ids,
            cos_sin_cache,
            cos_sin_cache.stride(0),
            num_heads,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        _q_copy_scalar_all_kernel[(512, q_items)](
            rope_q,
            q,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
        if slot_mapping.numel() != 0:
            for token in range(slot_mapping.numel()):
                _kv_copy_scalar_kernel[(512,)](
                    kv,
                    k_cache,
                    slot_mapping,
                    kv.stride(0),
                    k_cache.stride(0),
                    k_cache.stride(1),
                    cache_block_size,
                    TOKEN=token,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
                _kv_rope_scalar_kernel[(32,)](
                    kv,
                    k_cache,
                    slot_mapping,
                    position_ids,
                    cos_sin_cache,
                    kv.stride(0),
                    k_cache.stride(0),
                    k_cache.stride(1),
                    cos_sin_cache.stride(0),
                    cache_block_size,
                    TOKEN=token,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
