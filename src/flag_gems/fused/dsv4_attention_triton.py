import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.fused.flashmla_sparse import triton_flash_mla_sparse_fwd
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul import (
    w8a8_block_fp8_matmul,
)


_INT32_MAX = 2147483647
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def _next_power_of_2_or_1(x: int) -> int:
    return 1 if x <= 1 else triton.next_power_of_2(x)


def _default_scale_slots(nope_dim: int) -> int:
    """DeepSeek-V4 fp8_ds_mla stores one padding scale byte after NoPE scales."""
    return triton.cdiv(nope_dim, 64) + (1 if nope_dim % 64 == 0 else 0)


def _as_cache_2d(k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.ndim == 2:
        return k_cache
    return k_cache.contiguous().view(k_cache.shape[0], -1)


@triton.jit(do_not_specialize=["eps"])
def _dsv4_fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    if task == 0:
        size = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        row_out = q_out_ptr + token_idx * q_out_stride
        weight_ptr = q_weight_ptr
    else:
        size = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        row_out = kv_out_ptr + token_idx * kv_out_stride
        weight_ptr = kv_weight_ptr

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / size
    rrms = tl.rsqrt(var + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + offs, y.to(row_out.dtype.element_ty), mask=mask)


def dsv4_fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 QR/KV RMSNorm with vLLM-compatible two-output semantics."""
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    with torch_device_fn.device(qr.device):
        _dsv4_fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
            qr,
            qr_out,
            q_weight,
            qr.stride(0),
            qr_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            eps,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            BLOCK_SIZE=block_size,
        )
    return qr_out, kv_out


@triton.jit(do_not_specialize=["eps"])
def _dsv4_qnorm_rope_kv_rope_quant_insert_kernel(
    q_ptr,
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_ptr,
    eps,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_t,
    kv_stride_d,
    cache_block_stride,
    cos_sin_stride_t,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    TOKEN_DATA_SIZE: tl.constexpr,
    SCALE_SLOTS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    Q_RMS_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    task = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    mask_head = offs < HEAD_DIM
    rope_start = HEAD_DIM - ROPE_DIM
    pos = tl.load(positions_ptr + token_idx).to(tl.int64)

    if task < NUM_HEADS:
        q_base = q_ptr + token_idx.to(tl.int64) * q_stride_t + task * q_stride_h
        q = tl.load(q_base + offs * q_stride_d, mask=mask_head, other=0.0).to(tl.float32)
        var = tl.sum(q * q, axis=0) / HEAD_DIM
        rrms = tl.rsqrt(var + eps)
        q = q * rrms

        is_rope = (offs >= rope_start) & mask_head
        rope_off = offs - rope_start
        pair_id = rope_off // 2
        is_even = (rope_off % 2) == 0
        mate_rope_off = tl.where(is_even, rope_off + 1, rope_off - 1)
        mate_off = rope_start + mate_rope_off
        q_mate = tl.load(
            q_base + mate_off * q_stride_d,
            mask=is_rope,
            other=0.0,
        ).to(tl.float32) * rrms
        cos = tl.load(
            cos_sin_ptr + pos * cos_sin_stride_t + pair_id,
            mask=is_rope,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + pos * cos_sin_stride_t + pair_id + (ROPE_DIM // 2),
            mask=is_rope,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(is_even, q * cos - q_mate * sin, q_mate * sin + q * cos)
        out = tl.where(is_rope, rotated, q)
        tl.store(q_base + offs * q_stride_d, out.to(q_ptr.dtype.element_ty), mask=mask_head)
        return

    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1:
        return

    block_idx = slot // CACHE_BLOCK_SIZE
    pos_in_block = slot - block_idx * CACHE_BLOCK_SIZE
    cache_block = k_cache_ptr + block_idx.to(tl.int64) * cache_block_stride
    token_data = cache_block + pos_in_block * TOKEN_DATA_SIZE
    scale_base = cache_block + CACHE_BLOCK_SIZE * TOKEN_DATA_SIZE + pos_in_block * SCALE_SLOTS

    kv_base = kv_ptr + token_idx.to(tl.int64) * kv_stride_t
    kv_vals = tl.load(kv_base + offs * kv_stride_d, mask=mask_head, other=0.0).to(tl.float32)

    is_k_rope = (offs >= NOPE_DIM) & (offs < NOPE_DIM + ROPE_DIM)
    k_rope_off = offs - NOPE_DIM
    k_pair_id = k_rope_off // 2
    k_is_even = (k_rope_off % 2) == 0
    k_mate_rope_off = tl.where(k_is_even, k_rope_off + 1, k_rope_off - 1)
    k_mate_off = NOPE_DIM + k_mate_rope_off
    kv_mate = tl.load(kv_base + k_mate_off * kv_stride_d, mask=is_k_rope, other=0.0).to(tl.float32)
    k_cos = tl.load(
        cos_sin_ptr + pos * cos_sin_stride_t + k_pair_id,
        mask=is_k_rope,
        other=1.0,
    ).to(tl.float32)
    k_sin = tl.load(
        cos_sin_ptr + pos * cos_sin_stride_t + k_pair_id + (ROPE_DIM // 2),
        mask=is_k_rope,
        other=0.0,
    ).to(tl.float32)
    kv_rot = tl.where(k_is_even, kv_vals * k_cos - kv_mate * k_sin, kv_mate * k_sin + kv_vals * k_cos)
    kv_final = tl.where(is_k_rope, kv_rot, kv_vals)

    for qblock in tl.static_range(0, SCALE_SLOTS):
        start = qblock * Q_RMS_BLOCK
        qoffs = start + tl.arange(0, Q_RMS_BLOCK)
        qmask = qoffs < NOPE_DIM
        if start < NOPE_DIM:
            x = tl.load(kv_base + qoffs * kv_stride_d, mask=qmask, other=0.0).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
            exponent = tl.ceil(tl.log2(amax / FP8_MAX))
            scale = tl.exp2(exponent)
            qx = tl.clamp(x / scale, -FP8_MAX, FP8_MAX)
            qx_fp8 = qx.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            tl.store(token_data + qoffs, qx_fp8, mask=qmask)
            encoded = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
            tl.store(scale_base + qblock, encoded.to(tl.uint8))
        else:
            tl.store(scale_base + qblock, tl.zeros((), dtype=tl.uint8))

    rope_offsets = tl.arange(0, Q_RMS_BLOCK)
    for rblock in tl.static_range(0, ROPE_DIM, Q_RMS_BLOCK):
        r_offs = rblock + rope_offsets
        r_mask = r_offs < ROPE_DIM
        vals = tl.load(
            kv_base + (NOPE_DIM + r_offs) * kv_stride_d,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        pair = r_offs // 2
        even = (r_offs % 2) == 0
        mate = tl.where(even, r_offs + 1, r_offs - 1)
        mate_vals = tl.load(
            kv_base + (NOPE_DIM + mate) * kv_stride_d,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        cos = tl.load(cos_sin_ptr + pos * cos_sin_stride_t + pair, mask=r_mask, other=1.0).to(tl.float32)
        sin = tl.load(
            cos_sin_ptr + pos * cos_sin_stride_t + pair + (ROPE_DIM // 2),
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        rot = tl.where(even, vals * cos - mate_vals * sin, mate_vals * sin + vals * cos)
        bf16_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        tl.store(bf16_ptr + r_offs, rot.to(tl.bfloat16), mask=r_mask)


def dsv4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> None:
    """In-place q RMSNorm/RoPE and DeepSeek-V4 fp8_ds_mla cache insert.

    The cache layout is block-major: token data for every token in the block
    comes first, followed by per-token UE8M0 scale bytes. This file owns the
    layout adaptation instead of changing existing FlagGems cache operators.
    """
    assert q.ndim == 3 and kv.ndim == 2
    assert q.shape[0] == kv.shape[0] == positions.shape[0]
    assert slot_mapping.shape[0] == positions.shape[0]
    assert q.stride(-1) == 1 and kv.stride(-1) == 1
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.stride(-1) == 1

    head_dim = q.shape[-1]
    assert kv.shape[-1] == head_dim
    assert rope_dim % 2 == 0 and head_dim >= rope_dim
    if nope_dim is None:
        nope_dim = head_dim - rope_dim
    assert nope_dim + rope_dim <= head_dim
    if scale_slots is None:
        scale_slots = _default_scale_slots(nope_dim)

    k_cache_2d = _as_cache_2d(k_cache)
    token_data_size = nope_dim + rope_dim * 2
    block = triton.next_power_of_2(head_dim)
    qblock = 64
    with torch_device_fn.device(q.device):
        _dsv4_qnorm_rope_kv_rope_quant_insert_kernel[
            (q.shape[0], q.shape[1] + 1)
        ](
            q,
            kv,
            k_cache_2d,
            slot_mapping,
            positions.to(torch.int64),
            cos_sin_cache,
            eps,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            k_cache_2d.stride(0),
            cos_sin_cache.stride(0),
            NUM_HEADS=q.shape[1],
            HEAD_DIM=head_dim,
            NOPE_DIM=nope_dim,
            ROPE_DIM=rope_dim,
            BLOCK_SIZE=block,
            CACHE_BLOCK_SIZE=block_size,
            TOKEN_DATA_SIZE=token_data_size,
            SCALE_SLOTS=scale_slots,
            FP8_MAX=448.0,
            Q_RMS_BLOCK=qblock,
            num_warps=8,
        )


@triton.jit
def _dsv4_dequantize_and_gather_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    scale_slots: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    cache_block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    num_workers: tl.constexpr,
    HAVE_GATHER_LENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    if HAVE_GATHER_LENS:
        gather_len = tl.load(gather_lens_ptr + req_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for local_i in range(worker_idx, gather_len, num_workers):
        pos = start_pos + local_i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos - block_in_seq * cache_block_size
        physical_block = tl.load(block_table_ptr + req_idx * max_blocks_per_seq + block_in_seq)
        cache_block = k_cache_ptr + physical_block.to(tl.int64) * cache_block_stride
        token_data = cache_block + pos_in_block * token_data_size
        scale_base = cache_block + cache_block_size * token_data_size + pos_in_block * scale_slots
        out_row = out_ptr + req_idx * out_stride0 + (offset + local_i) * out_stride1

        for qblock in tl.static_range(0, scale_slots):
            qoffs = qblock * quant_block + tl.arange(0, quant_block)
            qmask = qoffs < nope_dim
            x_u8 = tl.load(token_data + qoffs, mask=qmask, other=0).to(tl.uint8)
            x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            encoded = tl.load(scale_base + qblock)
            scale = tl.exp2(encoded.to(tl.float32) - 127.0)
            x = x_fp8 * scale
            tl.store(out_row + qoffs, x.to(tl.bfloat16), mask=qmask)

        bf16_ptr = (token_data + nope_dim).to(tl.pointer_type(tl.bfloat16))
        for rblock in tl.static_range(0, rope_dim, 16):
            roffs = rblock + tl.arange(0, 16)
            rmask = roffs < rope_dim
            vals = tl.load(bf16_ptr + roffs, mask=rmask, other=0.0)
            tl.store(out_row + nope_dim + roffs, vals, mask=rmask)


def dsv4_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int = 0,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> None:
    assert out.ndim == 3 and out.dtype == torch.bfloat16
    assert seq_lens.ndim == 1 and block_table.ndim == 2
    assert seq_lens.shape[0] == block_table.shape[0] <= out.shape[0]
    output_dim = out.shape[-1]
    if nope_dim is None:
        nope_dim = output_dim - rope_dim
    if scale_slots is None:
        scale_slots = _default_scale_slots(nope_dim)
    assert nope_dim + rope_dim <= output_dim

    k_cache_2d = _as_cache_2d(k_cache)
    token_data_size = nope_dim + rope_dim * 2
    num_reqs = seq_lens.shape[0]
    num_workers = 128
    with torch_device_fn.device(out.device):
        _dsv4_dequantize_and_gather_k_cache_kernel[(num_reqs, num_workers)](
            out,
            out.stride(0),
            out.stride(1),
            k_cache_2d,
            seq_lens,
            block_table,
            offset,
            gather_lens,
            block_table.shape[-1],
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            scale_slots=scale_slots,
            quant_block=64,
            cache_block_size=block_size,
            token_data_size=token_data_size,
            cache_block_stride=k_cache_2d.stride(0),
            output_dim=output_dim,
            num_workers=num_workers,
            HAVE_GATHER_LENS=gather_lens is not None,
        )


@triton.jit
def _dsv4_compute_global_topk_indices_and_lens_kernel(
    global_indices_ptr,
    global_stride,
    lens_ptr,
    local_indices_ptr,
    local_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    count = tl.zeros((), dtype=tl.int32)

    for start in range(0, topk, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < topk
        local_idx = tl.load(local_indices_ptr + token_idx * local_stride + offs, mask=mask, other=-1)
        valid = local_idx >= 0
        block_idx = local_idx // block_size
        block_off = local_idx - block_idx * block_size
        block_no = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_idx,
            mask=mask & valid,
            other=0,
        )
        slot = block_no * block_size + block_off
        slot = tl.where(valid, slot, -1)
        tl.store(global_indices_ptr + token_idx * global_stride + offs, slot, mask=mask)
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


def dsv4_compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    num_tokens, topk = topk_indices.shape
    global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    with torch_device_fn.device(topk_indices.device):
        _dsv4_compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
            global_indices,
            global_indices.stride(0),
            lens,
            topk_indices,
            topk_indices.stride(0),
            topk,
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            is_valid_token,
            BLOCK=1024,
        )
    return global_indices, lens


@triton.jit
def _dsv4_combine_topk_swa_indices_kernel(
    combined_ptr,
    combined_stride,
    lens_ptr,
    topk_ptr,
    topk_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_idx, query_end, num_workers):
        token_in_query = token_idx - query_start
        pos = start_pos + token_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        offs = tl.arange(0, PADDED_TOP_K)
        mask = offs < topk_len
        topk_vals = tl.load(topk_ptr + token_idx * topk_stride + offs, mask=mask, other=-1)
        tl.store(combined_ptr + token_idx * combined_stride + offs, topk_vals + M * batch_idx, mask=mask)

        swa_offs = tl.arange(0, PADDED_WINDOW_SIZE)
        tl.store(
            combined_ptr + token_idx * combined_stride + topk_len + swa_offs,
            M * batch_idx + N + swa_offs + pos - swa_len + 1 - gather_start,
            mask=(swa_offs < swa_len) & (swa_offs < WINDOW_SIZE),
        )
        tl.store(lens_ptr + token_idx, topk_len + swa_len)


def dsv4_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined = torch.full(
        (num_tokens, combined_topk),
        -1,
        device=topk_indices.device,
        dtype=torch.int32,
    )
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    with torch_device_fn.device(topk_indices.device):
        _dsv4_combine_topk_swa_indices_kernel[(num_reqs, 128)](
            combined,
            combined.stride(0),
            lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            PADDED_TOP_K=_next_power_of_2_or_1(topk_indices.shape[-1]),
            PADDED_WINDOW_SIZE=_next_power_of_2_or_1(window_size),
        )
    return combined, lens


def dsv4_flash_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlagGems-only sparse prefill wrapper with vLLM-style optional out."""
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    sq, h, dt = q.shape
    skv, vg, _ = kv.shape
    assert d_v == 512
    assert kv.shape[-1] == dt
    assert indices.shape[:2] == (sq, vg)
    assert vg == 1
    assert h in (64, 128)
    assert dt in (512, 576)
    if attn_sink is not None:
        assert attn_sink.shape == (h,)
    if topk_length is not None:
        assert topk_length.shape == (sq,)

    _, _, topk = indices.shape
    # The shared FlagGems sparse kernel has a non-causal loop skip keyed on
    # SKV. vLLM sparse prefill must still process every top-k slot when
    # topk > skv, so pass a logical SKV large enough for the loop and sanitize
    # physically out-of-range indices before the launch.
    kernel_skv = max(skv, topk)
    indices_for_kernel = indices
    if kernel_skv != skv:
        invalid = (indices < 0) | (indices >= skv)
        indices_for_kernel = torch.where(
            invalid,
            torch.full_like(indices, -1),
            indices,
        ).contiguous()
    td = dt - d_v
    dp = triton.next_power_of_2(d_v)
    tdp = 0 if td == 0 else triton.next_power_of_2(td)
    group = h // vg
    bh = max(16, min(32, triton.next_power_of_2(group)))
    nh = triton.cdiv(group, bh)
    bk = 16
    if out is None:
        output = torch.zeros((sq, h, d_v), device=q.device, dtype=q.dtype)
    else:
        assert out.shape == (sq, h, d_v)
        assert out.dtype == q.dtype
        output = out
        output.zero_()
    max_logits = torch.full((sq, h), float("-inf"), device=q.device, dtype=torch.float32)
    lse = torch.full((sq, h), float("-inf"), device=q.device, dtype=torch.float32)
    q_idx_i64 = q.numel() > _INT32_MAX
    output_idx_i64 = output.numel() > _INT32_MAX
    grid = (sq, vg * nh, 1)
    with torch_device_fn.device(q.device):
        triton_flash_mla_sparse_fwd[grid](
            q,
            kv,
            indices_for_kernel,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits,
            lse,
            q.stride(1),
            q.stride(0),
            q.stride(2),
            kv.stride(1),
            kv.stride(0),
            kv.stride(2),
            indices_for_kernel.stride(1),
            indices_for_kernel.stride(0),
            indices_for_kernel.stride(2),
            attn_sink.stride(0) if attn_sink is not None else 0,
            topk_length.stride(0) if topk_length is not None else 0,
            output.stride(1),
            output.stride(0),
            output.stride(2),
            max_logits.stride(1),
            max_logits.stride(0),
            lse.stride(1),
            lse.stride(0),
            sq,
            kernel_skv,
            topk,
            d_v,
            td,
            dp,
            tdp,
            group,
            bk,
            bh,
            False,
            q_idx_i64,
            output_idx_i64,
            attn_sink is not None,
            topk_length is not None,
        )
    return output, max_logits, lse


@triton.jit
def _dsv4_build_decode_sparse_prefill_inputs_one_cache_kernel(
    kv_out_ptr,
    indices_out_ptr,
    k_cache_ptr,
    source_indices_ptr,
    source_length_ptr,
    cache_block_stride,
    source_indices_stride_t,
    slot_base,
    total_slots,
    source_topk,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    scale_slots: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    HAVE_LENGTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    local_slot = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    global_slot = slot_base + local_slot
    out_kv_idx = token_idx * total_slots + global_slot

    valid_len = source_topk
    if HAVE_LENGTH:
        valid_len = tl.minimum(tl.load(source_length_ptr + token_idx), source_topk)
    is_valid = local_slot < valid_len
    source_index = tl.load(
        source_indices_ptr + token_idx * source_indices_stride_t + local_slot,
        mask=local_slot < source_topk,
        other=-1,
    )
    is_valid = is_valid & (source_index >= 0)

    block_idx = source_index // cache_block_size
    pos_in_block = source_index - block_idx * cache_block_size
    token_data = (
        k_cache_ptr
        + block_idx.to(tl.int64) * cache_block_stride
        + pos_in_block * token_data_size
    )
    scale_base = (
        k_cache_ptr
        + block_idx.to(tl.int64) * cache_block_stride
        + cache_block_size * token_data_size
        + pos_in_block * scale_slots
    )

    for qblock in tl.static_range(0, scale_slots):
        qoffs = qblock * quant_block + d
        mask = (qoffs < nope_dim) & is_valid
        x_u8 = tl.load(token_data + qoffs, mask=mask, other=0).to(tl.uint8)
        x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        encoded = tl.load(scale_base + qblock, mask=is_valid, other=127)
        scale = tl.exp2(encoded.to(tl.float32) - 127.0)
        x = x_fp8 * scale
        tl.store(
            kv_out_ptr + out_kv_idx * (nope_dim + rope_dim) + qoffs,
            x.to(tl.bfloat16),
            mask=qoffs < nope_dim,
        )

    bf16_ptr = (token_data + nope_dim).to(tl.pointer_type(tl.bfloat16))
    for rblock in tl.static_range(0, rope_dim, BLOCK_D):
        roffs = rblock + d
        mask = (roffs < rope_dim) & is_valid
        vals = tl.load(bf16_ptr + roffs, mask=mask, other=0.0)
        tl.store(
            kv_out_ptr + out_kv_idx * (nope_dim + rope_dim) + nope_dim + roffs,
            vals,
            mask=roffs < rope_dim,
        )

    tl.store(
        indices_out_ptr + token_idx * total_slots + global_slot,
        tl.where(is_valid, out_kv_idx, -1),
    )


def dsv4_flash_mla_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    head_dim_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 64,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse decode via fp8_ds_mla gather plus FlagGems sparse prefill kernel."""
    if q.ndim == 4:
        bsz, sq, heads, head_dim = q.shape
        q_flat = q.reshape(bsz * sq, heads, head_dim).contiguous()
    else:
        assert q.ndim == 3
        bsz, sq = q.shape[0], 1
        heads, head_dim = q.shape[1], q.shape[2]
        q_flat = q.contiguous()
    assert indices.ndim in (2, 3)
    if indices.ndim == 3:
        indices_flat = indices.reshape(q_flat.shape[0], indices.shape[-1]).contiguous()
    else:
        indices_flat = indices.contiguous()
    assert indices_flat.shape[0] == q_flat.shape[0]
    if nope_dim is None:
        nope_dim = head_dim - rope_dim
    if scale_slots is None:
        scale_slots = _default_scale_slots(nope_dim)
    assert head_dim == nope_dim + rope_dim
    assert head_dim_v == 512

    extra_flat = None
    if extra_indices_in_kvcache is not None:
        if extra_indices_in_kvcache.ndim == 3:
            extra_flat = extra_indices_in_kvcache.reshape(q_flat.shape[0], extra_indices_in_kvcache.shape[-1]).contiguous()
        else:
            extra_flat = extra_indices_in_kvcache.contiguous()

    topk = indices_flat.shape[-1]
    extra_topk = 0 if extra_flat is None else extra_flat.shape[-1]
    total_slots = topk + extra_topk
    kv = torch.empty((q_flat.shape[0] * total_slots, 1, head_dim), device=q.device, dtype=torch.bfloat16)
    sparse_indices = torch.full(
        (q_flat.shape[0], 1, total_slots),
        -1,
        device=q.device,
        dtype=torch.int32,
    )
    # Keep the full padded slot range visible to FlashMLA. Invalid entries are
    # explicitly set to -1 by the gather kernels, so shorter per-token lengths
    # and extra-cache gaps are semantically masked without compacting rows.
    combined_lens = torch.full(
        (q_flat.shape[0],),
        total_slots,
        device=q.device,
        dtype=torch.int32,
    )
    k_cache_2d = _as_cache_2d(k_cache)
    extra_cache_2d = _as_cache_2d(extra_k_cache) if extra_k_cache is not None else k_cache_2d
    token_data_size = nope_dim + rope_dim * 2

    with torch_device_fn.device(q.device):
        _dsv4_build_decode_sparse_prefill_inputs_one_cache_kernel[
            (q_flat.shape[0], topk)
        ](
            kv,
            sparse_indices,
            k_cache_2d,
            indices_flat,
            topk_length,
            k_cache_2d.stride(0),
            indices_flat.stride(0),
            0,
            total_slots,
            topk,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            scale_slots=scale_slots,
            quant_block=64,
            cache_block_size=block_size,
            token_data_size=token_data_size,
            HAVE_LENGTH=topk_length is not None,
            BLOCK_D=64,
            num_warps=4,
        )
        if extra_flat is not None and extra_topk > 0:
            _dsv4_build_decode_sparse_prefill_inputs_one_cache_kernel[
                (q_flat.shape[0], extra_topk)
            ](
                kv,
                sparse_indices,
                extra_cache_2d,
                extra_flat,
                extra_topk_length,
                extra_cache_2d.stride(0),
                extra_flat.stride(0),
                topk,
                total_slots,
                extra_topk,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                scale_slots=scale_slots,
                quant_block=64,
                cache_block_size=block_size,
                token_data_size=token_data_size,
                HAVE_LENGTH=extra_topk_length is not None,
                BLOCK_D=64,
                num_warps=4,
            )

    if out is None:
        out_flat = None
    else:
        out_flat = out.reshape(q_flat.shape[0], heads, head_dim_v)
    output, _, lse = dsv4_flash_mla_sparse_prefill(
        q_flat,
        kv,
        sparse_indices,
        sm_scale,
        d_v=head_dim_v,
        attn_sink=attn_sink,
        topk_length=combined_lens,
        out=out_flat,
    )
    if q.ndim == 4:
        output = output.view(bsz, sq, heads, head_dim_v)
        lse = lse.view(bsz, sq, heads).transpose(1, 2).contiguous()
    return output, lse


def dsv4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if equation != "bhr,hdr->bhd":
        raise NotImplementedError(f"unsupported equation: {equation}")
    if tuple(recipe) != (1, 128, 128):
        raise NotImplementedError(f"unsupported recipe: {recipe}")

    assert a.ndim == 3 and b.ndim == 3 and out.ndim == 3
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == torch.float32 and b_scale.dtype == torch.float32
    assert out.dtype == torch.bfloat16

    batch, num_groups, k_dim = a.shape
    b_groups, n_dim, b_k_dim = b.shape
    assert num_groups == b_groups
    assert k_dim == b_k_dim
    assert out.shape == (batch, num_groups, n_dim)
    assert k_dim % 128 == 0 and n_dim % 128 == 0
    assert a_scale.shape == (batch, num_groups, k_dim // 128)
    assert b_scale.shape == (num_groups, n_dim // 128, k_dim // 128)

    block_size = [128, 128]
    for group_idx in range(num_groups):
        out_group = w8a8_block_fp8_matmul(
            a[:, group_idx, :],
            b[group_idx],
            a_scale[:, group_idx, :],
            b_scale[group_idx],
            block_size,
            output_dtype=out.dtype,
        )
        out[:, group_idx, :].copy_(out_group)


def dsv4_attention_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    *,
    k_cache: Optional[torch.Tensor] = None,
    slot_mapping: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    prefill_indices: Optional[torch.Tensor] = None,
    decode_indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_decode_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    block_size: int = 64,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Minimal tensor-level DSV4 attention path.

    This is intentionally independent from vLLM layer/context objects. Callers
    pass already projected q/kv tensors plus explicit sparse metadata.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    if k_cache is not None and slot_mapping is not None and cos_sin_cache is not None:
        dsv4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            k_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            eps=eps,
            block_size=block_size,
            rope_dim=rope_dim,
            nope_dim=nope_dim,
            scale_slots=scale_slots,
        )
    if prefill_indices is not None:
        dsv4_flash_mla_sparse_prefill(
            q,
            kv.view(-1, 1, kv.shape[-1]).contiguous(),
            prefill_indices,
            sm_scale,
            d_v=out.shape[-1],
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )
        return out
    if decode_indices is not None:
        assert k_cache is not None
        dsv4_flash_mla_sparse_decode(
            q.unsqueeze(1) if q.ndim == 3 else q,
            k_cache,
            decode_indices,
            sm_scale,
            head_dim_v=out.shape[-1],
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_decode_indices,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
            out=out.unsqueeze(1) if out.ndim == 3 else out,
            block_size=block_size,
            rope_dim=rope_dim,
            nope_dim=nope_dim,
            scale_slots=scale_slots,
        )
        return out
    raise ValueError("Either prefill_indices or decode_indices must be provided.")


def dsv4_vllm_deepseek_v4_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    from vllm.forward_context import get_forward_context

    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.attention_impl(hidden_states, positions, out)


__all__ = [
    "dsv4_attention_triton",
    "dsv4_compute_global_topk_indices_and_lens",
    "dsv4_combine_topk_swa_indices",
    "dsv4_dequantize_and_gather_k_cache",
    "dsv4_fp8_einsum",
    "dsv4_flash_mla_sparse_decode",
    "dsv4_flash_mla_sparse_prefill",
    "dsv4_fused_q_kv_rmsnorm",
    "dsv4_qnorm_rope_kv_rope_quant_insert",
    "dsv4_vllm_deepseek_v4_attention",
]
