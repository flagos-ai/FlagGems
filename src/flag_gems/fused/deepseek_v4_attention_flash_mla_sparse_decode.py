from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd
from flag_gems.runtime import torch_device_fn


def _default_scale_slots(nope_dim: int) -> int:
    return triton.cdiv(nope_dim, 64) + (1 if nope_dim % 64 == 0 else 0)


def _as_cache_2d(k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.ndim == 2:
        return k_cache
    if k_cache.ndim == 3:
        if k_cache.is_contiguous():
            return k_cache.view(k_cache.shape[0], -1)
        return k_cache.contiguous().view(k_cache.shape[0], -1)
    raise ValueError(f"k_cache must be 2D or 3D, got shape={tuple(k_cache.shape)}")


@triton.jit
def _build_decode_sparse_prefill_inputs_one_cache_kernel(
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


def flash_mla_sparse_decode(
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
            extra_flat = extra_indices_in_kvcache.reshape(
                q_flat.shape[0], extra_indices_in_kvcache.shape[-1]
            ).contiguous()
        else:
            extra_flat = extra_indices_in_kvcache.contiguous()

    topk = indices_flat.shape[-1]
    extra_topk = 0 if extra_flat is None else extra_flat.shape[-1]
    total_slots = topk + extra_topk
    kv = torch.empty(
        (q_flat.shape[0] * total_slots, 1, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    sparse_indices = torch.full(
        (q_flat.shape[0], 1, total_slots), -1, device=q.device, dtype=torch.int32
    )
    combined_lens = torch.full(
        (q_flat.shape[0],), total_slots, device=q.device, dtype=torch.int32
    )
    k_cache_2d = _as_cache_2d(k_cache)
    extra_cache_2d = (
        _as_cache_2d(extra_k_cache) if extra_k_cache is not None else k_cache_2d
    )
    token_data_size = nope_dim + rope_dim * 2

    with torch_device_fn.device(q.device):
        _build_decode_sparse_prefill_inputs_one_cache_kernel[(q_flat.shape[0], topk)](
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
            _build_decode_sparse_prefill_inputs_one_cache_kernel[
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

    out_flat = None if out is None else out.reshape(q_flat.shape[0], heads, head_dim_v)
    output, _, lse = flash_mla_sparse_fwd(
        q_flat,
        kv,
        sparse_indices,
        sm_scale,
        d_v=head_dim_v,
        attn_sink=attn_sink,
        topk_length=combined_lens,
    )
    if out_flat is not None:
        out_flat.copy_(output)
        output = out_flat
    if q.ndim == 4:
        output = output.view(bsz, sq, heads, head_dim_v)
        lse = lse.view(bsz, sq, heads).transpose(1, 2).contiguous()
    return output, lse


__all__ = ["flash_mla_sparse_decode"]
