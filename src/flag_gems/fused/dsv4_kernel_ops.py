from collections.abc import Sequence
from typing import Optional

import torch

from flag_gems.fused.dsv4_attention import (
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_flash_mla_sparse_decode,
    dsv4_flash_mla_sparse_prefill,
    dsv4_fp8_einsum,
    dsv4_fused_q_kv_rmsnorm,
    dsv4_qnorm_rope_kv_rope_quant_insert,
)


def dsv4_kernel_fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
):
    return dsv4_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def dsv4_kernel_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
    *,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> None:
    dsv4_qnorm_rope_kv_rope_quant_insert(
        q=q,
        kv=kv,
        k_cache=k_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        eps=eps,
        block_size=block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )


def dsv4_kernel_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    *,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> None:
    dsv4_dequantize_and_gather_k_cache(
        out=out,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=offset,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )


def dsv4_kernel_compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: Optional[torch.Tensor] = None,
):
    if is_valid_token is None:
        is_valid_token = torch.ones(
            (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
        )
    return dsv4_compute_global_topk_indices_and_lens(
        topk_indices=topk_indices,
        token_to_req_indices=token_to_req,
        block_table=block_table,
        block_size=block_size,
        is_valid_token=is_valid_token,
    )


def dsv4_kernel_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
):
    return dsv4_combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=window_size,
        compress_ratio=compress_ratio,
        topk=topk,
        M=M,
        N=N,
    )


def dsv4_kernel_flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    return dsv4_flash_mla_sparse_prefill(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )


def dsv4_kernel_flash_mla_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    sm_scale: float,
    head_dim_v: int,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 64,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
):
    return dsv4_flash_mla_sparse_decode(
        q=q,
        k_cache=k_cache,
        indices=indices_in_kvcache,
        sm_scale=sm_scale,
        head_dim_v=head_dim_v,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        out=out,
        block_size=block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )


def dsv4_kernel_deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: Sequence[int],
) -> None:
    dsv4_fp8_einsum(
        a=a,
        a_scale=a_scale,
        b=b,
        b_scale=b_scale,
        out=out,
        equation=equation,
        recipe=list(recipe),
    )


def dsv4_kernel_get_mla_metadata(*args, **kwargs):
    from vllm.v1.attention.ops.flashmla import get_mla_metadata

    return get_mla_metadata(*args, **kwargs)


def dsv4_kernel_persistent_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    torch.ops._C.persistent_topk(logits, lengths, output, workspace, k, max_seq_len)


def dsv4_kernel_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk: int,
) -> None:
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        out_indices,
        num_rows,
        stride0,
        stride1,
        topk,
    )


def dsv4_kernel_cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        cu_seq_lens,
    )


__all__ = [
    "dsv4_kernel_combine_topk_swa_indices",
    "dsv4_kernel_compute_global_topk_indices_and_lens",
    "dsv4_kernel_cp_gather_indexer_k_quant_cache",
    "dsv4_kernel_deepseek_v4_fp8_einsum",
    "dsv4_kernel_dequantize_and_gather_k_cache",
    "dsv4_kernel_flash_mla_sparse_decode",
    "dsv4_kernel_flash_mla_sparse_fwd",
    "dsv4_kernel_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "dsv4_kernel_fused_q_kv_rmsnorm",
    "dsv4_kernel_get_mla_metadata",
    "dsv4_kernel_persistent_topk",
    "dsv4_kernel_top_k_per_row_prefill",
]
