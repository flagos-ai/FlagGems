from typing import Optional, Tuple

import torch

import flag_gems
from flag_gems.patches.patch_util import patch_module_method, patch_vllm_lib


def custom_gems_rms_forward_cuda(self, x, residual=None):
    from flag_gems.modules.normalization import gems_rms_forward

    return gems_rms_forward(x, residual, self.weight, self.variance_epsilon)


def custom_gems_rope_forward_cuda(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]

    query_shape = query.shape
    key_shape = key.shape
    query = query.view(num_tokens, -1, self.head_size)
    key = key.view(num_tokens, -1, self.head_size)

    query_rot = query[..., : self.rotary_dim]
    key_rot = key[..., : self.rotary_dim]
    if self.rotary_dim < self.head_size:
        query_pass = query[..., self.rotary_dim :]
        key_pass = key[..., self.rotary_dim :]

    cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

    q_embed, k_embed = gems_rope_forward(
        query_rot,
        key_rot,
        cos,
        sin,
        position_ids=positions,
        rotary_interleaved=not self.is_neox_style,
        inplace=True,  # set inplace to True for vLLM compatibility
    )

    if self.rotary_dim < self.head_size:
        query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
    else:
        query = q_embed.reshape(query_shape)
        key = k_embed.reshape(key_shape)

    return query, key


def custom_gems_silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


def custom_gems_write_to_paged_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    from flag_gems.fused.reshape_and_cache import reshape_and_cache

    reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping.flatten(),
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def custom_gems_flash_mla_forward(
    self,
    q_nope,
    q_pe,
    kv_c_and_k_pe_cache,
    attn_metadata,
) -> torch.Tensor:
    from flag_gems.fused import flash_mla

    assert kv_c_and_k_pe_cache.numel() > 0
    assert attn_metadata.decode is not None

    if self.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError("FP8 Triton MLA not yet supported")

    batch, num_head_q, head_dim_v = q_nope.shape
    seqlen_q = 1

    q = torch.cat([q_nope, q_pe], dim=-1)
    head_dim = q.shape[-1]
    q = q.view(batch, seqlen_q, num_head_q, head_dim)

    # Add a head dim of 1
    kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
    PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

    block_table = attn_metadata.decode.block_table
    output = flash_mla(
        q,
        block_table,
        kv_c_and_k_pe_cache,
        None,
        PAGE_SIZE,
        batch,
        seqlen_q,
        attn_metadata.decode.seq_lens,
        num_head_q,
        None,
        head_dim,
        head_dim_v,
        True,
    )

    o = self._v_up_proj_and_o_proj(output)
    return o


def custom_gems_flash_attention_impl_forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,  #: FlashAttentionMetadata,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from flag_gems import flash_attn_varlen_func, reshape_and_cache_flash

    assert output is not None, "Output tensor must be provided."

    if output_scale is not None:
        raise NotImplementedError(
            "fused output quantization is not yet supported" " for FlashAttentionImpl"
        )

    if attn_metadata is None:
        # Profiling run.
        return output

    num_actual_tokens = attn_metadata.num_actual_tokens
    key_cache, value_cache = kv_cache.unbind(0)

    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )

    # TODO: Support FP8
    if self.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "FP8 quantization is not yet supported for FlashAttentionImpl"
        )
        # key_cache = key_cache.view(torch.float8_e4m3fn)
        # value_cache = value_cache.view(torch.float8_e4m3fn)
        # num_tokens, num_heads, head_size = query.shape
        # query, _ = ops.scaled_fp8_quant(
        #     query.reshape((num_tokens, num_heads * head_size)).contiguous(),
        #     layer._q_scale,
        # )
        # query = query.reshape((num_tokens, num_heads, head_size))

    # Compute attention and update output up to `num_actual_tokens`.
    use_local_attn = self.use_irope and attn_metadata.local_attn_metadata is not None

    if not attn_metadata.use_cascade or use_local_attn:
        if use_local_attn:
            assert attn_metadata.local_attn_metadata is not None
            local_metadata = attn_metadata.local_attn_metadata
            cu_seqlens_q = local_metadata.local_query_start_loc
            seqused_k = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
            block_table = local_metadata.local_block_table
            scheduler_metadata = local_metadata.local_scheduler_metadata
        else:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=2,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )
        return output

    # TODO: Support cascade_attention.
    raise NotImplementedError("Cascade attention is not implemented in flag_gems.")


def custom_silu_and_mul(out: torch.Tensor, input: torch.Tensor):
    d = input.size(-1) // 2
    x, y = input.split(d, dim=-1)
    flag_gems.silu_and_mul_out(x, y, out)


def custom_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    flag_gems.moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )


def custom_topk_softmax(
    topk_weights, topk_indices, token_expert_indices, gating_output
):
    flag_gems.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )


def custom_get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    seqused_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
):
    return flag_gems.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        headdim,
        headdim_v,
        qkv_dtype,
        seqused_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=cu_seqlens_k_new,
        seqused_q=seqused_q,
        leftpad_k=leftpad_k,
        page_size=page_size,
        max_seqlen_k_new=max_seqlen_k_new,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        has_softcap=has_softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )


def apply_gems_patches_to_vllm(verbose=True):
    import vllm  # noqa: F401
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

    patch_module_method(RMSNorm, "forward_cuda", custom_gems_rms_forward_cuda, verbose)
    patch_module_method(
        RotaryEmbedding, "forward_cuda", custom_gems_rope_forward_cuda, verbose
    )
    patch_module_method(
        PagedAttention,
        "write_to_paged_cache",
        custom_gems_write_to_paged_cache,
        verbose,
    )
    patch_module_method(SiluAndMul, "forward_cuda", custom_gems_silu_and_mul, verbose)
    patch_module_method(
        TritonMLAImpl, "_forward_decode", custom_gems_flash_mla_forward, verbose
    )
    patch_module_method(
        FlashAttentionImpl, "forward", custom_gems_flash_attention_impl_forward, verbose
    )
    patch_vllm_lib("_C", "silu_and_mul", custom_silu_and_mul, "CUDA", verbose)
    patch_vllm_lib(
        "_moe_C", "moe_align_block_size", custom_moe_align_block_size, "CUDA", verbose
    )
    patch_vllm_lib("_moe_C", "topk_softmax", custom_topk_softmax, "CUDA", verbose)
    patch_vllm_lib(
        "_vllm_fa3_C",
        "get_scheduler_metadata",
        custom_get_scheduler_metadata,
        "CUDA",
        verbose,
    )
