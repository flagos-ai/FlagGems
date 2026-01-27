import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

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
    output_block_scale: Optional[torch.Tensor] = None,
    **kwargs,
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
    # use_local_attn = self.use_irope and attn_metadata.local_attn_metadata is not None
    use_local_attn = (
        getattr(self, "use_irope", False)
        and getattr(attn_metadata, "local_attn_metadata", None) is not None
    )
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
            s_aux=None,
            num_splits=0,
            cp_world_size=1,
            cp_rank=0,
            cp_tot_seqused_k=None,
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


def custom_moe_grouped_topk(
    gating_output: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    from flag_gems.fused import grouped_topk

    return grouped_topk(
        scores=gating_output,
        n_group=n_group,
        topk_group=topk_group,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        bias=bias,
        scoring_func=scoring_func,
    )


def custom_topk_softmax(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    flag_gems.topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
    )


def custom_apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
):
    return flag_gems.apply_repetition_penalties(
        logits, prompt_mask, output_mask, repetition_penalties
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


def custom_per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
):
    from flag_gems.ops import per_token_group_quant_fp8

    column_major_scales = output_s.stride(0) < output_s.stride(1)

    x_q, x_s = per_token_group_quant_fp8(
        x=input,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_ue8m0=scale_ue8m0,
    )

    output_q.copy_(x_q)
    output_s.copy_(x_s)


def custom_cutlass_scaled_mm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return flag_gems.cutlass_scaled_mm(output, input, weight, scale_a, scale_b, bias)


def custom_concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    return flag_gems.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


def custom_gems_flashattn_mla_forward_decode(
    self,
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata,  # FlashAttnMLAMetadata
    layer,  # AttentionLayer
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from flag_gems import flash_attn_varlen_func

    assert kv_c_and_k_pe_cache.numel() > 0
    assert attn_metadata.decode is not None

    if type(q) is tuple:
        q_nope, q_pe = q
    else:
        q_nope, q_pe = torch.split(
            q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

    if self.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError("FP8 FlashAttention MLA not yet supported")

    kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
    k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]

    # NOTE(matt): During CUDA graph capture, max_query_len can be 0, but the
    # kernel uses this to calculate grid dimensions. Ensure it's at least 1
    # to prevent invalid grid configuration during graph capture.
    max_seqlen_q = max(attn_metadata.decode.max_query_len, 1)

    attn_out = flash_attn_varlen_func(
        q=q_pe,
        k=k_pe_cache.unsqueeze(-2),  # Add head dim of 1
        v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
        q_v=q_nope,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=attn_metadata.decode.query_start_loc,
        max_seqlen_k=attn_metadata.decode.max_seq_len,
        seqused_k=attn_metadata.decode.seq_lens,
        block_table=attn_metadata.decode.block_table,
        softmax_scale=self.scale,
        causal=True,
        return_softmax_lse=self.need_to_return_lse_for_decode,
        fa_version=2,
        scheduler_metadata=attn_metadata.decode.scheduler_metadata,
        num_splits=0,
        cp_world_size=self.dcp_world_size,
        cp_rank=self.dcp_rank,
        cp_tot_seqused_k=attn_metadata.decode.dcp_tot_seq_lens,
    )

    if self.need_to_return_lse_for_decode:
        o, lse = attn_out
        # FA returns LSE in shape [ H, B ] but DCP wants [ B, H ]
        return o, lse.transpose(0, 1)  # [ H, B ] -> [ B, H ]
    else:
        o = attn_out
        return o, None


def patch_vllm_vit_to_attn(vitw):
    _orig_vit = vitw.vit_xformers_attn_wrapper

    def _seqlens_to_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
        cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        return F.pad(cu_seqlens, (1, 0))

    def _wrapped_vit_xformers_attn_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if os.getenv("VIT_ATTN_BACKEND", "xformers") == "no-sdpa":
            return _orig_vit(q, k, v, seqlens)

        cu_seqlens = _seqlens_to_cu_seqlens(seqlens)

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            ctx = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
        except Exception:
            from torch.backends.cuda import sdp_kernel

            ctx = sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            )

        with ctx:
            return torch.ops.vllm.torch_sdpa_wrapper(q, k, v, cu_seqlens)

    vitw.vit_xformers_attn_wrapper = _wrapped_vit_xformers_attn_wrapper


def apply_gems_patches_to_vllm(verbose=True):
    import vllm  # noqa: F401
    import vllm._custom_ops as ops  # noqa: F401
    from vllm.attention.ops import vit_attn_wrappers as vitw
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLAImpl
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
    patch_module_method(
        FlashAttnMLAImpl,
        "_forward_decode",
        custom_gems_flashattn_mla_forward_decode,
        verbose,
    )
    patch_vllm_lib("_C", "silu_and_mul", custom_silu_and_mul, "CUDA", verbose)
    patch_vllm_lib("_C", "cutlass_scaled_mm", custom_cutlass_scaled_mm, "CUDA", verbose)
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
    patch_vllm_lib("_moe_C", "grouped_topk", custom_moe_grouped_topk, "CUDA", verbose)
    patch_vllm_lib(
        "_C",
        "per_token_group_fp8_quant",
        custom_per_token_group_fp8_quant,
        "CUDA",
        verbose,
    )
    patch_vllm_lib(
        "_C",
        "apply_repetition_penalties_",
        custom_apply_repetition_penalties,
        "CUDA",
        verbose,
    )

    patch_vllm_lib(
        "_C_cache_ops",
        "concat_and_cache_mla",
        custom_concat_and_cache_mla,
        "CUDA",
        verbose,
    )
    patch_vllm_vit_to_attn(vitw)
