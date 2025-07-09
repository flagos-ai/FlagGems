from typing import Optional, Tuple

import torch

from flag_gems.patches.patch_util import patch_module_method


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


def apply_gems_patches_to_vllm(verbose=True):
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
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
