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
    )

    if self.rotary_dim < self.head_size:
        query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
    else:
        query = q_embed.reshape(query_shape)
        key = k_embed.reshape(key_shape)

    return query, key


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


def apply_gems_patches_to_vllm(verbose=True):
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

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
