# Modified by Songlin Yang & Ali Hatamizadeh

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from lit_gpt.config import Config
from torch import Tensor
from typing_extensions import Self
from xformers.ops import SwiGLU

from .fused_rotary_embedding import apply_rotary_emb_func
from .gated_delta_net import GatedDeltaNet

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
# import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = MBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        factory_kwargs = {"device": "cuda", "dtype": torch.float32}
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        if config.mamba:
            if self.config.fused_add_norm:
                if layer_norm_fn is None or rms_norm_fn is None:
                    raise ImportError(
                        "Failed to import Triton LayerNorm / RMSNorm kernels"
                    )

            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                    h=nn.ModuleList(
                        create_block(
                            config.n_embd,
                            ssm_cfg=None,
                            norm_epsilon=config.norm_eps,
                            rms_norm=config.rms_norm,
                            residual_in_fp32=config.residual_in_fp32,
                            fused_add_norm=config.fused_add_norm,
                            layer_idx=i,
                            **factory_kwargs,
                        )
                        for i in range(config.n_layer)
                    ),
                    ln_f=(nn.LayerNorm if not config.rms_norm else RMSNorm)(
                        config.n_embd,
                        eps=config.norm_eps,
                        **factory_kwargs,
                    ),
                )
            )

        else:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                    h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                    ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
                )
            )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.max_len = self.config.block_size
        self.mamba_init = config.mamba or config.mamba_init
        if self.mamba_init:
            self.tie_weights()

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
                )
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
            else:
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd)
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (
                name in ["out_proj.weight", "fc2.weight"]
                or (name == "proj.weight" and isinstance(module, LLaMAMLP))
                or (
                    name == "w3.weight"
                    and isinstance(module, SwiGLU)
                    or (
                        name == "proj.weight"
                        and isinstance(module, CausalSelfAttention)
                    )
                )
            ):  # if use xformer swiglu, fc2 layer will be renamed to w3
                if self.mamba_init:
                    n_residuals_per_layer = (
                        1 if self.config.mamba or not self.config.mlp else 2
                    )
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                else:
                    nn.init.normal_(
                        p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer
                    )

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.mamba:
            hidden_states = self.transformer.wte(idx)
            residual = None
            for block in self.transformer.h:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=None
                )
            norm_f = self.transformer.ln_f
            if not self.config.fused_add_norm:
                residual = (
                    (hidden_states + residual)
                    if residual is not None
                    else hidden_states
                )
                hidden_states = norm_f(residual.to(dtype=norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = (
                    rms_norm_fn if isinstance(norm_f, RMSNorm) else layer_norm_fn
                )
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    norm_f.weight,
                    norm_f.bias,
                    eps=norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.config.residual_in_fp32,
                )
            return self.lm_head(hidden_states)

        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        # assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        # assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"
        if not self.config.nope:
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            elif T > self.max_len:
                self.max_len = T
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if use_kv_cache:
            if not self.config.nope:
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            if not self.config.nope:
                cos = cos[:T]
                sin = sin[:T]
            mask = None
        if self.config.nope:
            rope = None
        else:
            rope = (cos, sin)
        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, rope, max_seq_length)
        else:
            if self.config.nope:
                self.kv_caches = self.kv_caches or self.build_kv_caches(
                    x, max_seq_length, None
                )
            else:
                self.kv_caches = self.kv_caches or self.build_kv_caches(
                    x, max_seq_length, cos.size(-1) * 2
                )
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, max_seq_length, mask, input_pos, self.kv_caches[i]
                )

        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int
    ) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups
        if rope_cache_length is not None:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                rope_cache_length
                + self.config.head_size
                - int(self.config.rotary_percentage * self.config.head_size),
            )
        else:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                self.config.head_size,
            )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (
                torch.zeros(k_cache_shape, device=device),
                torch.zeros(v_cache_shape, device=device),
            )
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.use_gated_deltanet = (
            layer_idx % config.gated_delta_per_layer == 0
            if config.gated_delta_per_layer > 0
            else False
        )
        if self.use_gated_deltanet:
            self.attn = GatedDeltaNet(hidden_size=config.n_embd)
        else:
            self.attn = CausalSelfAttention(
                config,
                n_embd=config.n_embd,
                layer_idx=layer_idx,
            )
        if (
            not config.shared_attention_norm
            and config.mlp
            and not config.parallel_residual
        ):
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = config.mlp_class(
                config,
            )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)

        if self.use_gated_deltanet:
            h, _, new_kv_cache = self.attn(n_1, attention_mask=mask)
        else:
            h, new_kv_cache = self.attn(
                n_1, rope, max_seq_length, mask, input_pos, kv_cache
            )
        if self.config.parallel_residual:
            assert self.config.shared_attention_norm
            if self.config.mlp:
                h = h + self.mlp(n_1)
            x = x + h
        else:
            x = x + h
            if self.config.mlp:
                n_2 = self.norm_2(x)
                h = self.mlp(n_2)
                x = x + h
        return x, new_kv_cache


class MBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


class CausalSelfAttention(nn.Module):
    def __init__(
        self, config: Config, layer_idx: int, n_embd: int, head_size=None
    ) -> None:
        super().__init__()
        self.local = layer_idx % config.full_per_layer < config.full_per_layer - 1
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.n_query_groups, total_qkv, self.head_size
        )  # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)
        if self.sc:
            q = causal_conv1d_fn(
                x=q.transpose(-1, -2),
                weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                bias=self.q_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            k = causal_conv1d_fn(
                x=k.transpose(-1, -2),
                weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                bias=self.k_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            v = causal_conv1d_fn(
                x=v.transpose(-1, -2),
                weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                bias=self.v_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)

        q = q.reshape(B, T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.head_size)
        v = v.reshape(B, T, -1, self.head_size)

        if not self.config.nope:
            cos, sin = rope
            # apply rope in fp32 significanly stabalize training
            # fused rope expect (batch_size, seqlen, nheads, headdim)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y, kv_cache

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        scale = 1.0 / math.sqrt(self.head_size)

        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            if self.local and self.config.local_window > -1:
                win_tuple = (self.config.local_window - 1, 0)
            else:
                win_tuple = (-1, -1)
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
                window_size=win_tuple,
            )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class LLaMAMLP(nn.Module):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            _pack_weights=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swiglu(x)
        return x


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin
