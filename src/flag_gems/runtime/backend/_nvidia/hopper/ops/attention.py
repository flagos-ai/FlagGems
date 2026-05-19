# src/flag_gems/runtime/backend/_nvidia/hopper/ops/attention.py
"""
Hopper-specific overrides for flash-attention varlen entry points.
Loaded by BackendArchEvent.get_arch_ops() on sm_90+; replaces the generic
flash_attn_varlen_func / flash_attn_varlen_opt_func in flag_gems globals.
"""
import logging

import torch

from flag_gems.config import use_c_extension
from flag_gems.ops.flash_api import mha_varlan_fwd, mha_varlan_fwd_opt
from flag_gems.runtime import torch_device_fn

from .flash_api_v3 import is_fa3_supported, mha_varlan_fwd_v3

logger = logging.getLogger(__name__)


def _maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits=0,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    fa_version=3,
):
    if fa_version not in (2, 3):
        raise RuntimeError(f"Unsupported fa_version={fa_version}")
    if fa_version == 3 and not is_fa3_supported():
        logger.warning("FA3 requested but unavailable; falling back to FA2.")
        fa_version = 2
    if num_splits > 0:
        raise RuntimeError("num_splits > 0 not implemented.")
    if use_c_extension:
        from flag_gems.ops.attention import flash_attn_varlen_func as _generic

        return _generic(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k,
            seqused_k,
            q_v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            block_table,
            return_softmax_lse,
            out,
            scheduler_metadata,
            q_descale,
            k_descale,
            v_descale,
            s_aux,
            num_splits,
            cp_world_size,
            cp_rank,
            cp_tot_seqused_k,
            fa_version,
        )

    assert cu_seqlens_k is not None or seqused_k is not None
    assert cu_seqlens_k is None or seqused_k is None
    assert block_table is None or seqused_k is not None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    real_window_size = (
        (-1, -1) if window_size is None else (window_size[0], window_size[1])
    )

    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    max_seqlen_q = (
        max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
    )
    max_seqlen_k = (
        max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k
    )

    _launcher = mha_varlan_fwd_v3 if fa_version == 3 else mha_varlan_fwd
    # logger.info("HOPPER_OVERRIDE flash_attn_varlen_func entered (fa_version=%d)", fa_version)
    out, q, k, v, softmax_lse, *_ = _launcher(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        real_window_size[0],
        real_window_size[1],
        softcap,
        return_softmax_lse and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def flash_attn_varlen_opt_func(*args, **kwargs):
    # 对应原 attention.py 中 _opt 路径：底层 launcher 换成 mha_varlan_fwd_opt
    # 当 fa_version=3 时仍调用 mha_varlan_fwd_v3。
    ...
