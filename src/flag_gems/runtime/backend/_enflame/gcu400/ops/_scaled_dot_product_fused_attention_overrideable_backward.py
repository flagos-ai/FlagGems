# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Optional

import torch
import triton

from .flash_attention_backward import (
    _disable_aabs_for_small_seqlen,
    _flash_attn_bwd_dkv,
    _flash_attn_bwd_dq_fused,
    _get_bias_strides_dense,
    _parse_philox,
    flash_attn_backward,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCU400 (L600/Libra) tuning extension for the dense backward kernels.
#
# The upstream autotune pools only carry num_warps>=4 tiles; on GCU400 the
# [64,128]-class dot tiles are markedly faster at 1-2 warps (measured
# 2026-11-10: dq 64x128/nw2 -38% on large shapes, dkv 128x64/nw1 -15%), and
# causal/non-causal optima differ (S2048/D128 dkv two-state gap 1.6x), so
# IS_CAUSAL must be part of the autotune cache key. Both changes are applied
# idempotently to the vendor autotuner objects at import time; kernel logic
# and numerics are config-invariant (gradient values do not depend on the
# tile decomposition).
# ---------------------------------------------------------------------------
_GCU400_DQ_EXTRA_CONFIGS = [
    (64, 128, 2, 2),
    (64, 128, 1, 2),
    (64, 128, 2, 3),
    (64, 64, 2, 2),
    (128, 128, 2, 2),
]

_GCU400_DKV_EXTRA_CONFIGS = [
    (128, 64, 1, 2),
    (128, 64, 2, 2),
    (128, 128, 2, 2),
    (64, 128, 1, 2),
    (64, 128, 2, 2),
    (64, 64, 1, 2),
]


def _extend_gcu400_autotune_configs():
    for autotuner, extras in (
        (_flash_attn_bwd_dq_fused, _GCU400_DQ_EXTRA_CONFIGS),
        (_flash_attn_bwd_dkv, _GCU400_DKV_EXTRA_CONFIGS),
    ):
        existing = {
            (
                config.kwargs.get("BLOCK_M"),
                config.kwargs.get("BLOCK_N"),
                config.num_warps,
                config.num_stages,
            )
            for config in autotuner.configs
        }
        for block_m, block_n, num_warps, num_stages in extras:
            if (block_m, block_n, num_warps, num_stages) in existing:
                continue
            autotuner.configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
            existing.add((block_m, block_n, num_warps, num_stages))
        keys = getattr(autotuner, "keys", None)
        if keys is not None and "IS_CAUSAL" not in keys:
            keys.append("IS_CAUSAL")


_extend_gcu400_autotune_configs()


# ---------------------------------------------------------------------------
# Launch retry: the host toolchain occasionally kills clang/ld.lld with a
# signal during autotune compile storms (see compiler_problems P02), which
# surfaces as an in-process exception. Successful compilations land in the
# on-disk triton cache, so retrying converges monotonically; the retry only
# ever runs on the exceptional path.
# ---------------------------------------------------------------------------
_LAUNCH_RETRIES = 4


def _launch_kernel_with_retry(launch, op_desc):
    for attempt in range(1, _LAUNCH_RETRIES + 1):
        try:
            return launch()
        except Exception as error:
            if attempt == _LAUNCH_RETRIES:
                raise
            logger.warning(
                "%s launch failed (attempt %d/%d): %s; retrying",
                op_desc,
                attempt,
                _LAUNCH_RETRIES,
                error,
            )


# ---------------------------------------------------------------------------
# Dense BHSD direct launch.
#
# The ATen forward fixes the dense layout to BHSD while the kernels address
# (batch, seq, head, dim) order through explicit strides, so the wrapper can
# hand the BHSD strides over in semantic order (s0, s2, s1, s3) instead of
# staging five permuted contiguous copies plus three permute-backs. Layout
# guards fail closed: anything unusual falls back to the original
# permute+contiguous path.
# ---------------------------------------------------------------------------
def _dense_bhsd_direct_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    attn_bias,
    is_causal,
    softmax_scale,
    bias_req_grad,
):
    Batch, H_q, SeqLen_q, Head_Dim = query.shape
    _, H_k, SeqLen_k, _ = key.shape
    if H_q % H_k != 0:
        return None
    if not (
        query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and out.is_contiguous()
        and grad_out.is_contiguous()
    ):
        return None
    if grad_out.shape != out.shape:
        return None
    if logsumexp.dim() != 3 or logsumexp.dtype != torch.float32:
        return None
    if tuple(logsumexp.shape) != (Batch, H_q, SeqLen_q):
        return None
    if not logsumexp.is_contiguous():
        return None
    has_bias = attn_bias is not None
    if has_bias:
        if attn_bias.dim() != 4:
            return None
        if bias_req_grad and not attn_bias.is_contiguous():
            return None

    group_size = H_q // H_k
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Head_Dim))
    wl = wr = SeqLen_q + SeqLen_k + 1

    L = logsumexp
    D = torch.empty_like(L)
    dQ = torch.empty_like(query)
    dK = torch.empty_like(key)
    dV = torch.empty_like(value)
    dBias: Optional[torch.Tensor] = None
    if bias_req_grad:
        dBias = torch.empty_like(attn_bias)

    bias_ptr = attn_bias if has_bias else query
    dbias_ptr = dBias if bias_req_grad else query
    alibi_ptr = query

    sb_b, sb_h, sb_m, sb_n = _get_bias_strides_dense(attn_bias if has_bias else None)

    dq_args = (
        query,
        key,
        value,
        grad_out,
        out,
        L,
        D,
        dQ,
        bias_ptr,
        dbias_ptr,
        alibi_ptr,
        scale,
        group_size,
        wl,
        wr,
        0.0,
        1.0,
        0,
        0,
        0,
        0,
        0,
        query.stride(0),
        query.stride(2),
        query.stride(1),
        query.stride(3),
        key.stride(0),
        key.stride(2),
        key.stride(1),
        key.stride(3),
        value.stride(0),
        value.stride(2),
        value.stride(1),
        value.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        L.stride(0),
        L.stride(1),
        D.stride(0),
        D.stride(1),
        sb_b,
        sb_h,
        sb_m,
        sb_n,
        SeqLen_q,
        SeqLen_k,
    )
    dq_kwargs = {
        "HEAD_DIM": Head_Dim,
        "IS_CAUSAL": is_causal,
        "IS_DROPOUT": False,
        "HAS_WINDOW": False,
        "HAS_ALIBI": False,
        "HAS_BIAS": has_bias,
        "DO_BIAS_GRAD": bias_req_grad,
    }

    dkv_args = (
        query,
        key,
        value,
        grad_out,
        L,
        D,
        dK,
        dV,
        bias_ptr,
        alibi_ptr,
        scale,
        group_size,
        wl,
        wr,
        0.0,
        1.0,
        0,
        0,
        0,
        0,
        0,
        query.stride(0),
        query.stride(2),
        query.stride(1),
        query.stride(3),
        key.stride(0),
        key.stride(2),
        key.stride(1),
        key.stride(3),
        value.stride(0),
        value.stride(2),
        value.stride(1),
        value.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        L.stride(0),
        L.stride(1),
        sb_b,
        sb_h,
        sb_m,
        sb_n,
        SeqLen_q,
        SeqLen_k,
    )
    dkv_kwargs = {
        "HEAD_DIM": Head_Dim,
        "IS_CAUSAL": is_causal,
        "IS_DROPOUT": False,
        "HAS_WINDOW": False,
        "HAS_ALIBI": False,
        "HAS_BIAS": has_bias,
    }

    with _disable_aabs_for_small_seqlen(SeqLen_q, SeqLen_k):
        grid_dq = lambda META: (triton.cdiv(SeqLen_q, META["BLOCK_M"]), Batch, H_q)
        _launch_kernel_with_retry(
            lambda: _flash_attn_bwd_dq_fused[grid_dq](*dq_args, **dq_kwargs),
            "flash_attn_bwd_dq_fused",
        )
        grid_dkv = lambda META: (triton.cdiv(SeqLen_k, META["BLOCK_N"]), Batch, H_k)
        _launch_kernel_with_retry(
            lambda: _flash_attn_bwd_dkv[grid_dkv](*dkv_args, **dkv_kwargs),
            "flash_attn_bwd_dkv",
        )

    return dQ, dK, dV, dBias


def scaled_dot_product_fused_attention_overrideable_backward(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    grad_input_mask,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    logger.debug(
        "GEMS_ENFLAME _scaled_dot_product_fused_attention_overrideable_backward"
    )

    need_dq, need_dk, need_dv, need_dbias = grad_input_mask

    is_dropout = dropout_p > 0.0
    rng_tuple = None
    if is_dropout:
        if philox_seed is None or philox_offset is None:
            raise ValueError(
                "philox_seed and philox_offset are required when dropout_p > 0"
            )
        rng_tuple = _parse_philox(philox_seed, philox_offset)

    use_varlen = (cum_seq_q is not None) and (cum_seq_k is not None)
    bias_req_grad = bool(need_dbias) and (attn_bias is not None)

    if use_varlen or is_dropout:
        # Varlen and dropout keep the vendor flash_attn_backward behavior
        # (varlen layouts are already Total-first; dropout needs the philox
        # plumbing through the shared host).
        dQ, dK, dV, dBias = flash_attn_backward(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp.float(),
            cu_seq_q=cum_seq_q,
            cu_seq_k=cum_seq_k,
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            is_dropout=is_dropout,
            dropout_p=dropout_p,
            rng_state=rng_tuple,
            is_causal=is_causal,
            softmax_scale=scale,
            attn_bias=attn_bias,
            bias_requires_grad=bias_req_grad,
        )
        if not need_dq:
            dQ = torch.zeros_like(query)
        if not need_dk:
            dK = torch.zeros_like(key)
        if not need_dv:
            dV = torch.zeros_like(value)
        if not bias_req_grad:
            dBias = None
        return dQ, dK, dV, dBias

    # Dense fast path: launch the kernels directly on the ATen BHSD layout
    # (strides handed over in semantic (s0, s2, s1, s3) order, no staging
    # copies). Falls back to the permute+contiguous path on any layout guard.
    result = _dense_bhsd_direct_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        attn_bias,
        is_causal,
        scale,
        bias_req_grad,
    )
    if result is not None:
        dQ, dK, dV, dBias = result
    else:
        grad_out_bshd = grad_out.permute(0, 2, 1, 3).contiguous()
        query_bshd = query.permute(0, 2, 1, 3).contiguous()
        key_bshd = key.permute(0, 2, 1, 3).contiguous()
        value_bshd = value.permute(0, 2, 1, 3).contiguous()
        out_bshd = out.permute(0, 2, 1, 3).contiguous()

        dQ_bshd, dK_bshd, dV_bshd, dBias = flash_attn_backward(
            grad_out_bshd,
            query_bshd,
            key_bshd,
            value_bshd,
            out_bshd,
            logsumexp.float(),
            cu_seq_q=None,
            cu_seq_k=None,
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            is_dropout=False,
            dropout_p=0.0,
            rng_state=None,
            is_causal=is_causal,
            softmax_scale=scale,
            attn_bias=attn_bias,
            bias_requires_grad=bias_req_grad,
        )

        dQ = dQ_bshd.permute(0, 2, 1, 3).contiguous()
        dK = dK_bshd.permute(0, 2, 1, 3).contiguous()
        dV = dV_bshd.permute(0, 2, 1, 3).contiguous()

    if not need_dq:
        dQ = torch.zeros_like(query)
    if not need_dk:
        dK = torch.zeros_like(key)
    if not need_dv:
        dV = torch.zeros_like(value)
    if not bias_req_grad:
        dBias = None

    return dQ, dK, dV, dBias


# ---------------------------------------------------------------------------
# The benchmark and test harnesses resolve the op through the
# ``flag_gems.ops`` namespace (gems_op entry), which still points at the
# generic module binding. Rebind it (idempotently) to this specialized
# implementation so both the gems_op entry and the ATen dispatch key (which
# resolves through the live flag_gems.<name> attribute) run the GCU400 path.
# ---------------------------------------------------------------------------
def _patch_gems_ops_entry():
    import flag_gems.ops as _gems_ops

    if (
        getattr(
            _gems_ops.scaled_dot_product_fused_attention_overrideable_backward,
            "__module__",
            "",
        )
        != __name__
    ):
        _gems_ops.scaled_dot_product_fused_attention_overrideable_backward = (
            scaled_dot_product_fused_attention_overrideable_backward
        )


try:
    _patch_gems_ops_entry()
except Exception as error:  # pragma: no cover - import-order edge cases
    logger.warning("gems ops entry rebind skipped: %s", error)
