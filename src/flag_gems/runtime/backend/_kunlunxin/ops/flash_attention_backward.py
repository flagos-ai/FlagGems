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
# ============================================================================
# [fab-c171 2026-09-19] Flash-attention BACKWARD family via the XPU
# launch-table binding. Dense (B, S, H, D) contiguous inputs are dispatched
# to xfa mha_varlen_bwd by the C handler (third_party/xpu/device/xpu3/
# launch_extra.cpp: handle_fa_bwd). The carrier kernel below is never actually
# executed: the launch table intercepts it by name. If the handler honestly
# falls back (INT_MIN) the carrier body poisons dQ with NaN so a missed
# binding can never look like a valid result.
# The reference side (test-time calls outside use_gems) is served by the
# backend monkey_patch CPU reference; nothing here changes that.
# ============================================================================

import logging

import torch
import triton
import triton.language as tl

from .contiguous import contiguous
from .to import to_copy

logger = logging.getLogger(__name__)

__all__ = ["flash_attention_backward"]


@triton.jit
def _fab_bwd_carrier(
    DOUT,
    Q,
    K,
    V,
    OUT,
    LSE,
    DQ,
    DK,
    DV,
    SCALE,
    BATCH: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_NUM_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WIN_LEFT: tl.constexpr,
    WIN_RIGHT: tl.constexpr,
):
    # Poison-only fallback body (see file-header note).
    if tl.program_id(0) == 0:
        tl.store(
            DQ + tl.arange(0, 1),
            tl.full((1,), float("nan"), DQ.dtype.element_ty),
        )


def _fab_bwd_launch(
    grad_out,
    query,
    key,
    value,
    out,
    lse,
    sm_scale,
    is_causal,
    window_size_left,
    window_size_right,
):
    """Launch the bound backward carrier on (B, S, H, D) contiguous inputs."""
    batch, q_len, q_heads, head_dim = query.shape
    kv_len = key.shape[1]
    kv_heads = key.shape[2]
    dq = torch.empty_like(query)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)
    _fab_bwd_carrier[(1,)](
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        dq,
        dk,
        dv,
        float(sm_scale),
        BATCH=batch,
        Q_CTX=q_len,
        KV_CTX=kv_len,
        HEAD_NUM=q_heads,
        HEAD_NUM_K=kv_heads,
        HEAD_DIM=head_dim,
        IS_CAUSAL=bool(is_causal),
        WIN_LEFT=int(window_size_left),
        WIN_RIGHT=int(window_size_right),
    )
    return dq, dk, dv


def flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    unused,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
):
    """aten::_flash_attention_backward on (B, S, H, D) dense inputs.

    Dispatches to xfa mha_varlen_bwd through the launch-table binding;
    varlen (cum_seq) and dropout are not part of the binding's contract.
    """
    logger.debug("GEMS_KUNLUNXIN FLASH_ATTENTION_BACKWARD")
    if cum_seq_q is not None or cum_seq_k is not None:
        raise NotImplementedError(
            "kunlunxin flash_attention_backward binding supports dense inputs only"
        )
    if dropout_p and float(dropout_p) > 0.0:
        raise NotImplementedError(
            "kunlunxin flash_attention_backward binding requires dropout_p=0"
        )
    if scale is None:
        head_dim = query.shape[-1]
        scale = 1.0 / (head_dim**0.5)
    wl = -1 if window_size_left is None else int(window_size_left)
    wr = -1 if window_size_right is None else int(window_size_right)
    # The upstream logsumexp is a non-contiguous [B, H, S] view of a [B, S, H]
    # ordered buffer; the bound kernel reads it as dense [B, H, S]. Materialize
    # it before launching (same convention as the efficient-attention family).
    lse = contiguous(logsumexp[:, :, : query.shape[1]])
    if lse.dtype != torch.float32:
        lse = to_copy(lse, dtype=torch.float32)
    dq, dk, dv = _fab_bwd_launch(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        scale,
        is_causal,
        wl,
        wr,
    )
    return dq, dk, dv
