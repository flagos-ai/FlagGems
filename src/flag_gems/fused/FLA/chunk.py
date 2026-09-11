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


# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import logging

import torch

from flag_gems.fused.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_gems.fused.FLA.chunk_fused_tail_vblock import (
    can_use_fused_tail_vblock,
    chunk_gated_delta_rule_fused_tail_vblock,
)
from flag_gems.fused.FLA.chunk_o import chunk_fwd_o
from flag_gems.fused.FLA.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flag_gems.fused.FLA.utils import SUPPRESS_LEVEL
from flag_gems.fused.FLA.wy_fast import recompute_w_u_fwd

logger = logging.getLogger(__name__)


def _chunk_size_for_sequence(T: int, is_varlen: bool) -> int:
    if is_varlen:
        return 64
    return min(64, max(16, 1 << (T - 1).bit_length()))


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    logger.debug("GEMS CHUNK GATED DELTA RULE FWD")
    q_contiguous = q.is_contiguous()
    k_contiguous = k.is_contiguous()
    v_contiguous = v.is_contiguous()
    g_contiguous = g.is_contiguous()
    beta_contiguous = beta.is_contiguous()
    initial_state_contiguous = initial_state is None or initial_state.is_contiguous()
    cu_seqlens_contiguous = cu_seqlens is None or cu_seqlens.is_contiguous()
    if not (
        q_contiguous
        and k_contiguous
        and v_contiguous
        and g_contiguous
        and beta_contiguous
        and initial_state_contiguous
        and cu_seqlens_contiguous
    ):
        if not q_contiguous:
            q = q.contiguous()
        if not k_contiguous:
            k = k.contiguous()
        if not v_contiguous:
            v = v.contiguous()
        if not g_contiguous:
            g = g.contiguous()
        if not beta_contiguous:
            beta = beta.contiguous()
        if not initial_state_contiguous:
            initial_state = initial_state.contiguous()
        if not cu_seqlens_contiguous:
            cu_seqlens = cu_seqlens.contiguous()

    chunk_size = _chunk_size_for_sequence(q.shape[1], cu_seqlens is not None)

    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3 and can_use_fused_tail_vblock(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    ):
        o, final_state = chunk_gated_delta_rule_fused_tail_vblock(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            scale=scale,
        )
        return g, o, A, final_state, None, None, None
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


def naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state):
    """
    Naive reference implementation of chunk_gated_delta_rule_fwd.
    Implements the gated delta rule recurrence token-by-token:
        S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})
        o_t = q_t @ S_t * scale
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    S = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    )
    outputs = []

    for t in range(T):
        q_t = q[:, t, :, :]  # (B, H, K)
        k_t = k[:, t, :, :]  # (B, H, K)
        v_t = v[:, t, :, :]  # (B, H, V)
        g_t = g[:, t, :]  # (B, H)
        beta_t = beta[:, t, :]  # (B, H)

        # Gating: S = exp(g_t) * S
        gate = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        S = gate * S

        # Delta rule: S += beta_t * k_t^T @ (v_t - k_t @ S)
        # k_t: (B, H, K) -> (B, H, K, 1)
        # v_t: (B, H, V) -> (B, H, 1, V)
        kS = torch.einsum("bhk,bhkv->bhv", k_t, S)  # (B, H, V)
        delta = v_t - kS  # (B, H, V)
        # outer product: k_t^T @ delta -> (B, H, K, V)
        update = torch.einsum("bhk,bhv->bhkv", k_t, delta) * beta_t.unsqueeze(
            -1
        ).unsqueeze(-1)
        S = S + update

        # Output: o_t = q_t @ S * scale
        o_t = torch.einsum("bhk,bhkv->bhv", q_t, S) * scale  # (B, H, V)
        outputs.append(o_t)

    o = torch.stack(outputs, dim=1)  # (B, T, H, V)
    return o, S
