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

# This file contains a guarded direct forward path for small chunk_gated_delta_rule
# shapes. It follows the recurrent definition directly and falls back to the
# chunk decomposition for unsupported cases.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.utils import libentry

_DIRECT_MAX_T = 128
_DIRECT_MAX_K = 128
_DIRECT_MAX_V = 128
_DIRECT_BV = 32


@libentry()
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "STORE_FINAL_STATE": lambda args: args["final_state"] is not None,
    }
)
@triton.jit
def _chunk_gated_delta_rule_direct_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    initial_state,
    final_state,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    i_hg = i_h // (H // Hg)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = (
            initial_state + ((i_b * H + i_h) * K * V) + o_k[:, None] * V + o_v[None, :]
        )
        b_h += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    q_base = q + ((i_b * T * Hg + i_hg) * K)
    k_base = k + ((i_b * T * Hg + i_hg) * K)
    v_base = v + ((i_b * T * H + i_h) * V)
    o_base = o + ((i_b * T * H + i_h) * V)
    g_base = g + i_b * T * H + i_h
    beta_base = beta + i_b * T * H + i_h
    for i_t in range(0, T):
        b_q = tl.load(q_base + i_t * Hg * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        b_k = tl.load(k_base + i_t * Hg * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.maximum(tl.sqrt(tl.sum(b_q * b_q)), 1e-6)
            b_k = b_k / tl.maximum(tl.sqrt(tl.sum(b_k * b_k)), 1e-6)
        b_v = tl.load(v_base + i_t * H * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(g_base + i_t * H).to(tl.float32)
        b_beta = tl.load(beta_base + i_t * H).to(tl.float32)

        b_h *= exp(b_g)
        b_v = (b_v - tl.sum(b_h * b_k[:, None], axis=0)) * b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * (b_q * scale)[:, None], axis=0)
        tl.store(
            o_base + i_t * H * V + o_v,
            b_o.to(o.dtype.element_ty),
            mask=mask_v,
        )

    if STORE_FINAL_STATE:
        p_ht = final_state + ((i_b * H + i_h) * K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def can_use_chunk_gated_delta_rule_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    if cu_seqlens is not None:
        return False
    if initial_state is not None:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    if not (g.is_contiguous() and beta.is_contiguous()):
        return False
    B, T, Hg, K = q.shape
    Bv, Tv, H, V = v.shape
    return (
        B == Bv
        and T == Tv
        and 0 < T <= _DIRECT_MAX_T
        and 0 < K <= _DIRECT_MAX_K
        and 0 < V <= _DIRECT_MAX_V
        and H % Hg == 0
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def chunk_gated_delta_rule_direct_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    BK = triton.next_power_of_2(K)
    use_one_warp = (K <= 16 and V <= 16) or (
        q.dtype == torch.float32 and K <= 32 and V <= 32
    )
    BV = min(triton.next_power_of_2(V), 16 if use_one_warp else _DIRECT_BV)

    o = torch.empty_like(v)
    final_state = (
        torch.empty(B, H, K, V, device=v.device, dtype=torch.float32)
        if output_final_state
        else None
    )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), B * H)

    _chunk_gated_delta_rule_direct_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        initial_state=initial_state,
        final_state=final_state,
        scale=float(scale),
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1 if use_one_warp else (4 if K >= 128 else 2),
        num_stages=1 if K <= 16 and V <= 16 else (2 if use_one_warp else 3),
    )
    return o, final_state
