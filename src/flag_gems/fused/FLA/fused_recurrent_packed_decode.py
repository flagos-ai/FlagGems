# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Triton kernel for fused_recurrent_gated_delta_rule packed decode.

Hybrid dispatch with two dedicated kernels optimized for the decode phase
(one token per sequence, packed mixed_qkv input).

Tier breakdown:
- B <= 4: _kernel_vtile_loop, BV=32, NV=4, num_warps=4, num_stages=4
- B > 4:  _kernel_per_vtile,  BV=32, NV=4, num_warps=1, num_stages=4

Optimizations:
1. Two dedicated kernels (instead of constexpr dispatch) for better
   compiler specialization.
2. V-tile loop for small B eliminates redundant Q/K/gating loads across
   V-tiles that would otherwise be repeated across NV separate blocks.
3. State h loaded FIRST in per-V-tile path to overlap 16KB global memory
   latency with subsequent register-only gating computation.
4. num_stages=4 for deeper software pipelining: pipelines all 4 state
   tile loads in the V-tile loop; better overlaps 8KB state loads across
   concurrent blocks in the per-V-tile path.
5. State-base-address precomputation reuses address arithmetic for both
   load and store.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _kernel_vtile_loop(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """V-tile loop kernel for small B (<=4). Single block per (token, head)
    processes all NV V-tiles sequentially, eliminating redundant Q/K/gating
    loads that would otherwise be repeated across NV separate blocks."""

    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    p_idx = ssm_state_indices + i_n * stride_indices_seq
    state_idx = tl.load(p_idx).to(tl.int64)

    if state_idx < 0:
        p_o = o + (i_n * HV + i_hv) * V
        for i_v in range(NV):
            o_v = i_v * BV + tl.arange(0, BV)
            mask_v = o_v < V
            zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
            tl.store(p_o + o_v, zero, mask=mask_v)
        return

    # Load Q, K once (reused across all V-tiles)
    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # Load gating params once (reused across all V-tiles)
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    decay = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val)

    # Precompute loop-invariant base addresses
    base_h = state_idx * stride_init_state_token + i_hv * V * K
    base_v = (2 * H * K) + i_hv * V
    base_o = (i_n * HV + i_hv) * V

    for i_v in range(NV):
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        # Load state h tile [BV, BK]
        p_h0 = h0 + base_h + o_v[:, None] * K + o_k[None, :]
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        # Load V tile [BV]
        p_v = p_mixed + base_v + o_v
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        # Delta rule update
        b_h *= decay
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)

        tl.store(o + base_o + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
        tl.store(
            ht + base_h + o_v[:, None] * K + o_k[None, :],
            b_h.to(ht.dtype.element_ty),
            mask=mask_h,
        )


@triton.jit
def _kernel_per_vtile(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """Per-V-tile kernel for B > 4. Each block processes one V-tile of one
    (token, head) pair. Grid=(NV, B*HV) provides 4x more blocks than the
    V-tile-loop approach, maximizing SM occupancy for larger batch sizes.

    State h is loaded FIRST to overlap its 16KB global memory latency with
    subsequent register-only gating computation (softplus, exp, sigmoid).
    """

    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_idx = ssm_state_indices + i_n * stride_indices_seq
    state_idx = tl.load(p_idx).to(tl.int64)

    if state_idx < 0:
        p_o = o + (i_n * HV + i_hv) * V + o_v
        zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    # Precompute state base address for reuse in load and store
    state_base = state_idx * stride_init_state_token + i_hv * V * K

    # Load state h FIRST: 16KB global load with longest latency.
    # Issued before Q/K/V/gating so the compiler can overlap this with
    # register-only arithmetic below.
    p_h = h0 + state_base + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    # Load Q, K, V
    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # Gating computation (register-only)
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    decay = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val)

    # Delta rule update
    b_h *= decay
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # Store output
    p_o = o + (i_n * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)

    # Store state (reuse precomputed state_base)
    tl.store(
        ht + state_base + o_v[:, None] * K + o_k[None, :],
        b_h.to(ht.dtype.element_ty),
        mask=mask_h,
    )


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple:
    """Fused packed decode for gated delta rule (GDN) linear attention.

    Takes a single packed mixed_qkv tensor [B, qkv_dim] and computes the
    recurrent gated delta rule update in a single Triton kernel.

    Args:
        mixed_qkv: [B, 2*H*K + HV*V] packed Q, K, V projections.
        a: [B, HV] input-dependent gating A.
        b: [B, HV] input-dependent gating B (sigmoid-gated beta).
        A_log: [HV] log-space decay parameter (shared across batch).
        dt_bias: [HV] time-step bias (shared across batch).
        scale: attention scale factor (typically head_k_dim ** -0.5).
        initial_state: [num_slots, HV, V, K] full state pool.
        out: [B, 1, HV, V] output tensor (written in-place).
        ssm_state_indices: [B] per-request state slot indices.
        use_qk_l2norm_in_kernel: whether to L2-normalize Q/K in kernel.

    Returns:
        (out, initial_state) tuple. `out` and `initial_state` are modified
        in-place.
    """
    logger.debug("GEMS FUSED_RECURRENT_GATED_DELTA_RULE_PACKED_DECODE")
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K

    BK = triton.next_power_of_2(K)

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    num_stages = 4

    # Two-tier dispatch:
    # B <= 4: V-tile loop (1 block per (token,head), num_warps=4)
    #   Eliminates redundant Q/K/gating loads across V-tiles.
    #   Max ~64 blocks total (few enough that SMs aren't the bottleneck).
    # B > 4:  Per-V-tile grid (4 blocks per (token,head), num_warps=1)
    #   4x more blocks for SM occupancy at scale.
    #   Redundant Q/K/gating loads negligible vs state I/O.
    if B <= 4:
        grid = (B * HV,)
        num_warps = 4
        _kernel_vtile_loop[grid](
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            o=out,
            h0=initial_state,
            ht=initial_state,
            ssm_state_indices=ssm_state_indices,
            scale=scale,
            stride_mixed_qkv_tok=stride_mixed_qkv_tok,
            stride_a_tok=stride_a_tok,
            stride_b_tok=stride_b_tok,
            stride_init_state_token=stride_init_state_token,
            stride_final_state_token=stride_final_state_token,
            stride_indices_seq=stride_indices_seq,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            NV=NV,
            SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (NV, B * HV)
        num_warps = 1
        _kernel_per_vtile[grid](
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            o=out,
            h0=initial_state,
            ht=initial_state,
            ssm_state_indices=ssm_state_indices,
            scale=scale,
            stride_mixed_qkv_tok=stride_mixed_qkv_tok,
            stride_a_tok=stride_a_tok,
            stride_b_tok=stride_b_tok,
            stride_init_state_token=stride_init_state_token,
            stride_final_state_token=stride_final_state_token,
            stride_indices_seq=stride_indices_seq,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out, initial_state
