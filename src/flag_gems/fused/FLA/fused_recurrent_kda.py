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
#
# This implementation is adapted from the flash-linear-attention recurrent KDA
# kernel, originally licensed under MIT.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Triton recurrent KDA kernels for one-token serving decode."""

import torch
import triton
import triton.language as tl

from flag_gems.fused.FLA.triton_ops_helper import exp


@triton.jit
def fused_recurrent_kda_decode_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    state,
    cu_seqlens,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_g_head: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_beta_head: tl.constexpr,
    stride_beta_value: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_value: tl.constexpr,
    stride_state_key: tl.constexpr,
    stride_cu_seqlens: tl.constexpr,
    stride_state_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GROUP_V: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
):
    """Update a V-first state cache for one token from every active sequence."""
    i_v_group, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    if USE_CU_SEQLENS:
        bos = tl.load(cu_seqlens + i_n * stride_cu_seqlens).to(tl.int64)
        eos = tl.load(cu_seqlens + (i_n + 1) * stride_cu_seqlens).to(tl.int64)
        sequence_length = eos - bos
        if sequence_length == 0:
            # Empty sequences are CUDA Graph padding and have no packed output.
            return
        if sequence_length != 1:
            # This kernel is deliberately limited to recurrent decode.  Avoid
            # mutating state when it is accidentally handed a prefill sequence.
            return
        i_token = bos
    else:
        i_token = i_n

    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)

    # Serving reserves slot zero for graph padding. It must never be read or
    # written, and the corresponding output is defined as zero.
    if state_idx <= 0:
        for i_tile in tl.range(0, GROUP_V, loop_unroll_factor=1):
            o_v = (i_v_group * GROUP_V + i_tile) * BV + tl.arange(0, BV)
            mask_v = o_v < V
            p_o = o + i_token * stride_o_token + i_hv * stride_o_head + o_v
            tl.store(p_o, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
        return

    p_q = q + i_token * stride_q_token + i_h * stride_q_head + o_k
    p_k = k + i_token * stride_k_token + i_h * stride_k_head + o_k
    b_q = tl.load(p_q, mask=mask_k, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    b_k = tl.load(p_k, mask=mask_k, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    if USE_QK_L2NORM_IN_KERNEL:
        b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = g + i_token * stride_g_token + i_hv * stride_g_head + o_k
    b_g = tl.load(p_g, mask=mask_k, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    if USE_GATE_IN_KERNEL:
        if HAS_BIAS:
            b_g += tl.load(
                dt_bias + i_hv * K + o_k,
                mask=mask_k,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
        b_a = exp(tl.load(A_log + i_hv).to(tl.float32)) if HAS_A else 1.0
        if USE_LOWER_BOUND:
            b_g = lower_bound * tl.sigmoid(b_a * b_g)
        else:
            b_softplus = tl.where(b_g > 20.0, b_g, tl.log(1.0 + tl.exp(b_g)))
            b_g = -b_a * b_softplus

    b_decay = exp(b_g)
    if not IS_BETA_HEADWISE:
        p_beta = beta + i_token * stride_beta_token + i_hv * stride_beta_head
        b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta *= 2.0

    # Serialize a small group of V tiles in one program.  This keeps each state
    # tile small enough for a single warp while reusing q/k/g and their norms.
    for i_tile in tl.range(0, GROUP_V, loop_unroll_factor=1):
        o_v = (i_v_group * GROUP_V + i_tile) * BV + tl.arange(0, BV)
        mask_v = o_v < V
        mask_state = mask_v[:, None] & mask_k[None, :]

        p_state = (
            state
            + state_idx * stride_state_token
            + i_hv * stride_state_head
            + o_v[:, None] * stride_state_value
            + o_k[None, :] * stride_state_key
        )
        b_state = tl.load(
            p_state,
            mask=mask_state,
            other=0.0,
        ).to(tl.float32)
        p_v = v + i_token * stride_v_token + i_hv * stride_v_head + o_v
        b_v = tl.load(p_v, mask=mask_v, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )

        if IS_BETA_HEADWISE:
            p_beta = (
                beta
                + i_token * stride_beta_token
                + i_hv * stride_beta_head
                + o_v * stride_beta_value
            )
            b_beta = tl.load(
                p_beta, mask=mask_v, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            if APPLY_BETA_SIGMOID:
                b_beta = tl.sigmoid(b_beta)
                if ALLOW_NEG_EIGVAL:
                    b_beta *= 2.0

        b_state *= b_decay[None, :]
        b_v -= tl.sum(b_state * b_k[None, :], axis=1)
        b_v *= b_beta
        b_state += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_state * b_q[None, :], axis=1)

        p_o = o + i_token * stride_o_token + i_hv * stride_o_head + o_v
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
            eviction_policy="evict_first",
        )
        tl.store(
            p_state,
            b_state.to(p_state.dtype.element_ty),
            mask=mask_state,
            eviction_policy="evict_first",
        )


def _validate_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    out: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError("`q` must have shape [1, N, H, K].")
    B, N, H, K = q.shape
    if B != 1:
        raise ValueError("The decode fast path requires a flattened batch (B=1).")
    if k.shape != q.shape:
        raise ValueError("`k` must have the same shape as `q`.")
    if q.dtype != k.dtype:
        raise ValueError("`q` and `k` must have the same dtype.")
    if v.ndim != 4 or v.shape[:2] != (B, N):
        raise ValueError("`v` must have shape [1, N, HV, V].")
    if v.dtype != q.dtype:
        raise ValueError("`q`, `k`, and `v` must have the same dtype.")
    HV, V = v.shape[2:]
    if H <= 0 or HV < H or HV % H != 0:
        raise ValueError("`HV` must be a positive multiple of `H`.")
    if g.shape != (B, N, HV, K):
        raise ValueError(f"`g` must have shape {(B, N, HV, K)}.")
    if not use_gate_in_kernel and g.dtype != torch.float32:
        raise ValueError("Preprocessed KDA gate values must have float32 dtype.")
    if beta.shape not in ((B, N, HV), (B, N, HV, V)):
        raise ValueError(f"`beta` must have shape {(B, N, HV)} or {(B, N, HV, V)}.")
    if not use_beta_sigmoid_in_kernel and beta.dtype != torch.float32:
        raise ValueError("Preprocessed KDA beta values must have float32 dtype.")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, V, K):
        raise ValueError(
            "`initial_state` must use V-first layout [num_slots, HV, V, K]."
        )
    if initial_state.dtype != torch.float32:
        raise ValueError("`initial_state` must be float32 for serving KDA.")
    if initial_state.stride()[1:] != (V * K, K, 1):
        raise ValueError("`initial_state` must be contiguous inside each cache slot.")
    if initial_state.stride(0) < HV * V * K:
        raise ValueError("`initial_state` cache slots must not overlap.")
    if ssm_state_indices.ndim != 1:
        raise ValueError("`ssm_state_indices` must be one-dimensional.")
    if ssm_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("`ssm_state_indices` must have int32 or int64 dtype.")
    if any(x.stride(-1) != 1 for x in (q, k, v, g, beta)):
        raise ValueError("All inputs must be contiguous in their last dimension.")

    num_sequences = N
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("`cu_seqlens` must have shape [num_sequences + 1].")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("`cu_seqlens` must have int32 or int64 dtype.")
        if cu_seqlens.stride(0) != 1:
            raise ValueError("`cu_seqlens` must be contiguous.")
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences < N:
            raise ValueError(
                "The decode fast path requires at least one sequence per packed token."
            )
        if ssm_state_indices.numel() != num_sequences:
            raise ValueError(
                "`ssm_state_indices` must contain one slot for every sequence."
            )
    elif ssm_state_indices.numel() != N:
        raise ValueError("`ssm_state_indices` must contain one slot for every token.")

    tensors = (q, k, v, g, beta, initial_state, ssm_state_indices)
    if cu_seqlens is not None:
        tensors += (cu_seqlens,)
    if any(x.device != q.device for x in tensors):
        raise ValueError("All inputs must be on the same device.")
    if out is not None:
        if out.shape != v.shape or out.device != q.device:
            raise ValueError("`out` must match `v` in shape and device.")
        if out.dtype != v.dtype:
            raise ValueError("`out` must have the same dtype as `v`.")
        if out.stride(-1) != 1:
            raise ValueError("`out` must be contiguous in its last dimension.")

    if use_gate_in_kernel:
        if A_log is None and lower_bound is None:
            raise ValueError(
                "`A_log` is required for the unbounded in-kernel KDA gate."
            )
        if A_log is not None:
            if A_log.numel() != HV or not A_log.is_contiguous():
                raise ValueError("`A_log` must be contiguous with one value per head.")
            if A_log.device != q.device or A_log.dtype != torch.float32:
                raise ValueError(
                    "`A_log` must be a float32 tensor on the input device."
                )
        if dt_bias is not None:
            if dt_bias.numel() != HV * K or not dt_bias.is_contiguous():
                raise ValueError("`dt_bias` must be contiguous with shape [HV, K].")
            if dt_bias.device != q.device or dt_bias.dtype != torch.float32:
                raise ValueError("`dt_bias` must be float32 on the input device.")
        if lower_bound is not None and not (-5.0 <= lower_bound < 0.0):
            raise ValueError("`lower_bound` must be in the safe range [-5, 0).")
    elif any(x is not None for x in (A_log, dt_bias, lower_bound)):
        raise ValueError("Gate parameters require `use_gate_in_kernel=True`.")

    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`."
        )
    return num_sequences, H, HV, K


def fused_recurrent_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-token-per-sequence KDA decode and update the state in place.

    Active state indices must be unique and greater than zero. Index zero is
    reserved for CUDA Graph padding and produces a zero output. When
    ``cu_seqlens`` is given, every non-empty sequence must contain one token;
    empty sequences have no packed output and are skipped.
    """
    num_sequences, H, HV, K = _validate_decode_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        ssm_state_indices,
        cu_seqlens,
        A_log,
        dt_bias,
        lower_bound,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        out,
    )
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    if scale <= 0:
        raise ValueError("`scale` must be positive.")
    if out is None:
        out = torch.empty_like(v)

    BK = triton.next_power_of_2(K)
    if K == 128 and V == 128:
        # A single sequence benefits from a wider two-warp tile.  Other small
        # batches need more independent CTAs, while larger batches benefit from
        # serial V grouping that reuses q/k/g and their L2 norms.
        if num_sequences == 1:
            BV, group_v, num_warps, num_stages = 16, 1, 2, 1
        elif num_sequences * HV <= 256:
            BV, group_v, num_warps, num_stages = 8, 1, 1, 1
        else:
            BV, group_v, num_warps, num_stages = 4, 8, 1, 2
    else:
        BV, group_v, num_warps, num_stages = (
            min(triton.next_power_of_2(V), 8),
            1,
            1,
            2,
        )
    # The common path uses one token for every sequence. Its offsets are known
    # from tensor shapes, so avoid two metadata loads and a dynamic branch.
    use_cu_offsets = cu_seqlens is not None and num_sequences != q.shape[1]
    grid = (triton.cdiv(V, BV * group_v), num_sequences * HV)
    fused_recurrent_kda_decode_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=ssm_state_indices,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        scale=float(scale),
        stride_q_token=q.stride(1),
        stride_q_head=q.stride(2),
        stride_k_token=k.stride(1),
        stride_k_head=k.stride(2),
        stride_v_token=v.stride(1),
        stride_v_head=v.stride(2),
        stride_g_token=g.stride(1),
        stride_g_head=g.stride(2),
        stride_beta_token=beta.stride(1),
        stride_beta_head=beta.stride(2),
        stride_beta_value=beta.stride(3) if beta.ndim == 4 else 0,
        stride_o_token=out.stride(1),
        stride_o_head=out.stride(2),
        stride_state_token=initial_state.stride(0),
        stride_state_head=initial_state.stride(1),
        stride_state_value=initial_state.stride(2),
        stride_state_key=initial_state.stride(3),
        stride_cu_seqlens=cu_seqlens.stride(0) if cu_seqlens is not None else 0,
        stride_state_indices=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        GROUP_V=group_v,
        IS_BETA_HEADWISE=beta.ndim == 4,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        HAS_A=A_log is not None,
        HAS_BIAS=dt_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
        USE_CU_SEQLENS=use_cu_offsets,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-compatible entry for preprocessed one-token serving decode.

    Each non-empty sequence in ``cu_seqlens`` must contain exactly one token.
    Empty sequences are accepted for CUDA Graph padding.  Gate and beta are
    expected to have already been activated, matching the vLLM 0.24 ABI.
    """
    if not inplace_final_state:
        raise ValueError("The decode fast path requires `inplace_final_state=True`.")
    if num_accepted_tokens is not None:
        raise ValueError("Speculative decode must use the generic recurrent path.")
    if ssm_state_indices is None:
        raise ValueError("`ssm_state_indices` is required for serving decode.")
    return fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in wrapper for the preprocessed KDA decode call used by serving.

    Extra KDA policy arguments are intentionally ignored, as in vLLM 0.24:
    the serving model has already applied its safe gate and beta activation
    before entering this compatibility wrapper.  Raw-gate fusion is available
    explicitly through :func:`fused_recurrent_kda_decode`.
    """
    if initial_state is None:
        raise ValueError("`initial_state` is required for serving KDA.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
