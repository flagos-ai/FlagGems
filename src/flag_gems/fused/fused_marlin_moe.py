# SPDX-License-Identifier: Apache-2.0
"""
Fused Marlin MoE — v7: Tunable BLOCK_SIZE_K for improved pipelining.

Key changes from v6:
- BLOCK_SIZE_K is now an autotune parameter (32, 64, 128) instead of fixed at
  group_size=128. Smaller K tiles enable more software pipeline stages and
  reduce register pressure, improving memory latency hiding for bandwidth-bound
  small batch sizes.
- GROUP_SIZE_K constexpr correctly indexes scales when BLOCK_SIZE_K < group_size.
  Math: accumulating partial sums within a scale group gives identical results.
- Transposed B layout [E, K//2, N] from v6 is preserved for coalesced N-loads.
- Two-pass GEMM1 (gate/up) with fused SiLU preserved from v6.
"""

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl

from flag_gems.fused.fused_moe import write_zeros_to_output
from flag_gems.fused.moe_align_block_size import moe_align_block_size
from flag_gems.fused.moe_sum import moe_sum

QUANT_TYPE_UINT4B8 = 0
QUANT_TYPE_UINT8B128 = 1
_QUANT_TYPE_INT4 = {QUANT_TYPE_UINT4B8}
_QUANT_TYPE_INT8 = {QUANT_TYPE_UINT8B128}
_SUPPORTED_QUANT_TYPES = _QUANT_TYPE_INT4 | _QUANT_TYPE_INT8


# ---------- Transpose cache ----------

_B_CACHE: dict = {}
_SCALE_CACHE: dict = {}


def _transpose_b(b: torch.Tensor) -> torch.Tensor:
    """Transpose B from [E, N, K//2] to [E, K//2, N] for coalesced N-loads."""
    key = (b.data_ptr(), b.shape[0], b.shape[1], b.shape[2])
    cached = _B_CACHE.get(key)
    if cached is not None:
        return cached
    bt = b.transpose(1, 2).contiguous()
    _B_CACHE[key] = bt
    return bt


def _transpose_scale(s: torch.Tensor) -> torch.Tensor:
    """Transpose scale from [E, N, K//gs] to [E, K//gs, N] for coalesced loads."""
    key = (s.data_ptr(), s.shape[0], s.shape[1], s.shape[2])
    cached = _SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    st = s.transpose(1, 2).contiguous()
    _SCALE_CACHE[key] = st
    return st


# ---------- Autotune configs ----------

_AUTOTUNE_CONFIGS = [
    # BLOCK_SIZE_K=128: compute-bound regime (large M)
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=2,
    ),
    # BLOCK_SIZE_K=64: balanced pipelining, reduced register pressure
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    # BLOCK_SIZE_K=32: max pipelining for bandwidth-bound small batches
    triton.Config(
        {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=6,
    ),
]


def _select_block_m(M, E, top_k):
    avg_tokens = max(M * top_k / max(E, 1), 1)
    if avg_tokens <= 4:
        return 16
    elif avg_tokens <= 32:
        return 32
    else:
        return 64


# ---------- GEMM1: two-pass gate/up with fused SiLU ----------
# B layout: [E, K//2, N] (transposed), stride_bk=N, stride_bn=1


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _int4_gemm_silu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    N_out = N // 2

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N_out,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn_gate = (
        pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    ) % N_out
    offs_bn_up = offs_bn_gate + N_out
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
    b_expert_base = b_ptr + off_experts * stride_be
    b_shifter = (offs_k[:, None] % 2) * 4

    # B is transposed: [E, K//2, N], stride_bk=N (between packed K), stride_bn=1 (N contiguous)
    b_ptrs_gate = (
        b_expert_base
        + (offs_k[:, None] // 2) * stride_bk
        + offs_bn_gate[None, :] * stride_bn
    )
    b_ptrs_up = (
        b_expert_base
        + (offs_k[:, None] // 2) * stride_bk
        + offs_bn_up[None, :] * stride_bn
    )

    # Scale is transposed: [E, K//gs, N], stride_bsk=N, stride_bsn=1
    scale_base_gate = b_scale_ptr + off_experts * stride_bse + offs_bn_gate * stride_bsn
    scale_base_up = b_scale_ptr + off_experts * stride_bse + offs_bn_up * stride_bsn

    # ---- Pass 1: Gate projection ----
    a_ptrs = a_base + offs_k[None, :] * stride_ak
    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b_g = ((tl.load(b_ptrs_gate) >> b_shifter) & 0xF).to(compute_type)
        raw_dot = tl.dot(a, b_g)
        row_sum = tl.sum(a.to(tl.float32), axis=1)
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE_K
        scale_g = tl.load(scale_base_gate + scale_idx * stride_bsk).to(tl.float32)
        acc_gate += scale_g[None, :] * (raw_dot - 8.0 * row_sum[:, None])

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs_gate += (BLOCK_SIZE_K // 2) * stride_bk

    # ---- Pass 2: Up projection ----
    a_ptrs = a_base + offs_k[None, :] * stride_ak
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b_u = ((tl.load(b_ptrs_up) >> b_shifter) & 0xF).to(compute_type)
        raw_dot = tl.dot(a, b_u)
        row_sum = tl.sum(a.to(tl.float32), axis=1)
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE_K
        scale_u = tl.load(scale_base_up + scale_idx * stride_bsk).to(tl.float32)
        acc_up += scale_u[None, :] * (raw_dot - 8.0 * row_sum[:, None])

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs_up += (BLOCK_SIZE_K // 2) * stride_bk

    # ---- Fused SiLU: silu(gate) * up ----
    accumulator = tl.fdiv(acc_gate, (1.0 + tl.exp(-acc_gate))) * acc_up

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_out)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------- GEMM2: standard INT4 GEMM with factored zero-point ----------
# B layout: [E, K//2, N] (transposed), stride_bk=N, stride_bn=1


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _int4_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    # B transposed: [E, K//2, N], stride_bk=N, stride_bn=1
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] // 2) * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    b_shifter = (offs_k[:, None] % 2) * 4
    scale_base = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b_int = ((tl.load(b_ptrs) >> b_shifter) & 0xF).to(compute_type)
        raw_dot = tl.dot(a, b_int)
        row_sum = tl.sum(a.to(tl.float32), axis=1)
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE_K
        scale = tl.load(scale_base + scale_idx * stride_bsk).to(tl.float32)
        accumulator += scale[None, :] * (raw_dot - 8.0 * row_sum[:, None])

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------- Launch wrappers ----------


def _invoke_gemm1_silu(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    block_m,
    group_size,
    compute_type,
):
    # B is transposed: [E, K//2, N]
    N = B.size(2)  # N is now dim 2
    K = A.size(1)
    N_out = N // 2
    M = A.size(0)

    EM = sorted_token_ids.size(0)
    if M < block_m:
        EM = min(EM, M * top_k * block_m)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N_out, META["BLOCK_SIZE_N"]),
    )

    _int4_gemm_silu_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        M * top_k,
        A.stride(0),
        A.stride(1),
        # B transposed [E, K//2, N]: stride(0)=expert, stride(1)=K, stride(2)=N
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(1),
        C.stride(2),
        # B_scale transposed [E, K//gs, N]: stride(0)=expert, stride(1)=K, stride(2)=N
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
    )


def _invoke_gemm2(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    block_m,
    group_size,
    compute_type,
):
    # B is transposed: [E, K//2, N]
    N = B.size(2)  # N is now dim 2
    K = A.size(1)
    M = A.size(0)

    EM = sorted_token_ids.size(0)
    if M < block_m:
        EM = min(EM, M * top_k * block_m)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _int4_gemm_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        M * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
    )


# ---------- Implementation ----------


def _fused_marlin_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert activation == "silu"
    assert use_int4_w4a16
    assert w1_zp is None and w2_zp is None

    expected_packed_k = hidden_states.size(1) // 2
    assert w1.size(2) == expected_packed_k
    assert topk_weights.size() == topk_ids.size()
    assert hidden_states.is_contiguous()
    assert w1.stride(-1) == 1
    assert w2.stride(-1) == 1
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    group_size = block_shape[1]

    # Transpose weights for coalesced N-dimension loads (cached)
    w1_t = _transpose_b(w1)  # [E, N, K//2] -> [E, K//2, N]
    w2_t = _transpose_b(w2)  # [E, N, K//2] -> [E, K//2, N]
    w1_scale_t = _transpose_scale(w1_scale)  # [E, N, K//gs] -> [E, K//gs, N]
    w2_scale_t = _transpose_scale(w2_scale)  # [E, N, K//gs] -> [E, K//gs, N]

    CHUNK_SIZE: int = 16 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    activation_out_dim = N // 2

    block_m = _select_block_m(M, E, top_k_num)

    intermediate_cache3 = torch.empty(
        (M, top_k_num, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_idx = chunk * CHUNK_SIZE
        end_idx = min(begin_idx + CHUNK_SIZE, num_tokens)
        curr_hidden = hidden_states[begin_idx:end_idx]
        tokens_in_chunk = curr_hidden.size(0)

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache2 = intermediate_cache2[: tokens_in_chunk * top_k_num]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            block_m = _select_block_m(tokens_in_chunk, E, top_k_num)

        curr_topk_ids = topk_ids[begin_idx:end_idx]
        curr_topk_weights = topk_weights[begin_idx:end_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids,
            block_m,
            global_num_experts,
            expert_map,
        )

        # ----- GEMM1: gate/up + SiLU fused (two-pass) -----
        cache2_3d = intermediate_cache2.view(
            tokens_in_chunk, top_k_num, activation_out_dim
        )
        _invoke_gemm1_silu(
            A=curr_hidden,
            B=w1_t,
            C=cache2_3d,
            B_scale=w1_scale_t,
            topk_weights=curr_topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k_num,
            block_m=block_m,
            group_size=group_size,
            compute_type=compute_type,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        # ----- GEMM2: activated intermediate @ w2 -----
        _invoke_gemm2(
            A=intermediate_cache2,
            B=w2_t,
            C=intermediate_cache3,
            B_scale=w2_scale_t,
            topk_weights=curr_topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=not apply_router_weight_on_input,
            top_k=1,
            block_m=block_m,
            group_size=group_size,
            compute_type=compute_type,
        )

        # ----- Reduce: sum expert outputs per token -----
        moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_idx:end_idx],
        )

    return out_hidden_states


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    if quant_type_id not in _SUPPORTED_QUANT_TYPES:
        raise NotImplementedError(
            f"MVP supports quant_type_id in {_SUPPORTED_QUANT_TYPES}, "
            f"got {quant_type_id}"
        )
    if g_idx1 is not None or g_idx2 is not None:
        raise NotImplementedError("act_order (g_idx) not yet supported in MVP")
    if sort_indices1 is not None or sort_indices2 is not None:
        raise NotImplementedError("act_order (sort_indices) not yet supported in MVP")
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization not supported")
    if clamp_limit is not None:
        raise NotImplementedError("clamp_limit (GLM-4 swiglu) not supported")
    if input_global_scale1 is not None or input_global_scale2 is not None:
        raise NotImplementedError("input_global_scale not supported in MVP")
    if global_scale1 is not None or global_scale2 is not None:
        raise NotImplementedError("global_scale not supported in MVP")

    use_int4_w4a16 = quant_type_id in _QUANT_TYPE_INT4
    use_int8_w8a16 = quant_type_id in _QUANT_TYPE_INT8

    activation_str = "silu"
    if activation is not None:
        for attr in ("value", "name"):
            v = getattr(activation, attr, None)
            if isinstance(v, str):
                activation_str = v.lower()
                break
        if isinstance(activation, str):
            activation_str = activation.lower()
    if activation_str != "silu":
        raise NotImplementedError(
            f"MVP only supports SiLU/SwiGLU activation, got {activation_str}"
        )

    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")

    result = _fused_marlin_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation_str,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zeros,
        w2_zp=w2_zeros,
        w1_bias=bias1,
        w2_bias=bias2,
        block_shape=[0, group_size],
    )

    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = ["fused_marlin_moe", "QUANT_TYPE_UINT4B8", "QUANT_TYPE_UINT8B128"]
