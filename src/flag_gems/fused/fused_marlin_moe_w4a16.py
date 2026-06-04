from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

from flag_gems.fused.fused_moe import write_zeros_to_output
from flag_gems.fused.moe_align_block_size import moe_align_block_size
from flag_gems.fused.moe_sum import moe_sum
from flag_gems.fused.silu_and_mul import silu_and_mul_out

_W_PACK_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
_SCALE_PACK_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


def _pack_w_interleave(w: torch.Tensor, block_size_k: int) -> torch.Tensor:
    assert w.dtype == torch.uint8
    assert w.ndim == 3
    assert (
        block_size_k % 8 == 0
    ), f"BLOCK_SIZE_K={block_size_k} must be multiple of 8 (8 logical K per int32)"
    E, N_out, K_half = w.shape
    K = K_half * 2
    B = block_size_k // 8
    assert K % (8 * B) == 0, f"K={K} must be divisible by BLOCK_SIZE_K={block_size_k}"
    num_groups = K // (8 * B)

    _NIBBLE_PERM = (0, 4, 1, 5, 2, 6, 3, 7)
    _BIT_SHIFTS = tuple(4 * p for p in _NIBBLE_PERM)
    shifts = torch.tensor(_BIT_SHIFTS, dtype=torch.int32, device=w.device)
    out = torch.empty(E, K // 8, N_out, dtype=torch.int32, device=w.device)

    for e in range(E):
        we = w[e]  # (N_out, K//2) uint8
        low = (we & 0xF).to(torch.uint8)
        high = ((we >> 4) & 0xF).to(torch.uint8)
        unpacked = torch.stack([low, high], dim=-1).reshape(N_out, K)
        tiled = unpacked.reshape(N_out, num_groups, 8, B).transpose(-1, -2)
        # (N_out, num_groups, B, 8)
        packed = (tiled.to(torch.int32) << shifts).sum(dim=-1, dtype=torch.int32)
        # (N_out, num_groups, B) -> (N_out, K//8)
        packed = packed.reshape(N_out, K // 8)
        out[e].copy_(packed.transpose(0, 1))
    return out  # (E, K//8, N_out)


def _pack_scale_transpose(s: torch.Tensor) -> torch.Tensor:
    assert s.ndim == 3
    return s.transpose(-2, -1).contiguous()


def _cached_pack_w(w: torch.Tensor, block_size_k: int, cached: bool) -> torch.Tensor:
    if not cached:
        return _pack_w_interleave(w, block_size_k)
    per_w = _W_PACK_CACHE.get(w)
    if per_w is None:
        per_w = {}
        _W_PACK_CACHE[w] = per_w
    packed = per_w.get(block_size_k)
    if packed is None:
        packed = _pack_w_interleave(w, block_size_k)
        per_w[block_size_k] = packed
    return packed


def _cached_pack_scale(s: torch.Tensor, cached: bool) -> torch.Tensor:
    if not cached:
        return _pack_scale_transpose(s)
    packed = _SCALE_PACK_CACHE.get(s)
    if packed is None:
        packed = _pack_scale_transpose(s)
        _SCALE_PACK_CACHE[s] = packed
    return packed


def w4a16_pack(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    *,
    cached: bool = True,
    pack_strategy: str = "interleave",
    block_size_k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],]:
    if pack_strategy != "interleave":
        raise NotImplementedError(
            f"pack_strategy={pack_strategy!r} not supported (only 'interleave')"
        )
    w1_packed = _cached_pack_w(w1, block_size_k, cached=cached)
    w2_packed = _cached_pack_w(w2, block_size_k, cached=cached)
    w1_scale_packed = (
        _cached_pack_scale(w1_scale, cached=cached) if w1_scale is not None else None
    )
    w2_scale_packed = (
        _cached_pack_scale(w2_scale, cached=cached) if w2_scale is not None else None
    )
    return w1_packed, w2_packed, w1_scale_packed, w2_scale_packed


@triton.jit
def _dequant_int4_fp16(b, scales):
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32  r0, r1, r2, r3, r4, r5, r6, r8, r9, r10, r11, r12;
        .reg .b16  h0, h1, h2, h3, h4, h5, h6, h7;
        .reg .b16  s;
        mov.u32 r0, $8;
        shr.u32 r1, r0, 8;
        lop3.b32 r2, r0, 983055,     1677747200,  234;   // (r0 & 0x000F000F) | 0x64006400
        lop3.b32 r3, r0, 15728880,   1677747200,  234;   // (r0 & 0x00F000F0) | 0x64006400
        lop3.b32 r4, r1, 983055,     1677747200,  234;
        lop3.b32 r5, r1, 15728880,   1677747200,  234;
        mov.u32 r6,  1678271496;                          // 0x64086408 = (1032,1032)
        mov.u32 r8,   738208768;                          // 0x2C002C00 = (1/16,1/16)
        mov.u32 r9,  -729754496;                          // 0xD480D480 = (-72,-72)
        sub.f16x2     r10, r2, r6;
        sub.f16x2     r12, r4, r6;
        fma.rn.f16x2  r11, r3, r8, r9;
        fma.rn.f16x2  r4,  r5, r8, r9;
        mov.b32 {h0, h1}, r10;
        mov.b32 {h2, h3}, r11;
        mov.b32 {h4, h5}, r12;
        mov.b32 {h6, h7}, r4;
        mov.b16 s, $9;
        mul.f16 h0, h0, s;
        mul.f16 h1, h1, s;
        mul.f16 h2, h2, s;
        mul.f16 h3, h3, s;
        mul.f16 h4, h4, s;
        mul.f16 h5, h5, s;
        mul.f16 h6, h6, s;
        mul.f16 h7, h7, s;
        mov.b16 $0, h0;
        mov.b16 $1, h1;
        mov.b16 $2, h2;
        mov.b16 $3, h3;
        mov.b16 $4, h4;
        mov.b16 $5, h5;
        mov.b16 $6, h6;
        mov.b16 $7, h7;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[b, scales],
        dtype=(tl.float16,) * 8,
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


@triton.jit
def _dequant_int4_bf16(b, scales):
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32  r0, r1, r2, r3, q0, q1, q2, q3, s0, s1, s2, s3, magic;
        .reg .b16  h0, h1, h2, h3, h4, h5, h6, h7;
        .reg .b16  s;
        mov.u32 r0, $8;
        shr.u32 r1, r0, 4;          // high nibble of bytes 0,2 -> bits 0-3
        shr.u32 r2, r0, 8;          // low  nibble of bytes 1,3 -> bits 0-3
        shr.u32 r3, r0, 12;         // high nibble of bytes 1,3 -> bits 0-3
        // (x & 0x000F000F) | 0x43004300 -> bf16x2 of (128+nibble, 128+nibble)
        lop3.b32 q0, r0, 983055, 1124090624, 234;
        lop3.b32 q1, r1, 983055, 1124090624, 234;
        lop3.b32 q2, r2, 983055, 1124090624, 234;
        lop3.b32 q3, r3, 983055, 1124090624, 234;
        mov.u32 magic, 1124614920;  // 0x43084308 = (136,136)
        sub.rn.bf16x2 s0, q0, magic;
        sub.rn.bf16x2 s1, q1, magic;
        sub.rn.bf16x2 s2, q2, magic;
        sub.rn.bf16x2 s3, q3, magic;
        mov.b32 {h0, h1}, s0;       // (n0-8, n4-8)
        mov.b32 {h2, h3}, s1;       // (n1-8, n5-8)
        mov.b32 {h4, h5}, s2;       // (n2-8, n6-8)
        mov.b32 {h6, h7}, s3;       // (n3-8, n7-8)
        mov.b16 s, $9;
        mul.rn.bf16 h0, h0, s;
        mul.rn.bf16 h1, h1, s;
        mul.rn.bf16 h2, h2, s;
        mul.rn.bf16 h3, h3, s;
        mul.rn.bf16 h4, h4, s;
        mul.rn.bf16 h5, h5, s;
        mul.rn.bf16 h6, h6, s;
        mul.rn.bf16 h7, h7, s;
        mov.b16 $0, h0;
        mov.b16 $1, h1;
        mov.b16 $2, h2;
        mov.b16 $3, h3;
        mov.b16 $4, h4;
        mov.b16 $5, h5;
        mov.b16 $6, h6;
        mov.b16 $7, h7;
        }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[b, scales],
        dtype=(tl.bfloat16,) * 8,
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


@triton.jit
def _stack_along_dim0(a, b, X: tl.constexpr, Y: tl.constexpr):
    j = tl.join(a, b)  # (X, Y, 2)
    p = tl.permute(j, (2, 0, 1))  # (2, X, Y)
    return tl.reshape(p, (2 * X, Y))  # (2X, Y) block-concat


@triton.jit
def _stack_8(bs, K_PACK: tl.constexpr, N: tl.constexpr):
    s01 = _stack_along_dim0(bs[0], bs[1], K_PACK, N)  # (2*K_PACK, N)
    s23 = _stack_along_dim0(bs[2], bs[3], K_PACK, N)
    s45 = _stack_along_dim0(bs[4], bs[5], K_PACK, N)
    s67 = _stack_along_dim0(bs[6], bs[7], K_PACK, N)
    s0123 = _stack_along_dim0(s01, s23, 2 * K_PACK, N)  # (4*K_PACK, N)
    s4567 = _stack_along_dim0(s45, s67, 2 * K_PACK, N)
    return _stack_along_dim0(s0123, s4567, 4 * K_PACK, N)  # (8*K_PACK, N)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=2
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _w4a16_moe_gemm_kernel(
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
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,  # token tile (MMA M-dim, or N-dim if SWAP_AB)
    BLOCK_SIZE_N: tl.constexpr,  # weight tile (MMA N-dim, or M-dim if SWAP_AB)
    BLOCK_SIZE_K: tl.constexpr,  # logical-K tile (must match packing)
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,  # = quant group_size (e.g. 128)
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8

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
        if SWAP_AB:
            offs_cn0 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs0 = (
                c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn0[:, None]
            )
            c_mask0 = token_mask[None, :] & (offs_cn0[:, None] < N)
            tl.store(
                c_ptrs0,
                tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=compute_type),
                mask=c_mask0,
            )
        else:
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
    offs_ak_pack = tl.arange(0, BLOCK_SIZE_K_PACK)
    offs_bk = tl.arange(0, BLOCK_SIZE_K_PACK)

    if SWAP_AB:
        a_base = a_ptr + (offs_token[None, :] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[:, None] * stride_bn
            + offs_bk[None, :] * stride_bk
        )
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bk[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    scale_base = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_packed = tl.load(b_ptrs)
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE_K
        scale = tl.load(scale_base + scale_idx * stride_bsg)
        scale_bc = scale[:, None] if SWAP_AB else scale[None, :]

        if compute_type == tl.float16:
            bs = _dequant_int4_fp16(b_packed, scale_bc)
        else:
            bs = _dequant_int4_bf16(b_packed, scale_bc)

        k_logical_base = k * BLOCK_SIZE_K
        for j in tl.static_range(8):
            k_off = k_logical_base + j * BLOCK_SIZE_K_PACK
            if SWAP_AB:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[:, None]) * stride_ak
                a_j = tl.load(
                    a_j_ptrs, mask=token_mask[None, :], other=0.0
                )  # (K_PACK, M)
                accumulator = tl.dot(bs[j], a_j, acc=accumulator)  # (N, M)
            else:
                a_j_ptrs = a_base + (k_off + offs_ak_pack[None, :]) * stride_ak
                a_j = tl.load(
                    a_j_ptrs, mask=token_mask[:, None], other=0.0
                )  # (M, K_PACK)
                accumulator = tl.dot(a_j, bs[j], acc=accumulator)  # (M, N)

        b_ptrs += BLOCK_SIZE_K_PACK * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * (
            moe_weight[None, :] if SWAP_AB else moe_weight[:, None]
        )

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if SWAP_AB:
        c_ptrs = c_ptr + stride_cm * offs_token[None, :] + stride_cn * offs_cn[:, None]
        c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_w4a16_moe_gemm(
    A: torch.Tensor,  # (M, K) for GEMM1, (M*top_k, K) for GEMM2
    B: torch.Tensor,  # (E, K//8, N) int32
    C: torch.Tensor,  # (M, top_k, N) or (M*top_k, N) view
    B_scale: torch.Tensor,  # (E, K/gs, N) fp16/bf16
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    block_size_k: int,
    group_size: int,
    compute_type,  # tl.float16 or tl.bfloat16
    swap_ab: bool = False,
):
    M_a = A.size(0)
    K = A.size(1)
    N = B.size(2)
    EM = sorted_token_ids.size(0)
    if M_a < block_m:
        EM = min(EM, M_a * top_k * block_m)

    if C.ndim == 3:
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _w4a16_moe_gemm_kernel[grid](
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
        A.size(0) * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        stride_cm,
        stride_cn,
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWAP_AB=swap_ab,
    )


def fused_moe_w4a16_gptq(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    group_size: int = 128,
    apply_router_weight_on_input: bool = False,
    inplace: bool = False,
    swap_ab: bool = True,
) -> torch.Tensor:
    assert activation == "silu"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert hidden_states.is_contiguous()
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1

    M = hidden_states.size(0)
    K = hidden_states.size(1)
    E = w1.size(0)
    intermediate_size = w1.size(1) // 2
    top_k_num = topk_ids.size(1)

    assert w1.shape == (E, 2 * intermediate_size, K // 2)
    assert w2.shape == (E, K, intermediate_size // 2)
    assert K % group_size == 0
    assert intermediate_size % group_size == 0
    assert w1_scale.shape == (E, 2 * intermediate_size, K // group_size)
    assert w2_scale.shape == (E, K, intermediate_size // group_size)
    assert w1_scale.dtype == hidden_states.dtype
    assert w2_scale.dtype == hidden_states.dtype
    assert topk_weights.shape == topk_ids.shape

    block_size_k = group_size
    # Compute_type for the kernel.
    if hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.bfloat16

    w1_packed, w2_packed, w1_scale_packed, w2_scale_packed = w4a16_pack(
        w1,
        w2,
        w1_scale,
        w2_scale,
        block_size_k=block_size_k,
        cached=True,
    )

    cache13_size = M * top_k_num * max(2 * intermediate_size, K)
    cache13 = torch.empty(
        cache13_size, device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache1 = cache13[: M * top_k_num * 2 * intermediate_size].view(
        M * top_k_num, 2 * intermediate_size
    )
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    avg_tokens = max(M * top_k_num // max(E, 1), 1)
    cutoff = 8 if swap_ab else 16
    block_m = 16 if avg_tokens <= cutoff else (32 if avg_tokens <= 64 else 64)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_m,
        num_experts=E,
        expert_map=None,
    )

    _invoke_w4a16_moe_gemm(
        A=hidden_states,
        B=w1_packed,
        C=intermediate_cache1,
        B_scale=w1_scale_packed,
        topk_weights=topk_weights if apply_router_weight_on_input else None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=apply_router_weight_on_input,
        top_k=top_k_num,
        block_m=block_m,
        block_size_k=block_size_k,
        group_size=group_size,
        compute_type=compute_type,
        swap_ab=swap_ab,
    )

    gate = intermediate_cache1[:, :intermediate_size]
    up = intermediate_cache1[:, intermediate_size:]
    silu_and_mul_out(gate, up, intermediate_cache2)

    _invoke_w4a16_moe_gemm(
        A=intermediate_cache2,
        B=w2_packed,
        C=intermediate_cache3,
        B_scale=w2_scale_packed,
        topk_weights=topk_weights if not apply_router_weight_on_input else None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=not apply_router_weight_on_input,
        top_k=1,
        block_m=block_m,
        block_size_k=block_size_k,
        group_size=group_size,
        compute_type=compute_type,
        swap_ab=swap_ab,
    )

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    moe_sum(intermediate_cache3, out_hidden_states)

    return out_hidden_states


__all__ = ["w4a16_pack", "fused_moe_w4a16_gptq"]
