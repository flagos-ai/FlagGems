import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    B,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    stride_ab,
    stride_ah,
    stride_ar,
    stride_asb,
    stride_ash,
    stride_ask,
    stride_bh,
    stride_bd,
    stride_br,
    stride_bsh,
    stride_bsd,
    stride_bsk,
    stride_ob,
    stride_oh,
    stride_od,
    N_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    n_m_blocks = tl.cdiv(B, BLOCK_M)
    n_n_blocks = tl.cdiv(D, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * n_n_blocks
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(n_m_blocks - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    a_base = a_ptr + pid_h * stride_ah
    b_base = b_ptr + pid_h * stride_bh

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    d_blk_idx = n_start // 128

    for k_blk in range(N_K_BLOCKS):
        k_start = k_blk * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = a_base + offs_m[:, None] * stride_ab + offs_k[None, :] * stride_ar
        a_mask = (offs_m[:, None] < B) & (offs_k[None, :] < R)
        a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_base + offs_n[None, :] * stride_bd + offs_k[:, None] * stride_br
        b_mask = (offs_n[None, :] < D) & (offs_k[:, None] < R)
        b_fp8 = tl.load(b_ptrs, mask=b_mask, other=0.0)

        dot_result = tl.dot(a_fp8, b_fp8)

        a_scale_ptrs = (
            a_scale_ptr + offs_m * stride_asb + pid_h * stride_ash + k_blk * stride_ask
        )
        a_scale_vals = tl.load(a_scale_ptrs, mask=offs_m < B, other=0.0)
        b_scale_val = tl.load(
            b_scale_ptr
            + pid_h * stride_bsh
            + d_blk_idx * stride_bsd
            + k_blk * stride_bsk
        )

        acc += dot_result * (a_scale_vals[:, None] * b_scale_val)

    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_ob
        + pid_h * stride_oh
        + offs_n[None, :] * stride_od
    )
    out_mask = (offs_m[:, None] < B) & (offs_n[None, :] < D)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str = "bhr,hdr->bhd",
    recipe: tuple | None = None,
) -> None:
    """FP8 block-scaled einsum for DeepSeek V4 O-projection.

    Aligned with vLLM's deepseek_v4_fp8_einsum API.

    Args:
        a: [B, H, R] float8_e4m3fn input tensor
        a_scale: [B, H, ceil(R/128)] float32 block scales for a
        b: [H, D, R] float8_e4m3fn weight tensor
        b_scale: [H, ceil(D/128), ceil(R/128)] float32 block scales for b
        out: [B, H, D] bfloat16 pre-allocated output tensor (modified in-place)
        equation: Einsum equation (currently only "bhr,hdr->bhd" supported)
        recipe: Optional tuning recipe (sfa_gran_k, sfb_gran_k, sfb_gran_mn)
    """
    if equation != "bhr,hdr->bhd":
        raise NotImplementedError(f"Only 'bhr,hdr->bhd' is supported, got '{equation}'")

    logger.debug("GEMS FP8_EINSUM_FORWARD")

    B, H, R = a.shape
    _, D, _ = b.shape

    BLOCK_K = 128
    N_K_BLOCKS = (R + BLOCK_K - 1) // BLOCK_K

    if B <= 16:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 16, 128, 4, 4
    elif B <= 32:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 16, 64, 4, 4
    elif B <= 64:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 5
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 128, 8, 4

    GROUP_SIZE_M = 8
    n_m_blocks = (B + BLOCK_M - 1) // BLOCK_M
    n_n_blocks = (D + BLOCK_N - 1) // BLOCK_N
    grid = (n_m_blocks * n_n_blocks, H)

    _fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        B,
        H,
        R,
        D,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N_K_BLOCKS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
