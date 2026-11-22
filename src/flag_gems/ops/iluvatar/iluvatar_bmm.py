import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, out_ptr,
    B, M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_b = pid_b < B
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        a = tl.load(
            a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + (k_offs + offs_k)[None, :] * stride_ak,
            mask=mask_b & mask_m[:, None] & ((k_offs + offs_k) < K)[None, :],
            other=0.0
        )
        b = tl.load(
            b_ptr + pid_b * stride_bb + (k_offs + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_b & ((k_offs + offs_k) < K)[:, None] & mask_n[None, :],
            other=0.0
        )
        acc += tl.dot(a, b)

    out = acc.to(tl.float16)
    out_mask = mask_b & mask_m[:, None] & mask_n[None, :]
    tl.store(
        out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out, mask=out_mask
    )

def iluvatar_bmm(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    B, M, K = a.shape
    Bb, Kb, N = b.shape
    assert B == Bb and K == Kb
    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (B, grid_m, grid_n)
    _bmm_kernel[grid](
        a, b, out,
        B, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out
