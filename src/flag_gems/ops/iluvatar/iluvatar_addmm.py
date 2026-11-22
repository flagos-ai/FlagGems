import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(
    bias_ptr, a_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_bm, stride_bn,
    stride_am, stride_ak,
    stride_bk, stride_bn_,
    stride_om, stride_on,
    beta, alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_offs + offs_k)[None, :] * stride_ak,
            mask=mask_m[:, None] & ((k_offs + offs_k) < K)[None, :],
            other=0.0
        )
        b = tl.load(
            b_ptr + (k_offs + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn_,
            mask=((k_offs + offs_k) < K)[:, None] & mask_n[None, :],
            other=0.0
        )
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=mask, other=0.0)
    acc = acc * alpha + bias * beta
    out = acc.to(tl.float16)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=mask)

def iluvatar_addmm(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta=1.0, alpha=1.0, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    M, K = mat1.shape
    Kb, N = mat2.shape
    assert K == Kb
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    _addmm_kernel[grid](
        bias, mat1, mat2, out,
        M, K, N,
        bias.stride(0), bias.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        beta, alpha,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out
