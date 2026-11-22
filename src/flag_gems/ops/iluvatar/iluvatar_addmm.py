import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(
    bias_ptr, a_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_bm, stride_bn,
    stride_am, stride_ak,
    stride_bk, stride_bn2,
    stride_om, stride_on,
    alpha, beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn2
    bias = bias_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        a_val = tl.load(a, mask=a_mask, other=0.0)
        b_val = tl.load(b, mask=b_mask, other=0.0)
        acc += tl.dot(a_val, b_val)
        a += BLOCK_K * stride_ak
        b += BLOCK_K * stride_bk

    bias_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias_val = tl.load(bias, mask=bias_mask, other=0.0)
    acc = acc * alpha + bias_val * beta

    out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out, acc, mask=out_mask)

# 关键：对齐项目原有接口：addmm(bias, mat1, mat2, alpha=1, beta=1)
def iluvatar_addmm(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, alpha=1.0, beta=1.0) -> torch.Tensor:
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2
    assert bias.shape == (M, N)
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _addmm_kernel[grid](
        bias, mat1, mat2, out,
        M, K, N,
        bias.stride(0), bias.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        alpha, beta,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
