import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(
    bias_ptr, a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_bias_m, stride_bias_n,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)
    bias = tl.load(
        bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0
    )
    out_val = beta * bias + alpha * acc
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out_val.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

def iluvatar_addmm(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2
    out = torch.empty((M, N), device=bias.device, dtype=bias.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _addmm_kernel[grid](
        bias, mat1, mat2, out,
        M, N, K,
        bias.stride(0), bias.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        alpha=alpha, beta=beta
    )
    return out
