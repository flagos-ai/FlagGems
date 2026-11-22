import torch
import triton
import triton.language as tl

@triton.jit
def iluvatar_addmm_kernel(
    bias_ptr, mat1_ptr, mat2_ptr, out_ptr,
    M, N, K,
    stride_bias_m, stride_bias_n,
    stride_m1m, stride_m1k,
    stride_m2k, stride_m2n,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    bias = tl.load(
        bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            mat1_ptr + offs_m[:, None] * stride_m1m + (k + offs_k)[None, :] * stride_m1k,
            mask=mask_m[:, None] & ((k + offs_k)[None,:] < K),
            other=0.0
        )
        b = tl.load(
            mat2_ptr + (k + offs_k)[:, None] * stride_m2k + offs_n[None, :] * stride_m2n,
            mask=mask_n[None, :] & ((k + offs_k)[:,None] < K),
            other=0.0
        )
        acc += tl.dot(a, b)

    out_val = acc + bias
    out_val = out_val.to(tl.float16)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out_val,
        mask=mask_m[:, None] & mask_n[None, :]
    )

def iluvatar_addmm(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Standard addmm interface: addmm(bias, mat1, mat2)
    out = bias + mat1 @ mat2
    align with torch.addmm(input, mat1, mat2)
    """
    M, K = mat1.shape
    K, N = mat2.shape
    out = torch.empty((M, N), device=bias.device, dtype=bias.dtype)

    grid = (triton.cdiv(M,64), triton.cdiv(N,64))
    iluvatar_addmm_kernel[grid](
        bias, mat1, mat2, out,
        M, N, K,
        bias.stride(0), bias.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    )
    return out
