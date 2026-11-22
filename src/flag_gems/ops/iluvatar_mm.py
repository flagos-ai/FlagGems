import torch
import triton
import triton.language as tl

@triton.jit
def iluvatar_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=mask_m[:, None] & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_n[None, :] & ((k + offs_k)[:, None] < K),
            other=0.0
        )
        acc += tl.dot(a, b)

    c_dtype = tl.float16
    c_val = acc.to(c_dtype)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c_val,
        mask=mask_m[:, None] & mask_n[None, :]
    )

def iluvatar_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [M, K], b: [K, N]
    return: [M, N]
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    iluvatar_mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    )
    return c
