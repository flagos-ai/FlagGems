import torch
import triton
import triton.language as tl

@triton.jit
def iluvatar_bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ak = tl.load(
            a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=mask_m[:, None] & ((k + offs_k)[None,:] < K),
            other=0.0
        )
        bk = tl.load(
            b_ptr + pid_b * stride_bb + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_n[None, :] & ((k + offs_k)[:,None] < K),
            other=0.0
        )
        acc += tl.dot(ak, bk)

    c_val = acc.to(tl.float16)
    tl.store(
        c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c_val,
        mask=mask_m[:, None] & mask_n[None, :]
    )

def iluvatar_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 3 and b.dim() ==3
    B, M, K = a.shape
    B, K, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = (B, triton.cdiv(M,64), triton.cdiv(N,64))
    iluvatar_bmm_kernel[grid](
        a,b,c, B,M,N,K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )
    return c
