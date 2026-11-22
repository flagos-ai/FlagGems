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

    a = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        a_val = tl.load(a, mask=a_mask, other=0.0)
        b_val = tl.load(b, mask=b_mask, other=0.0)
        acc += tl.dot(a_val, b_val)
        a += BLOCK_K * stride_ak
        b += BLOCK_K * stride_bk

    out = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out, acc, mask=out_mask)

def iluvatar_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 3 and b.dim() == 3
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _bmm_kernel[grid](
        a, b, out,
        B, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
