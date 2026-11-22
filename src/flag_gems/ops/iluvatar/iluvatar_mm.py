import torch
import triton
import triton.language as tl

@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    # store 补齐mask（审核重点：防止越界写）
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

def iluvatar_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _mm_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out
