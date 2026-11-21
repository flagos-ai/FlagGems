import torch
import triton
import triton.language as tl


@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
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
            a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :],
            mask=(k + offs_k)[None, :] < K,
        )
        b = tl.load(
            b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :],
            mask=(k + offs_k)[:, None] < K,
        )
        acc += tl.dot(a, b)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def mm(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mm_kernel[grid](a, b, out, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    return out
