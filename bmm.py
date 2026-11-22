import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
):
    pid = tl.program_id(0)
    total_mn = GRID_M * GRID_N
    b_idx = pid // total_mn
    mn_idx = pid % total_mn
    pid_m = mn_idx // GRID_N
    pid_n = mn_idx % GRID_N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + b_idx * stride_ab + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(b_idx < B) & (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + b_idx * stride_bb + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(b_idx < B) & ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        c_ptr + b_idx * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(b_idx < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, K = a.shape
    B_b, K_b, N = b.shape
    assert B == B_b and K == K_b, "Batch / K dimension mismatch"
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (B * grid_m * grid_n,)

    bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GRID_M=grid_m,
        GRID_N=grid_n,
    )
    return c


__all__ = ["bmm"]
