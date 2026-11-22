import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
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
    stride_ob,
    stride_om,
    stride_on,
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

    a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def bmm(mat1: torch.Tensor, mat2: torch.Tensor):
    B, M, K = mat1.shape
    B2, K2, N = mat2.shape
    assert B == B2 and K == K2
    out = torch.empty((B, M, N), device=mat1.device, dtype=mat1.dtype)
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    bmm_kernel[grid](
        mat1,
        mat2,
        out,
        B,
        M,
        N,
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat1.stride(2),
        mat2.stride(0),
        mat2.stride(1),
        mat2.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return out
