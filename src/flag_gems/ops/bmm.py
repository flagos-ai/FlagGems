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
):
    pid = tl.program_id(axis=0)
    total_mn = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    batch_idx = pid // total_mn
    rem = pid % total_mn
    pid_m = rem // tl.cdiv(N, BLOCK_N)
    pid_n = rem % tl.cdiv(N, BLOCK_N)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + batch_idx * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + batch_idx * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mask_am = offs_am < M
    mask_bn = offs_bn < N
    mask_c = mask_am[:, None] & mask_bn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        mask_k = offs_k + k_offs < K
        a = tl.load(a_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_val = acc.to(tl.float16)
    c_ptrs = c_ptr + batch_idx * stride_cb + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, c_val, mask=mask_c)


def grid_bmm(B, M, N, BLOCK_M, BLOCK_N):
    return (B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)


class BMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        B, M, K = a.shape
        B2, K2, N = b.shape
        assert B == B2 and K == K2
        c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        bmm_kernel[grid_bmm(B, M, N, BLOCK_M, BLOCK_N)](
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
            num_stages=3,
            num_warps=4,
        )
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da = torch.bmm(dc, b.transpose(-1, -2))
        db = torch.bmm(a.transpose(-1, -2), dc)
        return da, db


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return BMMFunc.apply(a, b)
