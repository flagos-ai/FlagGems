import triton
import triton.language as tl
import torch


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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + pid_batch * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_batch * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - kk * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kk * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_batch * stride_cb + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def bmm(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == 3 and b.dim() == 3
    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]
    assert a.is_contiguous() and b.is_contiguous()
    B, M, K = a.shape
    B, K, N = b.shape
    grid = lambda META: (
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    bmm_kernel[grid](
        a,
        b,
        out,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return out
