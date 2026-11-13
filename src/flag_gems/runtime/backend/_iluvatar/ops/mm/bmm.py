# triton_src/mm/bmm.py
import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_kernel(
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
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """
    BMM: [B, M, K] @ [B, K, N] -> [B, M, N]
    ✅ batch维度放到program_id(1)并行，不是for循环遍历batch，大幅提升多batch利用率
    """
    pid_batch = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = (
        a_ptr
        + pid_batch * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_base = (
        b_ptr
        + pid_batch * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_base, mask=offs_k[None, :] < K - k_step * BLOCK_K, other=0.0)
        b = tl.load(b_base, mask=offs_k[:, None] < K - k_step * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (
        c_ptr
        + pid_batch * stride_cb
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc.to(tl.float16))


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.bmm(A,B), shape [B,M,K] @ [B,K,N]"""
    assert a.is_contiguous() and b.is_contiguous()
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        B,
    )
    _bmm_kernel[grid](
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
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_SIZE_M=8,
    )
    return c
