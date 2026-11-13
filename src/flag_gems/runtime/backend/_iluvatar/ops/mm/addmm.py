# triton_src/mm/addmm.py
import torch
import triton
import triton.language as tl


@triton.jit
def _addmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """
    addmm: out = alpha * A @ B + beta * C
    ✅ 融合实现：mma计算完成后，寄存器内完成alpha缩放 + beta*C切片，不写中间全局张量
    """
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

    a_base = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_base = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_base, mask=offs_k[None, :] < K - k_step * BLOCK_K, other=0.0)
        b = tl.load(b_base, mask=offs_k[:, None] < K - k_step * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # 融合：alpha*gemm + beta*C，寄存器内完成
    c_slice = tl.load(c_ptrs)
    acc = acc * alpha + c_slice.to(tl.float32) * beta

    out_ptrs = out_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16))


def addmm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    torch.addmm(c, a, b, alpha=alpha, beta=beta)
    out = alpha * A @ B + beta * C
    """
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
    assert (
        a.dtype == torch.float16
        and b.dtype == torch.float16
        and c.dtype == torch.float16
    )
    M, K = a.shape
    K2, N = b.shape
    Mc, Nc = c.shape
    assert K == K2 and M == Mc and N == Nc
    out = torch.empty_like(c)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _addmm_kernel[grid](
        a,
        b,
        c,
        out,
        alpha,
        beta,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_SIZE_M=8,
    )
    return out
