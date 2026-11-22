import triton
import triton.language as tl
import torch


@triton.jit
def addmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    alpha,
    beta,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k - kk * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k - kk * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc *= alpha

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    c_val = tl.load(c_ptrs, mask=c_mask, other=0.0)
    acc = acc + beta * c_val
    tl.store(c_ptrs, acc, mask=c_mask)


def addmm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha=1.0, beta=1.0):
    assert a.shape[1] == b.shape[0], "K dim mismatch"
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
    m, k = a.shape
    k, n = b.shape
    grid = lambda META: (triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),)
    out = c.clone()
    addmm_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        alpha,
        beta,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return out
