import torch
import triton
import triton.language as tl


@triton.jit
def addmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
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
    alpha,
    beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn

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

    c_val = tl.load(c_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    out_val = acc * alpha + c_val * beta
    out_val = out_val.to(tl.float16)
    out_ptrs = out_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    tl.store(out_ptrs, out_val, mask=mask_c)


def grid_addmm(N, BLOCK_N, GROUP_SIZE):
    num_pid_n = triton.cdiv(N, BLOCK_N)
    return (GROUP_SIZE * num_pid_n,)


class AddMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, a, b, alpha=1.0, beta=1.0):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE = 128, 128, 32, 8
        addmm_kernel[grid_addmm(N, BLOCK_N, GROUP_SIZE)](
            a,
            b,
            c,
            out,
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
            alpha,
            beta,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=3,
            num_warps=8,
        )
        ctx.save_for_backward(a, b)
        ctx.alpha = alpha
        ctx.beta = beta
        return out

    @staticmethod
    def backward(ctx, dout):
        a, b = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        dc = dout * beta
        da = torch.matmul(dout * alpha, b.T)
        db = torch.matmul(a.T, dout * alpha)
        return dc, da, db, None, None


def addmm(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha=1.0, beta=1.0) -> torch.Tensor:
    return AddMMFunc.apply(c, a, b, alpha, beta)
