import torch
import triton
import triton.language as tl


@triton.jit
def mm_conv_glu_fused_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
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
    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn

    mask_am = offs_am < M
    mask_bn = offs_bn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        mask_k = offs_k + k_offs < K
        x = tl.load(x_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    half = N // 2
    if pid_n * BLOCK_N >= half:
        return
    mask_out = offs_bn < half
    y1 = acc[:, :half]
    y2 = acc[:, half:]
    out_val = y1 * tl.sigmoid(y2)
    out_val = out_val.to(tl.float16)
    out_ptrs = out_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    tl.store(out_ptrs, out_val, mask=mask_am[:, None] & mask_out[None, :])


def grid_fused(N, BLOCK_N, GROUP_SIZE):
    num_pid_n = triton.cdiv(N, BLOCK_N)
    return (GROUP_SIZE * num_pid_n,)


class MMConvGLUFusedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        M, K = x.shape
        K2, N = w.shape
        assert K == K2
        assert N % 2 == 0
        out = torch.empty((M, N // 2), device=x.device, dtype=x.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE = 128, 128, 32, 8
        mm_conv_glu_fused_kernel[grid_fused(N, BLOCK_N, GROUP_SIZE)](
            x,
            w,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=3,
            num_warps=8,
        )
        ctx.save_for_backward(x, w)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w = ctx.saved_tensors
        y = torch.matmul(x, w)
        half = w.shape[1] // 2
        y1 = y[:, :half]
        y2 = y[:, half:]
        sig = torch.sigmoid(y2)
        dy1 = dout * sig
        dy2 = dout * y1 * sig * (1 - sig)
        dy = torch.cat([dy1, dy2], dim=-1)
        dx = torch.matmul(dy, w.T)
        dw = torch.matmul(x.T, dy)
        return dx, dw


def mm_conv_glu_fused(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return MMConvGLUFusedFunc.apply(x, w)
