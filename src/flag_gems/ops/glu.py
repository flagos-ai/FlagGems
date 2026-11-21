import torch
import triton
import triton.language as tl


@triton.jit
def glu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid
    offs_n = tl.arange(0, BLOCK_N)
    half = N // 2

    x1_ptrs = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    mask1 = offs_n < half
    x1 = tl.load(x1_ptrs, mask=mask1, other=0.0)

    x2_ptrs = x_ptr + offs_m * stride_xm + (offs_n + half) * stride_xn
    mask2 = offs_n + half < N
    x2 = tl.load(x2_ptrs, mask=mask2, other=0.0)

    out = x1 * tl.sigmoid(x2)
    out_ptrs = out_ptr + offs_m * stride_om + offs_n * stride_on
    tl.store(out_ptrs, out, mask=mask1)


def grid_glu(M):
    return (M,)


class GLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        M, N = x.shape
        assert N % 2 == 0
        out = torch.empty((M, N // 2), device=x.device, dtype=x.dtype)
        BLOCK_N = 256
        glu_kernel[grid_glu(M)](
            x,
            out,
            M,
            N,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        M, N = x.shape
        half = N // 2
        x1 = x[:, :half]
        x2 = x[:, half:]
        sig = torch.sigmoid(x2)
        dx1 = dout * sig
        dx2 = dout * x1 * sig * (1 - sig)
        dx = torch.cat([dx1, dx2], dim=-1)
        return dx


def glu(x: torch.Tensor) -> torch.Tensor:
    return GLUFunc.apply(x)
