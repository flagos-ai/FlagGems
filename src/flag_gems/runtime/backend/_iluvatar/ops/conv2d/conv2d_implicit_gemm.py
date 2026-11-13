# triton_src/conv2d/conv2d_implicit_gemm.py
import torch
import triton
import triton.language as tl


@triton.jit
def _implicit_gemm_conv3x3_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    K,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    stride_kn,
    stride_kc,
    stride_kh,
    stride_kw,
    stride_on,
    stride_ok,
    stride_oh,
    stride_ow,
    BLOCK_H: tl.constexpr = 32,
    BLOCK_W: tl.constexpr = 32,
    BLOCK_C: tl.constexpr = 16,
):
    """
    Implicit GEMM Conv3x3 stride=1 padding=1
    不生成im2col矩阵，直接在kernel内做滑动窗口采样，消除全局内存冗余写
    """
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_c = tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_H, BLOCK_W, K), dtype=tl.float32)
    for kh in range(3):
        for kw in range(3):
            x_h = offs_h + kh - 1
            x_w = offs_w + kw - 1
            mask_h = (x_h >= 0) & (x_h < H)
            mask_w = (x_w >= 0) & (x_w < W)
            x_mask = mask_h[:, None] & mask_w[None, :]
            for c_step in range(0, tl.cdiv(C, BLOCK_C)):
                c_idx = offs_c + c_step * BLOCK_C
                x_ptrs = (
                    x_ptr
                    + pid_n * stride_n
                    + c_idx[None, None, :] * stride_c
                    + x_h[:, None, None] * stride_h
                    + x_w[None, :, None] * stride_w
                )
                x_val = tl.load(
                    x_ptrs,
                    mask=x_mask[:, :, None] & (c_idx[None, None, :] < C),
                    other=0.0,
                )
                w_ptrs = (
                    w_ptr
                    + kh * stride_kh
                    + kw * stride_kw
                    + c_idx[:, None] * stride_kc
                    + tl.arange(0, K)[None, :] * stride_kn
                )
                w_val = tl.load(w_ptrs, mask=c_idx[:, None] < C, other=0.0)
                acc += tl.dot(x_val.reshape(BLOCK_H * BLOCK_W, BLOCK_C), w_val).reshape(
                    BLOCK_H, BLOCK_W, K
                )
    out_ptrs = (
        out_ptr
        + pid_n * stride_on
        + offs_h[:, None, None] * stride_oh
        + offs_w[None, :, None] * stride_ow
    )
    tl.store(out_ptrs, acc.to(tl.float16))


def conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
):
    """
    Conv2d wrapper：优先3x3 stride1 padding1 implicit gemm；其余case fallback aten
    x: [N,Cin,H,W], weight: [Cout,Cin,KH,KW]
    """
    kh, kw = weight.shape[-2], weight.shape[-1]
    if (
        kh == 3
        and kw == 3
        and stride == (1, 1)
        and padding == (1, 1)
        and dilation == (1, 1)
        and groups == 1
    ):
        N, C, H, W = x.shape
        K = weight.shape[0]
        out_h = H
        out_w = W
        out = torch.empty((N, K, out_h, out_w), device=x.device, dtype=x.dtype)
        grid = lambda META: (
            triton.cdiv(out_h, META["BLOCK_H"]),
            triton.cdiv(out_w, META["BLOCK_W"]),
            N,
        )
        _implicit_gemm_conv3x3_kernel[grid](
            x,
            weight,
            out,
            N,
            C,
            H,
            W,
            K,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            BLOCK_H=32,
            BLOCK_W=32,
            BLOCK_C=16,
        )
        if bias is not None:
            out = out + bias[None, :, None, None]
        return out
    else:
        # 非优化case，fallback torch原生
        return torch.nn.functional.conv2d(
            x, weight, bias, stride, padding, dilation, groups
        )
