import torch
import triton
import triton.language as tl


@triton.jit
def max_pool3d_kernel(
    x_ptr,
    out_ptr,
    D,
    H,
    W,
    oD,
    oH,
    oW,
    k,
    s,
    p,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
):
    idx = tl.program_id(0)
    pd = tl.program_id(1)
    hw_idx = tl.program_id(2)
    ph = hw_idx // oW
    pw = hw_idx % oW

    x_base = x_ptr + idx * stride_c
    max_val = -float("inf")
    for i in range(k):
        for j in range(k):
            for m in range(k):
                d_in = pd * s - p + i
                h_in = ph * s - p + j
                w_in = pw * s - p + m
                valid = (
                    d_in >= 0
                    and d_in < D
                    and h_in >= 0
                    and h_in < H
                    and w_in >= 0
                    and w_in < W
                )
                if valid:
                    offset = d_in * stride_d + h_in * stride_h + w_in * stride_w
                    val = tl.load(x_base + offset)
                    max_val = tl.maximum(max_val, val)

    out_idx = idx * (oD * oH * oW) + pd * (oH * oW) + ph * oW + pw
    tl.store(out_ptr + out_idx, max_val)


def max_pool3d_triton(x, kernel_size=3, stride=2, padding=1):
    B, C, D, H, W = x.shape
    oD = (D + 2 * padding - kernel_size) // stride + 1
    oH = (H + 2 * padding - kernel_size) // stride + 1
    oW = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.zeros(B * C, oD, oH, oW, device=x.device, dtype=x.dtype)
    grid = (B * C, oD, oH * oW)
    max_pool3d_kernel[grid](
        x,
        out,
        D,
        H,
        W,
        oD,
        oH,
        oW,
        kernel_size,
        stride,
        padding,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
    )
    return out.view(B, C, oD, oH, oW)
