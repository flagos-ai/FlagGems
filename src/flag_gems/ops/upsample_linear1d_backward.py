import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    n,
    c,
    in_w,
    out_w,
    go_stride_n,
    go_stride_c,
    go_stride_w,
    gi_stride_n,
    gi_stride_c,
    gi_stride_w,
    align_corners: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    total = n * c * in_w
    mask = offs < total

    x_in = offs % in_w
    tmp = offs // in_w
    c_idx = tmp % c
    n_idx = tmp // c

    x_in_f = x_in.to(tl.float32)
    in_w_f = tl.cast(in_w, tl.float32)
    out_w_f = tl.cast(out_w, tl.float32)

    if align_corners:
        if in_w > 1:
            center = x_in_f * (out_w_f - 1.0) / (in_w_f - 1.0)
        else:
            center = tl.zeros([BLOCK], dtype=tl.float32)
    else:
        center = (x_in_f + 0.5) * out_w_f / in_w_f - 0.5

    base = tl.floor(center).to(tl.int32)

    go_base = grad_out_ptr + n_idx * go_stride_n + c_idx * go_stride_c

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for i in range(-2, 3):
        x_out = base + i
        valid = (x_out >= 0) & (x_out < out_w)
        x_out_f = x_out.to(tl.float32)

        if align_corners:
            if out_w > 1:
                x_real = x_out_f * (in_w_f - 1.0) / (out_w_f - 1.0)
            else:
                x_real = tl.zeros([BLOCK], dtype=tl.float32)
        else:
            x_real = (x_out_f + 0.5) * in_w_f / out_w_f - 0.5

        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1

        x0_i = tl.maximum(x0_f, 0.0).to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, in_w_f - 1.0).to(tl.int32)

        g = tl.load(
            go_base + x_out * go_stride_w,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)

        same = x0_i == x1_i
        is_x0 = x_in.to(tl.int32) == x0_i
        is_x1 = x_in.to(tl.int32) == x1_i

        acc += tl.where(same & is_x0, g * (w0 + w1), 0.0)
        acc += tl.where(~same & is_x0, g * w0, 0.0)
        acc += tl.where(~same & is_x1, g * w1, 0.0)

    gi_ptr = (
        grad_in_ptr + n_idx * gi_stride_n + c_idx * gi_stride_c + x_in * gi_stride_w
    )
    tl.store(gi_ptr, acc, mask=mask)


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE LINEAR1D BACKWARD")
    assert grad_output.is_cuda

    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    grad_out_3d = grad_output.contiguous().view(n, c, out_w)

    grad_in = torch.zeros(
        (n, c, in_w),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    go_stride_n, go_stride_c, go_stride_w = grad_out_3d.stride()
    gi_stride_n, gi_stride_c, gi_stride_w = grad_in.stride()

    BLOCK = 512
    grid = (triton.cdiv(n * c * in_w, BLOCK),)

    upsample_linear1d_backward_kernel[grid](
        grad_out_3d,
        grad_in,
        n,
        c,
        in_w,
        out_w,
        go_stride_n,
        go_stride_c,
        go_stride_w,
        gi_stride_n,
        gi_stride_c,
        gi_stride_w,
        align_corners,
        BLOCK=BLOCK,
    )

    return grad_in
