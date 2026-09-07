import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _load_grad_output(
    grad_ptr,
    grad_base,
    d_out,
    h_out,
    w_out,
    mask,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
):
    offsets = grad_base + (d_out * H_OUT + h_out) * W_OUT + w_out
    return tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _sum_width_positions(
    grad_ptr,
    grad_base,
    d_out,
    h_out,
    w_idx,
    mask,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    W_IN: tl.constexpr,
    PAD_W0: tl.constexpr,
    PAD_W1: tl.constexpr,
):
    acc = _load_grad_output(
        grad_ptr,
        grad_base,
        d_out,
        h_out,
        w_idx + PAD_W0,
        mask,
        H_OUT,
        W_OUT,
    )

    if PAD_W0 > 0:
        left_mask = mask & (w_idx > 0) & (w_idx <= PAD_W0)
        acc += _load_grad_output(
            grad_ptr,
            grad_base,
            d_out,
            h_out,
            PAD_W0 - w_idx,
            left_mask,
            H_OUT,
            W_OUT,
        )

    if PAD_W1 > 0:
        right_mask = mask & (w_idx < W_IN - 1) & (w_idx >= W_IN - PAD_W1 - 1)
        acc += _load_grad_output(
            grad_ptr,
            grad_base,
            d_out,
            h_out,
            PAD_W0 + 2 * (W_IN - 1) - w_idx,
            right_mask,
            H_OUT,
            W_OUT,
        )

    return acc


@triton.jit
def _sum_height_positions(
    grad_ptr,
    grad_base,
    d_out,
    h_idx,
    w_idx,
    mask,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    PAD_H0: tl.constexpr,
    PAD_H1: tl.constexpr,
    PAD_W0: tl.constexpr,
    PAD_W1: tl.constexpr,
):
    acc = _sum_width_positions(
        grad_ptr,
        grad_base,
        d_out,
        h_idx + PAD_H0,
        w_idx,
        mask,
        H_OUT,
        W_OUT,
        W_IN,
        PAD_W0,
        PAD_W1,
    )

    if PAD_H0 > 0:
        top_mask = mask & (h_idx > 0) & (h_idx <= PAD_H0)
        acc += _sum_width_positions(
            grad_ptr,
            grad_base,
            d_out,
            PAD_H0 - h_idx,
            w_idx,
            top_mask,
            H_OUT,
            W_OUT,
            W_IN,
            PAD_W0,
            PAD_W1,
        )

    if PAD_H1 > 0:
        bottom_mask = mask & (h_idx < H_IN - 1) & (h_idx >= H_IN - PAD_H1 - 1)
        acc += _sum_width_positions(
            grad_ptr,
            grad_base,
            d_out,
            PAD_H0 + 2 * (H_IN - 1) - h_idx,
            w_idx,
            bottom_mask,
            H_OUT,
            W_OUT,
            W_IN,
            PAD_W0,
            PAD_W1,
        )

    return acc


@libentry()
@triton.jit
def reflection_pad3d_backward_kernel(
    grad_ptr,
    out_ptr,
    C: tl.constexpr,
    D_IN: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    D_OUT: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    PAD_D0: tl.constexpr,
    PAD_D1: tl.constexpr,
    PAD_H0: tl.constexpr,
    PAD_H1: tl.constexpr,
    PAD_W0: tl.constexpr,
    PAD_W1: tl.constexpr,
    stride_out_n,
    stride_out_c,
    stride_out_d,
    stride_out_h,
    stride_out_w,
    BLOCK_DHW: tl.constexpr,
):
    pid_dhw = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_n = pid_nc // C
    pid_c = pid_nc - pid_n * C

    in_offsets = pid_dhw * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = in_offsets < D_IN * H_IN * W_IN

    hw_in = H_IN * W_IN
    d_idx = in_offsets // hw_in
    rem = in_offsets - d_idx * hw_in
    h_idx = rem // W_IN
    w_idx = rem - h_idx * W_IN

    grad_base = pid_nc * D_OUT * H_OUT * W_OUT

    acc = _sum_height_positions(
        grad_ptr,
        grad_base,
        d_idx + PAD_D0,
        h_idx,
        w_idx,
        mask,
        H_OUT,
        W_OUT,
        H_IN,
        W_IN,
        PAD_H0,
        PAD_H1,
        PAD_W0,
        PAD_W1,
    )

    if PAD_D0 > 0:
        front_mask = mask & (d_idx > 0) & (d_idx <= PAD_D0)
        acc += _sum_height_positions(
            grad_ptr,
            grad_base,
            PAD_D0 - d_idx,
            h_idx,
            w_idx,
            front_mask,
            H_OUT,
            W_OUT,
            H_IN,
            W_IN,
            PAD_H0,
            PAD_H1,
            PAD_W0,
            PAD_W1,
        )

    if PAD_D1 > 0:
        back_mask = mask & (d_idx < D_IN - 1) & (d_idx >= D_IN - PAD_D1 - 1)
        acc += _sum_height_positions(
            grad_ptr,
            grad_base,
            PAD_D0 + 2 * (D_IN - 1) - d_idx,
            h_idx,
            w_idx,
            back_mask,
            H_OUT,
            W_OUT,
            H_IN,
            W_IN,
            PAD_H0,
            PAD_H1,
            PAD_W0,
            PAD_W1,
        )

    out_offsets = (
        pid_n * stride_out_n
        + pid_c * stride_out_c
        + d_idx * stride_out_d
        + h_idx * stride_out_h
        + w_idx * stride_out_w
    )
    tl.store(out_ptr + out_offsets, acc, mask=mask)


def _check_padding(padding, d_in, h_in, w_in):
    if isinstance(padding, int):
        pad_d0 = pad_d1 = pad_h0 = pad_h1 = pad_w0 = pad_w1 = padding
    else:
        if len(padding) != 6:
            raise ValueError("padding must contain 6 integers")
        pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding

    pads = (pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1)
    if any(pad < 0 for pad in pads):
        raise ValueError("reflection_pad3d does not support negative padding")
    if pad_d0 >= d_in or pad_d1 >= d_in:
        raise ValueError("depth padding size must be less than input depth")
    if pad_h0 >= h_in or pad_h1 >= h_in:
        raise ValueError("height padding size must be less than input height")
    if pad_w0 >= w_in or pad_w1 >= w_in:
        raise ValueError("width padding size must be less than input width")

    return pads


def reflection_pad3d_backward(grad_output, self, padding):
    """Compute gradient of reflection_pad3d forward pass."""
    logger.debug("GEMS REFLECTION_PAD3D_BACKWARD")

    if self.dim() != 5:
        raise ValueError("input must be a 5D tensor")

    N, C, D_in, H_in, W_in = self.shape
    pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = _check_padding(
        padding, D_in, H_in, W_in
    )

    D_out = D_in + pad_d0 + pad_d1
    H_out = H_in + pad_h0 + pad_h1
    W_out = W_in + pad_w0 + pad_w1

    expected_grad_shape = (N, C, D_out, H_out, W_out)
    if tuple(grad_output.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_grad_shape}"
        )

    if (
        pad_d0 == 0
        and pad_d1 == 0
        and pad_h0 == 0
        and pad_h1 == 0
        and pad_w0 == 0
        and pad_w1 == 0
    ):
        return torch.empty_like(self).copy_(grad_output)

    if N == 0 or C == 0:
        return torch.empty_like(self)

    grad_output = grad_output.contiguous()
    out = torch.empty_like(self)
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w = out.stride()

    BLOCK_DHW = 256
    grid = (triton.cdiv(D_in * H_in * W_in, BLOCK_DHW), N * C)

    reflection_pad3d_backward_kernel[grid](
        grad_output,
        out,
        C=C,
        D_IN=D_in,
        H_IN=H_in,
        W_IN=W_in,
        D_OUT=D_out,
        H_OUT=H_out,
        W_OUT=W_out,
        PAD_D0=pad_d0,
        PAD_D1=pad_d1,
        PAD_H0=pad_h0,
        PAD_H1=pad_h1,
        PAD_W0=pad_w0,
        PAD_W1=pad_w1,
        stride_out_n=stride_out_n,
        stride_out_c=stride_out_c,
        stride_out_d=stride_out_d,
        stride_out_h=stride_out_h,
        stride_out_w=stride_out_w,
        BLOCK_DHW=BLOCK_DHW,
        num_warps=4,
    )

    return out
