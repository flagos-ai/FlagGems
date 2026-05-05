"""
Triton implementation of conv_transpose2d operator.

Supports all PyTorch parameters: stride, padding, output_padding, groups, dilation.
Compatible with FlagTree backends (Iluvatar, Ascend, etc.) via standard Triton APIs.

Weight layout follows PyTorch convention: (C_in, C_out/groups, kH, kW).

Algorithm: For each output position (n, co, oh, ow), iterate over input channels
and kernel positions using element-wise accumulation. Uses small tiles (32, 16)
optimized for BI-V150's 16 SMs and 64-wide warps.

Backward pass uses PyTorch native conv_transpose2d with autograd for correctness.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    output_padding: int,
) -> int:
    """Compute output spatial dimension for conv_transpose2d."""
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


# Tile sizes optimized for BI-V150 (16 SMs, 64-wide warps)
BLOCK_NHWO = 32
BLOCK_CO = 16


@libentry()
@triton.jit
def conv_transpose2d_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    N,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    input_n_stride,
    input_c_stride,
    input_h_stride,
    input_w_stride,
    weight_ci_stride,
    weight_co_stride,
    weight_h_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_h_stride,
    output_w_stride,
    C_in_per_group: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NHWO: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Element-wise accumulation kernel for conv_transpose2d.

    For each output tile (BLOCK_NHWO spatial positions x BLOCK_CO channels),
    iterates over (C_in_per_group, kH, kW) and accumulates weighted input values.
    Uses small tiles (32, 16) optimized for BI-V150's 16 SMs.
    """
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    nhw_offset = pid_nhw * BLOCK_NHWO + tl.arange(0, BLOCK_NHWO)
    nh_offset = nhw_offset // W_out
    n_value = nh_offset // H_out
    oh_value = nh_offset % H_out
    ow_value = nhw_offset % W_out

    out_per_group = C_out // groups
    co_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_global = pid_group * out_per_group + co_offset

    inp_base = input_n_stride * n_value + input_c_stride * pid_group * C_in_per_group
    w_base = weight_ci_stride * pid_group * C_in_per_group + weight_co_stride * co_offset

    acc = tl.zeros((BLOCK_NHWO, BLOCK_CO), dtype=tl.float32)

    for ci in range(C_in_per_group):
        ci_inp_offset = input_c_stride * ci
        ci_w_offset = weight_ci_stride * ci
        for kh in range(kH):
            ih_num = oh_value + pad_h - kh * dil_h
            ih = ih_num // stride_h
            ih_valid = (ih_num >= 0) & (ih_num % stride_h == 0) & (ih < H_in)
            for kw in range(kW):
                iw_num = ow_value + pad_w - kw * dil_w
                iw = iw_num // stride_w
                valid = (
                    ih_valid
                    & (iw_num >= 0)
                    & (iw_num % stride_w == 0)
                    & (iw < W_in)
                    & (n_value < N)
                )

                inp_ptrs = (
                    input_pointer
                    + inp_base
                    + ci_inp_offset
                    + input_h_stride * ih
                    + input_w_stride * iw
                )
                inp_val = tl.load(inp_ptrs, mask=valid, other=0.0).to(tl.float32)

                w_ptrs = (
                    weight_pointer
                    + w_base
                    + ci_w_offset
                    + weight_h_stride * kh
                    + weight_w_stride * kw
                )
                w_mask = co_offset < out_per_group
                w_val = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                acc += inp_val[:, None] * w_val[None, :]

    bias_ptrs = bias_pointer + co_global[None, :]
    bias_mask = (co_offset < out_per_group)[None, :]
    bias_val = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias_val

    out_ptrs = (
        output_pointer
        + n_value[:, None] * output_n_stride
        + co_global[None, :] * output_c_stride
        + oh_value[:, None] * output_h_stride
        + ow_value[:, None] * output_w_stride
    )
    out_mask = (
        (n_value < N)[:, None]
        & (co_offset < out_per_group)[None, :]
        & (oh_value < H_out)[:, None]
        & (ow_value < W_out)[:, None]
    )
    tl.store(out_ptrs, acc, mask=out_mask)


class ConvTranspose2d(torch.autograd.Function):
    """Autograd-enabled conv_transpose2d using Triton forward kernel.

    Backward uses PyTorch native conv_transpose2d for correctness and performance.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        opad_h,
        opad_w,
        dil_h,
        dil_w,
        groups,
    ):
        C_in = input.shape[1]
        C_out_per_group = weight.shape[1]
        C_out = C_out_per_group * groups
        kH, kW = weight.shape[2], weight.shape[3]
        N, _, H_in, W_in = input.shape
        H_out = conv_transpose2d_output_size(H_in, kH, stride_h, pad_h, dil_h, opad_h)
        W_out = conv_transpose2d_output_size(W_in, kW, stride_w, pad_w, dil_w, opad_w)

        input = input.contiguous()
        weight = weight.contiguous()

        if bias is None:
            bias_data = torch.zeros(C_out, device=input.device, dtype=input.dtype)
        else:
            bias_data = bias.contiguous()

        # Cast to FP32 for kernel execution
        orig_dtype = input.dtype
        input_f32 = input.float()
        weight_f32 = weight.float()
        bias_f32 = bias_data.float()

        output_f32 = torch.empty(
            (N, C_out, H_out, W_out), device=input.device, dtype=torch.float32
        )

        in_per_group = C_in // groups

        grid = (
            triton.cdiv(N * H_out * W_out, BLOCK_NHWO),
            triton.cdiv(C_out // groups, BLOCK_CO),
            groups,
        )

        logger.debug(
            "GEMS CONV_TRANSPOSE2D FORWARD (triton): grid=%s, config=(%d,%d)",
            grid, BLOCK_NHWO, BLOCK_CO,
        )

        conv_transpose2d_kernel[grid](
            input_f32,
            weight_f32,
            output_f32,
            bias_f32,
            N,
            H_in,
            W_in,
            C_out,
            H_out,
            W_out,
            *input_f32.stride(),
            *weight_f32.stride(),
            *output_f32.stride(),
            in_per_group,
            kH,
            kW,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            groups,
            BLOCK_NHWO=BLOCK_NHWO,
            BLOCK_CO=BLOCK_CO,
            num_warps=4,
            num_stages=2,
        )

        output = output_f32.to(orig_dtype)

        ctx.save_for_backward(input, weight, bias)
        ctx.params = (stride_h, stride_w, pad_h, pad_w, opad_h, opad_w, dil_h, dil_w, groups)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using PyTorch native conv_transpose2d for correctness."""
        input, weight, bias = ctx.saved_tensors
        stride_h, stride_w, pad_h, pad_w, opad_h, opad_w, dil_h, dil_w, groups = ctx.params

        grad_output = grad_output.contiguous()

        input_r = input.detach().requires_grad_(True)
        weight_r = weight.detach().requires_grad_(True)
        bias_r = bias.detach().requires_grad_(True) if bias is not None else None

        with torch.enable_grad():
            out = torch.nn.functional.conv_transpose2d(
                input_r,
                weight_r,
                bias_r,
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                output_padding=(opad_h, opad_w),
                groups=groups,
                dilation=(dil_h, dil_w),
            )
            out.backward(grad_output)

        return (
            input_r.grad,
            weight_r.grad,
            bias_r.grad if bias_r is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Triton implementation of torch.nn.functional.conv_transpose2d.

    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_in, C_out/groups, kH, kW)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride (int or tuple)
        padding: Padding (int or tuple)
        output_padding: Output padding (int or tuple)
        groups: Number of groups
        dilation: Dilation (int or tuple)

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    assert input.ndim == 4, f"Input must be 4D, received shape {input.shape}"
    assert weight.ndim == 4, f"Weights must be 4D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, received shape {bias.shape if bias is not None else None}"

    C_in = input.shape[1]
    C_in_weight = weight.shape[0]
    C_out_per_group = weight.shape[1]
    C_out = C_out_per_group * groups

    assert (
        C_in == C_in_weight
    ), f"Input channels ({C_in}) must match weight dim 0 ({C_in_weight})"
    assert (
        C_in % groups == 0
    ), f"Input channels ({C_in}) must be divisible by groups ({groups})"
    assert C_out_per_group > 0, f"C_out/groups must be positive, got {C_out_per_group}"
    assert (
        bias is None or C_out == bias.shape[0]
    ), f"Bias size ({bias.shape[0]}) must match C_out ({C_out})"

    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (list, tuple)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(output_padding, (list, tuple)):
        opad_h, opad_w = output_padding
    else:
        opad_h = opad_w = output_padding

    if isinstance(dilation, (list, tuple)):
        dil_h, dil_w = dilation
    else:
        dil_h = dil_w = dilation

    N, _, H_in, W_in = input.shape
    kH, kW = weight.shape[2], weight.shape[3]

    if N == 0 or C_in == 0 or H_in == 0 or W_in == 0:
        H_out = conv_transpose2d_output_size(H_in, kH, stride_h, pad_h, dil_h, opad_h)
        W_out = conv_transpose2d_output_size(W_in, kW, stride_w, pad_w, dil_w, opad_w)
        return torch.empty(
            (N, C_out, max(H_out, 0), max(W_out, 0)),
            device=input.device,
            dtype=input.dtype,
        )

    H_out = conv_transpose2d_output_size(H_in, kH, stride_h, pad_h, dil_h, opad_h)
    W_out = conv_transpose2d_output_size(W_in, kW, stride_w, pad_w, dil_w, opad_w)

    assert H_out > 0 and W_out > 0, (
        f"Output size must be positive, got ({H_out}, {W_out}). "
        f"Check stride/padding/dilation/kernel parameters."
    )

    return ConvTranspose2d.apply(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        opad_h,
        opad_w,
        dil_h,
        dil_w,
        groups,
    )
