"""
Triton implementation of conv_transpose2d operator.

Supports all PyTorch parameters: stride, padding, output_padding, groups, dilation.
Compatible with FlagTree backends (Iluvatar, Ascend, etc.) via standard Triton APIs.

Weight layout follows PyTorch convention: (C_in, C_out/groups, kH, kW).

Algorithm:
- stride=1: conv2d-based approach (transform weight, call conv2d_forward_kernel)
- stride>1: Direct implicit GEMM kernel with conv2d-like loop structure
  (no input dilation, only iterates over valid kernel positions)

Backward pass:
    grad_input  = conv2d(grad_output, weight^T_flipped, stride=1,
                         padding=dil*(kH-1)-pad, dilation=stride)
    grad_weight = correlation(input, grad_output) with transposed coord mapping
    grad_bias   = grad_output.sum((0,2,3))
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
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


# =============================================================================
# Direct implicit GEMM kernel (for stride > 1)
# =============================================================================


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv_transpose2d_forward"),
    key=[
        "N",
        "H_in",
        "W_in",
        "C_out",
        "H_out",
        "W_out",
        "kH",
        "kW",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "groups",
    ],
)
@triton.jit
def conv_transpose2d_direct_kernel(
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
    C_in: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    """Implicit GEMM kernel for conv_transpose2d (stride > 1).

    Loads from original (non-dilated) input. Iterates only over valid
    kernel positions where (kh % stride_h == 0) and (kw % stride_w == 0).
    Same tiling structure as conv2d_forward_kernel for optimal compiler optimization.
    """
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    ni_ho_wo_offset = pid_nhw * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // W_out
    n_value = ni_ho_offset // H_out
    oh_value = ni_ho_offset % H_out
    ow_value = ni_ho_wo_offset % W_out

    out_per_group = C_out // groups
    co_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_global = pid_group * out_per_group + co_offset
    in_per_group = C_in // groups

    input_pointer += (
        input_n_stride * n_value + input_c_stride * pid_group * in_per_group
    )[:, None]
    weight_pointer += (
        weight_co_stride * co_offset + weight_co_stride * pid_group * out_per_group
    )[None, :]

    acc = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    BLOCK_CI_COUNT = (in_per_group + BLOCK_CI - 1) // BLOCK_CI
    valid_kh_count: tl.constexpr = (kH + stride_h - 1) // stride_h
    valid_kw_count: tl.constexpr = (kW + stride_w - 1) // stride_w
    valid_khw_count: tl.constexpr = valid_kh_count * valid_kw_count

    for hwc in range(valid_khw_count * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        kh_i = hw // valid_kw_count
        kw_i = hw % valid_kw_count
        kh = kh_i * stride_h
        kw = kw_i * stride_w

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        ih_num = oh_value + pad_h - kh * dil_h
        iw_num = ow_value + pad_w - kw * dil_w
        ih = ih_num // stride_h
        iw = iw_num // stride_w

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_h_stride * ih)[:, None]
            + (input_w_stride * iw)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_ci_stride * input_c_offset)[:, None]
            + (weight_h_stride * kh)
            + (weight_w_stride * kw)
        )

        input_mask = (
            (n_value < N)[:, None]
            & (input_c_offset < in_per_group)[None, :]
            & (ih_num >= 0)[:, None]
            & (ih < H_in)[:, None]
            & (iw_num >= 0)[:, None]
            & (iw < W_in)[:, None]
        )
        weight_mask = (input_c_offset < in_per_group)[:, None] & (
            co_offset < out_per_group
        )[None, :]

        inp_val = tl.load(curr_input_pointer, mask=input_mask).to(tl.float32)
        w_val = tl.load(curr_weight_pointer, mask=weight_mask).to(tl.float32)

        acc += tl.dot(inp_val, w_val, allow_tf32=False)

    bias_ptrs = bias_pointer + co_global[None, :]
    bias_mask = (co_offset < out_per_group)[None, :]
    bias_val = tl.load(bias_ptrs, mask=bias_mask).to(tl.float32)
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


# =============================================================================
# Autograd wrapper
# =============================================================================


class ConvTranspose2d(torch.autograd.Function):
    """Autograd-enabled conv_transpose2d using Triton forward kernel.

    stride=1: conv2d-based approach (reuses optimized conv2d_forward_kernel)
    stride>1: Direct implicit GEMM kernel (no input dilation)

    Backward uses Conv2d.apply for grad_input (reusing FlagGems conv2d Triton kernel)
    and PyTorch ops for grad_weight/grad_bias.
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

        output = torch.empty(
            (N, C_out, H_out, W_out), device=input.device, dtype=input.dtype
        )

        if stride_h == 1 and stride_w == 1:
            # --- stride=1: conv2d-based approach (cuDNN-style, already fast) ---
            logger.debug("GEMS CONV_TRANSPOSE2D FORWARD (via conv2d, stride=1)")
            weight_flipped = weight.flip([2, 3]).contiguous()
            if groups == 1:
                weight_conv2d = weight_flipped.permute(1, 0, 2, 3).contiguous()
            else:
                C_in_per_group = C_in // groups
                w = weight_flipped.reshape(
                    groups, C_in_per_group, C_out_per_group, kH, kW
                )
                w = w.permute(0, 2, 1, 3, 4)
                weight_conv2d = w.reshape(C_out, C_in_per_group, kH, kW).contiguous()

            conv2d_pad_h = dil_h * (kH - 1) - pad_h
            conv2d_pad_w = dil_w * (kW - 1) - pad_w

            from flag_gems.ops.conv2d import conv2d_forward_kernel

            grid = lambda META: (
                triton.cdiv(N * H_out * W_out, META["BLOCK_NI_HO_WO"]),
                triton.cdiv(C_out // groups, META["BLOCK_CO"]),
                groups,
            )

            conv2d_forward_kernel[grid](
                input,
                weight_conv2d,
                output,
                bias_data,
                N,
                H_in,
                W_in,
                C_out,
                H_out,
                W_out,
                *input.stride(),
                *weight_conv2d.stride(),
                *output.stride(),
                weight_conv2d.shape[1],
                kH,
                kW,
                1,
                1,
                conv2d_pad_h,
                conv2d_pad_w,
                dil_h,
                dil_w,
                groups=groups,
            )
        else:
            # --- stride>1: direct implicit GEMM (no input dilation) ---
            logger.debug("GEMS CONV_TRANSPOSE2D FORWARD (direct, stride>1)")

            in_per_group = C_in // groups

            grid = lambda META: (
                triton.cdiv(N * H_out * W_out, META["BLOCK_NI_HO_WO"]),
                triton.cdiv(C_out // groups, META["BLOCK_CO"]),
                groups,
            )

            conv_transpose2d_direct_kernel[grid](
                input,
                weight,
                output,
                bias_data,
                N,
                H_in,
                W_in,
                C_out,
                H_out,
                W_out,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                in_per_group,
                kH,
                kW,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
                groups=groups,
            )

        ctx.save_for_backward(input, weight, bias)
        ctx.params = (stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups)
        ctx.kernel_size = (kH, kW)
        ctx.input_shape = (N, C_in, H_in, W_in)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS CONV_TRANSPOSE2D BACKWARD")
        input, weight, bias = ctx.saved_tensors
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups = ctx.params
        kH, kW = ctx.kernel_size
        N, C_in, H_in, W_in = ctx.input_shape
        C_out_per_group = weight.shape[1]
        C_out = C_out_per_group * groups

        grad_output = grad_output.contiguous()
        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]

        # --- grad_input via conv2d ---
        weight_flipped = weight.flip([2, 3]).contiguous()
        if groups == 1:
            weight_conv2d = weight_flipped.permute(1, 0, 2, 3).contiguous()
        else:
            C_in_per_group = C_in // groups
            w = weight_flipped.reshape(groups, C_in_per_group, C_out_per_group, kH, kW)
            w = w.permute(0, 2, 1, 3, 4)
            weight_conv2d = w.reshape(C_out, C_in_per_group, kH, kW).contiguous()

        conv2d_pad_h = dil_h * (kH - 1) - pad_h
        conv2d_pad_w = dil_w * (kW - 1) - pad_w

        from flag_gems.ops.conv2d import Conv2d

        grad_input = Conv2d.apply(
            grad_output,
            weight_conv2d,
            None,
            1,
            (conv2d_pad_h, conv2d_pad_w),
            (dil_h, dil_w),
            groups,
        )

        # --- grad_weight ---
        grad_weight = torch.zeros_like(weight)

        for g in range(groups):
            ci_start = g * (C_in // groups)
            co_start = g * C_out_per_group
            ci_end = ci_start + C_in // groups
            co_end = co_start + C_out_per_group

            inp_g = input[:, ci_start:ci_end, :, :]
            go_g = grad_output[:, co_start:co_end, :, :]

            for kh in range(kH):
                for kw in range(kW):
                    oh_indices = []
                    ih_map = {}
                    for oh in range(H_out):
                        ih_num = oh + pad_h - kh * dil_h
                        if ih_num < 0:
                            continue
                        if stride_h > 1 and ih_num % stride_h != 0:
                            continue
                        ih = ih_num // stride_h
                        if ih >= H_in:
                            continue
                        oh_indices.append(oh)
                        ih_map[oh] = ih

                    ow_indices = []
                    iw_map = {}
                    for ow in range(W_out):
                        iw_num = ow + pad_w - kw * dil_w
                        if iw_num < 0:
                            continue
                        if stride_w > 1 and iw_num % stride_w != 0:
                            continue
                        iw = iw_num // stride_w
                        if iw >= W_in:
                            continue
                        ow_indices.append(ow)
                        iw_map[ow] = iw

                    if not oh_indices or not ow_indices:
                        continue

                    ih_vals = [ih_map[oh] for oh in oh_indices]
                    iw_vals = [iw_map[ow] for ow in ow_indices]

                    oh_t = torch.tensor(oh_indices, device=input.device)
                    ow_t = torch.tensor(ow_indices, device=input.device)
                    ih_t = torch.tensor(ih_vals, device=input.device)
                    iw_t = torch.tensor(iw_vals, device=input.device)

                    inp_selected = inp_g[:, :, ih_t[:, None], iw_t[None, :]]
                    go_selected = go_g[:, :, oh_t[:, None], ow_t[None, :]]

                    grad_weight[
                        ci_start:ci_end, co_start:co_end, kh, kw
                    ] += torch.einsum(
                        "nchw,nchw->nc", inp_selected.flatten(2), go_selected.flatten(2)
                    )

        # --- grad_bias ---
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return (
            grad_input,
            grad_weight,
            grad_bias,
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
