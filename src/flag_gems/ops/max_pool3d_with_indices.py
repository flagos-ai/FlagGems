import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1
    return output_size


def _parse_pool_params_3d(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 0:
            return default
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_d, kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_d, stride_h, stride_w = _parse_param(
        stride, "stride", default=(kernel_d, kernel_h, kernel_w)
    )
    padding_d, padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0, 0))
    dilation_d, dilation_h, dilation_w = _parse_param(
        dilation, "dilation", default=(1, 1, 1)
    )

    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_d}, {stride_h}, {stride_w})"
        )
    if padding_d < 0 or padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_d}, {padding_h}, {padding_w})"
        )
    if dilation_d <= 0 or dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_d}, {dilation_h}, {dilation_w})"
        )

    return (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    )


def _ensure_5d(x: torch.Tensor):
    if x.dim() == 4:
        return x.unsqueeze(0), True
    if x.dim() == 5:
        return x, False
    raise ValueError(f"Expected 4D or 5D input, but got {x.dim()}D")


@libentry()
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_D": 2, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_D": 4, "BLOCK_H": 4, "BLOCK_W": 4}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_D": 1, "BLOCK_H": 8, "BLOCK_W": 16}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_D": 2, "BLOCK_H": 16, "BLOCK_W": 8}, num_stages=3, num_warps=4
        ),
    ],
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_d_blocks = tl.cdiv(out_d, BLOCK_D)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    rem = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = rem // num_w_blocks
    w_block_idx = rem % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    d_out_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    input_base_ptr = (
        input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    )

    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                d_in = d_out_offsets[:, None, None] * stride_d - padding_d + kd * dilation_d
                h_in = h_out_offsets[None, :, None] * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets[None, None, :] * stride_w - padding_w + kw * dilation_w

                in_mask = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )
                current_idx = (
                    d_in * (in_h * in_w) + h_in * in_w + w_in
                ).to(tl.int64)

                is_new_max = current_val > max_val_acc
                max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_d * out_h * out_w
    output_block_ptr = (
        out_base_ptr
        + d_out_offsets[:, None, None] * out_h * out_w
        + h_out_offsets[None, :, None] * out_w
        + w_out_offsets[None, None, :]
    )
    indices_block_ptr = (
        indices_base_ptr
        + d_out_offsets[:, None, None] * out_h * out_w
        + h_out_offsets[None, :, None] * out_w
        + w_out_offsets[None, None, :]
    )

    out_mask = (
        (d_out_offsets[:, None, None] < out_d)
        & (h_out_offsets[None, :, None] < out_h)
        & (w_out_offsets[None, None, :] < out_w)
    )
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS MAX_POOL3D_WITH_INDICES FORWARD")
    input = input.contiguous()
    input, squeeze_n = _ensure_5d(input)

    params = _parse_pool_params_3d(kernel_size, stride, padding, dilation)
    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d = max_pool3d_output_size(
        in_d, kernel_d, stride_d, padding_d, dilation_d, ceil_mode
    )
    out_h = max_pool3d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool3d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        if squeeze_n:
            return output.squeeze(0), indices.squeeze(0)
        return output, indices

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_d, meta["BLOCK_D"])
        * triton.cdiv(out_h, meta["BLOCK_H"])
        * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    max_pool3d_forward_kernel[grid](
        input,
        output,
        indices,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    )

    if squeeze_n:
        return output.squeeze(0), indices.squeeze(0)
    return output, indices


def max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    output, _ = max_pool3d_with_indices(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    return output


def max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    grad_output, squeeze_n = _ensure_5d(grad_output)
    input, _ = _ensure_5d(input)
    indices, _ = _ensure_5d(indices)

    params = _parse_pool_params_3d(kernel_size, stride, padding, dilation)
    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    ) = params

    grad_input = torch.ops.aten.max_pool3d_with_indices_backward(
        grad_output,
        input,
        (kernel_d, kernel_h, kernel_w),
        (stride_d, stride_h, stride_w),
        (padding_d, padding_h, padding_w),
        (dilation_d, dilation_h, dilation_w),
        ceil_mode,
        indices,
    )
    if squeeze_n:
        return grad_input.squeeze(0)
    return grad_input
