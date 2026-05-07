import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """Compute one spatial dimension of the 3-D max-pool output."""
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1
    return output_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_stages=2, num_warps=8),
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
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Forward kernel for 3-D max pooling.

    Grid: (N * C, num_d_blocks * num_h_blocks * num_w_blocks)
    where num_h_blocks = cdiv(out_h, BLOCK_H),
          num_w_blocks = cdiv(out_w, BLOCK_W).
    The depth dimension is iterated inside the kernel via a loop
    over num_d_blocks output depth positions.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    hw_remainder = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    d_out = d_block_idx

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kd in tl.static_range(0, kernel_d):
        d_in = d_out * stride_d - padding_d + kd * dilation_d
        d_valid = (d_in >= 0) & (d_in < in_d)
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
                in_mask = (
                    d_valid & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
                )
                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )
                # Flat index in (D, H, W) space
                current_idx = d_in * in_h * in_w + h_in * in_w + w_in

                is_new_max = current_val > max_val_acc
                max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_spatial = out_h * out_w
    out_base_offset = pid_nc * out_d * out_spatial + d_out * out_spatial
    out_base_ptr = output_ptr + out_base_offset
    indices_base_ptr = indices_ptr + out_base_offset
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 32}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 32}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 64}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 64, "BLOCK_IN_W": 16}, num_warps=8),
    ],
    key=[
        "in_d",
        "in_h",
        "in_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def max_pool3d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Strides for grad_output/indices (contiguous layout: NC, D, H, W)
    out_stride_nc,
    out_stride_d,
    out_stride_h,
    out_stride_w,
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
    # Tiling parameters
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    """Backward kernel for 3-D max pooling.

    Grid: (N * C, num_d_in * num_h_blocks * num_w_blocks)
    For each input (d, h, w) position, iterate over all kernel
    offsets to find which output positions could have selected it,
    then accumulate the gradient.
    """
    nc_idx = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(in_h, BLOCK_IN_H)
    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)

    d_in_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    hw_remainder = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    # Flat index of current input position in (D, H, W) space
    current_input_flat_idx = (
        d_in_idx * in_h * in_w + h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    )
    grad_acc = tl.zeros((BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kd in tl.static_range(0, kernel_d):
        numerator_d = d_in_idx + padding_d - kd * dilation_d
        valid_d = numerator_d % stride_d == 0
        d_out = numerator_d // stride_d
        d_bounds = (d_out >= 0) & (d_out < out_d)
        d_valid = valid_d & d_bounds

        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                numerator_h = h_in_offsets[:, None] + padding_h - kh * dilation_h
                numerator_w = w_in_offsets[None, :] + padding_w - kw * dilation_w

                valid_map_mask = (
                    d_valid
                    & (numerator_h % stride_h == 0)
                    & (numerator_w % stride_w == 0)
                )
                h_out = numerator_h // stride_h
                w_out = numerator_w // stride_w
                out_bounds_mask = (
                    (h_out >= 0) & (h_out < out_h) & (w_out >= 0) & (w_out < out_w)
                )
                load_mask = valid_map_mask & out_bounds_mask

                safe_h_out = tl.where(load_mask, h_out, 0)
                safe_w_out = tl.where(load_mask, w_out, 0)
                safe_d_out = tl.where(load_mask, d_out, 0)
                out_offsets = (
                    safe_d_out * out_stride_d + safe_h_out * out_stride_h + safe_w_out
                )

                indices_block = tl.load(
                    indices_base_ptr + out_offsets, mask=load_mask, other=-1
                )
                match_mask = indices_block == current_input_flat_idx

                grad_block = tl.load(
                    grad_output_base_ptr + out_offsets,
                    mask=match_mask,
                    other=0.0,
                )
                grad_acc += grad_block

    in_spatial = in_h * in_w
    grad_input_base_ptr = grad_input_ptr + nc_idx * in_d * in_spatial
    grad_input_offsets = (
        d_in_idx * in_spatial + h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    )
    store_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Compute 3-D max pooling, returning (output, indices).

    Indices are flat offsets into the (D, H, W) spatial volume of the input.
    """
    logger.debug("GEMS MAX_POOL3D_WITH_INDICES")
    input = input.contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d = pool3d_output_size(in_d, kd, sd, pd, dd, ceil_mode)
    out_h = pool3d_output_size(in_h, kh, sh, ph, dh, ceil_mode)
    out_w = pool3d_output_size(in_w, kw, sw, pw, dw, ceil_mode)

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    grid = lambda meta: (
        in_n * in_c,
        out_d
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
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
    )

    return output, indices


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
    """Backward pass for 3-D max pooling."""
    logger.debug("GEMS MAX_POOL3D BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    out_spatial = out_h * out_w
    out_stride_nc = out_d * out_spatial
    out_stride_d = out_spatial
    out_stride_h = out_w
    out_stride_w = 1

    grid = lambda meta: (
        in_n * in_c,
        in_d
        * triton.cdiv(in_h, meta["BLOCK_IN_H"])
        * triton.cdiv(in_w, meta["BLOCK_IN_W"]),
    )

    max_pool3d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        out_stride_nc,
        out_stride_d,
        out_stride_h,
        out_stride_w,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
    )

    return grad_input.to(grad_output.dtype)
