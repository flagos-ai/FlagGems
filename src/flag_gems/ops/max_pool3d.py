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
    """Calculate output size for 3D max pooling."""
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
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 8, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 8, "BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 32, "BLOCK_W": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
    ],
    key=["out_d", "out_h", "out_w", "kernel_d", "kernel_h", "kernel_w"],
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
    return_indices: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Forward kernel for 3D max pooling."""
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)
    
    # Calculate block indices
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    
    w_block_idx = pid_dhw % num_w_blocks
    h_block_idx = (pid_dhw // num_w_blocks) % num_h_blocks
    d_block_idx = pid_dhw // (num_w_blocks * num_h_blocks)
    
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Output offsets
    d_out_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # 1. 获取基础类型，确保一致性
    dtype = input_ptr.dtype.element_ty
    
    # 2. 显式构造最小值，避免使用复杂的外部函数
    # 直接使用 tl.where 或特定的极小值
    inf_val = float("-inf")
    min_val = tl.full([], inf_val, dtype=dtype)

    # 3. 初始化累加器
    # 建议：如果 BLOCK_D, H, W 很大，确保你的 num_warps 设置足够
    max_val_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), inf_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), -1, dtype=tl.int32) # 通常 int32 足够，省寄存器
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Iterate over kernel
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                d_in = d_out_offsets[:, None, None] * stride_d - padding_d + kd * dilation_d
                h_in = h_out_offsets[None, :, None] * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets[None, None, :] * stride_w - padding_w + kw * dilation_w
                
                in_mask = (d_in >= 0) & (d_in < in_d) & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
                input_offset = d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                # 【补上这一行】从显存读取数据，注意 other 必须匹配 dtype
                current_val = tl.load(input_base_ptr + input_offset, mask=in_mask, other=min_val)

                # 计算当前展平后的索引，用于 return_indices
                # 建议显式转为 int32 以匹配 max_idx_acc
                current_idx = (d_in * in_h * in_w + h_in * in_w + w_in).to(tl.int32)
                
                # 1. 显式转为 fp32 (确保不是 i16)
                curr_val_f32 = current_val.to(tl.float32)
                max_acc_f32 = max_val_acc.to(tl.float32)

                # 2. 进行比较 (这里生成的是布尔掩码)
                is_new_max = curr_val_f32 > max_acc_f32

                # 3. 【关键】使用 tl.where 时，确保所有参数类型严格匹配
                # 如果 current_val 是 fp16，max_val_acc 也必须是 fp16
                max_val_acc = tl.where(is_new_max, current_val.to(dtype), max_val_acc.to(dtype))

                # 4. 索引更新保持 int32/int64 一致
                max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    # Store output
    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w
    d_out_offs = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offs = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offs = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    
    output_block_ptr = (
        out_base_ptr 
        + d_out_offs[:, None, None] * out_h * out_w 
        + h_out_offs[None, :, None] * out_w 
        + w_out_offs[None, None, :]
    )

    out_mask = (d_out_offs[:, None, None] < out_d) & (h_out_offs[None, :, None] < out_h) & (w_out_offs[None, None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    
    if return_indices:
        indices_base_ptr = indices_ptr + pid_nc * out_d * out_h * out_w
        indices_block_ptr = (
            indices_base_ptr 
            + d_out_offs[:, None, None] * out_h * out_w 
            + h_out_offs[None, :, None] * out_w 
            + w_out_offs[None, None, :]
        )
        tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 8, "BLOCK_W": 8}, num_warps=4),
        triton.Config({"BLOCK_D": 8, "BLOCK_H": 8, "BLOCK_W": 8}, num_warps=4),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 16, "BLOCK_W": 16}, num_warps=8),
        triton.Config({"BLOCK_D": 8, "BLOCK_H": 16, "BLOCK_W": 8}, num_warps=4),
    ],
    key=["in_d", "in_h", "in_w", "kernel_d", "kernel_h", "kernel_w"],
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
    # Strides for grad_output/indices
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
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Backward kernel for 3D max pooling."""
    nc_idx = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    num_h_blocks = tl.cdiv(in_h, BLOCK_H)
    
    w_block_idx = pid_dhw % num_w_blocks
    h_block_idx = (pid_dhw // num_w_blocks) % num_h_blocks
    d_block_idx = pid_dhw // (num_w_blocks * num_h_blocks)

    d_in_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    current_input_flat_idx = (
        d_in_offsets[:, None, None] * in_h * in_w 
        + h_in_offsets[None, :, None] * in_w 
        + w_in_offsets[None, None, :]
    )
    grad_acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                numerator_d = d_in_offsets[:, None, None] + padding_d - kd * dilation_d
                numerator_h = h_in_offsets[None, :, None] + padding_h - kh * dilation_h
                numerator_w = w_in_offsets[None, None, :] + padding_w - kw * dilation_w

                valid_map_mask = (
                    (numerator_d % stride_d == 0) 
                    & (numerator_h % stride_h == 0) 
                    & (numerator_w % stride_w == 0)
                )
                d_out = numerator_d // stride_d
                h_out = numerator_h // stride_h
                w_out = numerator_w // stride_w
                
                out_bounds_mask = (
                    (d_out >= 0) & (d_out < out_d) 
                    & (h_out >= 0) & (h_out < out_h) 
                    & (w_out >= 0) & (w_out < out_w)
                )
                load_mask = valid_map_mask & out_bounds_mask

                safe_d_out = tl.where(load_mask, d_out, 0)
                safe_h_out = tl.where(load_mask, h_out, 0)
                safe_w_out = tl.where(load_mask, w_out, 0)
                out_offsets = (
                    safe_d_out * out_stride_d 
                    + safe_h_out * out_stride_h 
                    + safe_w_out * out_stride_w
                )

                indices_block = tl.load(
                    indices_base_ptr + out_offsets, mask=load_mask, other=-1
                )
                match_mask = indices_block == current_input_flat_idx

                grad_block = tl.load(
                    grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
                )
                grad_acc += grad_block

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_d * in_h * in_w
    grad_input_offsets = (
        d_in_offsets[:, None, None] * in_h * in_w 
        + h_in_offsets[None, :, None] * in_w 
        + w_in_offsets[None, None, :]
    )
    store_mask = (
        (d_in_offsets[:, None, None] < in_d) 
        & (h_in_offsets[None, :, None] < in_h) 
        & (w_in_offsets[None, None, :] < in_w)
    )
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def _parse_pool_params(kernel_size, stride, padding, dilation):
    """Parse pooling parameters to handle different input formats."""
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_d, kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_d, stride_h, stride_w = _parse_param(
        stride, "stride", default=(kernel_d, kernel_h, kernel_w)
    )
    padding_d, padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0, 0))
    dilation_d, dilation_h, dilation_w = _parse_param(dilation, "dilation", default=(1, 1, 1))

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


def max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    3D max pooling operation.
    
    Args:
        input: Input tensor of shape (N, C, D, H, W)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to all three sides
        dilation: Spacing between kernel elements
        ceil_mode: When True, use ceil instead of floor to compute output shape
        return_indices: When True, return the max indices along with the outputs
    
    Returns:
        output: Output tensor
        indices: (optional) Indices of max values
    """
    logger.debug("GEMS MAX_POOL3D FORWARD")
    input = input.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
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
    
    if return_indices:
        indices = torch.empty(
            (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=torch.int64
        )
    else:
        indices = torch.empty(0, device=input.device, dtype=torch.int64)

    if output.numel() == 0:
        if return_indices:
            return output, indices
        return output

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
        return_indices=return_indices,
    )

    if return_indices:
        return output, indices
    return output


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """
    3D max pooling with indices.
    
    This is a convenience function that always returns indices.
    """
    return max_pool3d(
        input, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True
    )


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
    """
    Backward pass for 3D max pooling.
    
    Args:
        grad_output: Gradient of the output
        input: Original input tensor
        indices: Indices from forward pass
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding added to all three sides
        dilation: Spacing between kernel elements
        ceil_mode: When True, use ceil instead of floor to compute output shape
    
    Returns:
        grad_input: Gradient with respect to input
    """
    logger.debug("GEMS MAX_POOL3D BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
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
    out_d, out_h, out_w = grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_d, meta["BLOCK_D"]) 
        * triton.cdiv(in_h, meta["BLOCK_H"]) 
        * triton.cdiv(in_w, meta["BLOCK_W"]),
    )

    out_stride_nc = out_d * out_h * out_w
    out_stride_d = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

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

    return grad_input.to(grad_output.dtype)