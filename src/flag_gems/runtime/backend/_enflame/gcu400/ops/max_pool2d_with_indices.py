# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# forward kernel 的确定性分派参数：BLOCK 控制每个 tile 的输出元素数，
# NUM_WARPS 按访存型 kernel 调优，GRID_PROGRAMS=0 表示按 tile 数全量铺开。
FORWARD_BLOCK_SIZE = 1024
FORWARD_NUM_WARPS = 4
FORWARD_GRID_PROGRAMS = 0


def max_pool2d_output_size(
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
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.jit
def max_pool2d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    # 输入/输出形状：用于把扁平输出下标还原为 (n, c, oh, ow)
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    total_out_elements,
    # 池化参数保持编译期常量，便于展开窗口循环并优化下标除法
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    # 每个 program 处理一段连续输出元素，跨 (n, c, h, w) 扁平切分；
    # program 数随输出元素数增长，而非随 n*c 通道数增长。NUM_PROGRAMS 小于
    # tile 总数时由核内跨步循环复用固定数量的 program（匹配 SIP 拓扑）。
    num_tiles = tl.cdiv(total_out_elements, BLOCK)
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_PROGRAMS):
        out_offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        out_mask = out_offsets < total_out_elements

        ow = out_offsets % out_w
        oh = (out_offsets // out_w) % out_h
        nc_idx = out_offsets // (out_h * out_w)
        n_idx = nc_idx // in_c
        c_idx = nc_idx % in_c
        input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

        dtype = input_ptr.type.element_ty
        min_val = get_dtype_min(dtype)
        max_val_acc = tl.full((BLOCK,), min_val, dtype=dtype)
        max_idx_acc = tl.full((BLOCK,), -1, dtype=tl.int32)

        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                h_in = oh * stride_h - padding_h + kh * dilation_h
                w_in = ow * stride_w - padding_w + kw * dilation_w
                in_mask = (
                    out_mask & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
                )
                input_offsets = h_in * in_stride_h + w_in * in_stride_w
                current_val = tl.load(
                    input_base_ptr + input_offsets, mask=in_mask, other=min_val
                )
                current_idx = h_in * in_w + w_in
                is_new_max = current_val > max_val_acc
                max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

        tl.store(output_ptr + out_offsets, max_val_acc, mask=out_mask)
        tl.store(indices_ptr + out_offsets, max_idx_acc.to(tl.int64), mask=out_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 8}, num_warps=2),
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 16}, num_warps=2),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 8}, num_warps=2),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 32}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 32}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 64}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 64, "BLOCK_IN_W": 16}, num_warps=8),
    ],
    key=["in_h", "in_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_h,
    in_w,
    out_h,
    out_w,
    num_nc,
    # Strides for grad_output/indices
    out_stride_nc,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Tiling parameters
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    # Flattened (nc, spatial-block) tile space split across a 2D grid so that
    # neither grid dim exceeds the 65535 limit (e.g. N*C = 128*512 = 65536
    # alone overflows grid.x). Must match the host grid() below.
    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    num_h_blocks = tl.cdiv(in_h, BLOCK_IN_H)
    num_spatial = num_h_blocks * num_w_blocks
    total_blocks = num_nc * num_spatial
    grid_x = tl.minimum(65535, total_blocks)
    linear_pid = tl.program_id(0) + tl.program_id(1) * grid_x
    if linear_pid >= total_blocks:
        return

    nc_idx = linear_pid // num_spatial
    pid_hw = linear_pid % num_spatial
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    current_input_flat_idx = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    grad_acc = tl.zeros((BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            numerator_h = h_in_offsets[:, None] + padding_h - kh * dilation_h
            numerator_w = w_in_offsets[None, :] + padding_w - kw * dilation_w

            valid_map_mask = (numerator_h % stride_h == 0) & (
                numerator_w % stride_w == 0
            )
            h_out = numerator_h // stride_h
            w_out = numerator_w // stride_w
            out_bounds_mask = (
                (h_out >= 0) & (h_out < out_h) & (w_out >= 0) & (w_out < out_w)
            )
            load_mask = valid_map_mask & out_bounds_mask

            safe_h_out = tl.where(load_mask, h_out, 0)
            safe_w_out = tl.where(load_mask, w_out, 0)
            out_offsets = safe_h_out * out_stride_h + safe_w_out

            indices_block = tl.load(
                indices_base_ptr + out_offsets, mask=load_mask, other=-1
            )
            match_mask = indices_block == current_input_flat_idx

            grad_block = tl.load(
                grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
            )
            grad_acc += grad_block

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_h * in_w
    grad_input_offsets = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    store_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def _parse_pool_params(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_h, stride_w = _parse_param(stride, "stride", default=(kernel_h, kernel_w))
    padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0))
    dilation_h, dilation_w = _parse_param(dilation, "dilation", default=(1, 1))

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_h}, {stride_w})"
        )
    if padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_h}, {padding_w})"
        )
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_h}, {dilation_w})"
        )

    return (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


def max_pool2d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_ENFLAME MAX_POOL2D_WITH_INDICES")
    input = input.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h = max_pool2d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool2d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    total_out_elements = output.numel()
    # 1D 扁平输出分块：BLOCK 由迭代调优确定；GRID_PROGRAMS>0 时固定 program 数
    # 并在核内跨步循环，避免大 shape 下 program 调度开销。
    BLOCK = FORWARD_BLOCK_SIZE
    num_tiles = triton.cdiv(total_out_elements, BLOCK)
    num_programs = (
        num_tiles
        if FORWARD_GRID_PROGRAMS <= 0
        else min(FORWARD_GRID_PROGRAMS, num_tiles)
    )
    grid = (num_programs,)
    max_pool2d_forward_flat_kernel[grid](
        input,
        output,
        indices,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        total_out_elements,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        BLOCK=BLOCK,
        NUM_PROGRAMS=num_programs,
        num_warps=FORWARD_NUM_WARPS,
    )

    return output, indices


def max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    logger.debug("GEMS_ENFLAME MAX_POOL2D_BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    # The kernel writes every input position, so no zero-init is needed; and
    # accumulating in fp32 inside the kernel lets us store directly in the
    # output dtype, dropping both the memset pass and the final dtype
    # conversion pass of the original zeros_like(fp32) + .to(dtype) flow.
    grad_input = torch.empty_like(input, dtype=grad_output.dtype)

    if grad_input.numel() == 0:
        return grad_input

    def grid(meta):
        num_h_blocks = triton.cdiv(in_h, meta["BLOCK_IN_H"])
        num_w_blocks = triton.cdiv(in_w, meta["BLOCK_IN_W"])
        num_spatial = num_h_blocks * num_w_blocks
        total = in_n * in_c * num_spatial
        grid_x = min(65535, total)
        grid_y = triton.cdiv(total, grid_x)
        return (grid_x, grid_y)

    out_stride_nc = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        in_n * in_c,
        out_stride_nc,
        out_stride_h,
        out_stride_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )

    return grad_input


def max_pool2d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
):
    """Wrapper matching the aten::max_pool2d_with_indices_backward schema."""
    logger.debug("GEMS_ENFLAME MAX_POOL2D_WITH_INDICES_BACKWARD")
    return max_pool2d_backward(
        grad_output, self, indices, kernel_size, stride, padding, dilation, ceil_mode
    )
