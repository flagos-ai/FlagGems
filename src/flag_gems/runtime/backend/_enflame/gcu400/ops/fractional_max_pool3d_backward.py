# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``fractional_max_pool3d_backward``。

kernel 主体逐字复刻公共实现（保证 scatter 语义与数值一致），唯一改动是
去掉 ``@triton.autotune`` 的 5 组候选 config：在 GCU 上每组 config 都需要
独立冷编译并试跑，改为 host 侧确定性分档的固定 tile，消除这部分编译开销。
公共实现文件未修改。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；编译期配置，不改变算子语义。
BLOCK_H = 4
BLOCK_W = 8
NUM_WARPS = 4
NUM_STAGES = 1
SMALL_OUT_ELEMENTS = 262144
SMALL_BLOCK_H = 4
SMALL_BLOCK_W = 8
USE_FP32_ACC = 1
CONTIGUOUS_INPUTS = 1
USE_DEVICE_CONTEXT = 0
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def _fractional_max_pool3d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_n,
    in_c,
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
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    IS_FP32_ACC: tl.constexpr,
):
    """按 (nc, d*h块*w块) 二维 grid 把梯度 scatter 回输入位置。"""
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_hw_blocks = num_h_blocks * num_w_blocks

    d_idx = pid_dhw // num_hw_blocks
    hw_remainder = pid_dhw % num_hw_blocks
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_out_offsets < out_h
    w_mask = w_out_offsets < out_w
    out_mask = h_mask[:, None] & w_mask[None, :]

    out_base = pid_nc * out_stride_nc
    out_offsets = (
        out_base
        + d_idx * out_stride_d
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    grad_vals = tl.load(grad_output_ptr + out_offsets, mask=out_mask, other=0.0)
    idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)

    in_nc_offset = pid_nc * in_d * in_h * in_w
    in_offsets = in_nc_offset + idx_vals
    valid_mask = out_mask & (idx_vals >= 0)
    if IS_FP32_ACC:
        acc = grad_vals.to(tl.float32)
    else:
        acc = grad_vals
    tl.atomic_add(grad_input_ptr + in_offsets, acc, mask=valid_mask)


def fractional_max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    output_size,
    indices: torch.Tensor,
):
    logger.debug("GEMS_GCU400 FRACTIONAL_MAX_POOL3D_BACKWARD")
    if isinstance(output_size, int):
        out_d = out_h = out_w = output_size
    else:
        out_d, out_h, out_w = output_size

    in_n, in_c, in_d, in_h, in_w = input.shape
    if CONTIGUOUS_INPUTS:
        grad_output = grad_output.contiguous()
        indices = indices.contiguous()

    acc_dtype = torch.float32 if USE_FP32_ACC else grad_output.dtype
    grad_input = torch.zeros_like(input, dtype=acc_dtype)
    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    if out_d * out_h * out_w <= SMALL_OUT_ELEMENTS:
        block_h, block_w = SMALL_BLOCK_H, SMALL_BLOCK_W
    else:
        block_h, block_w = BLOCK_H, BLOCK_W

    grid = (
        in_n * in_c,
        out_d * triton.cdiv(out_h, block_h) * triton.cdiv(out_w, block_w),
    )
    out_stride_nc = out_d * out_h * out_w
    out_stride_d = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    device_context = (
        torch_device_fn.device(grad_output.device)
        if USE_DEVICE_CONTEXT
        else nullcontext()
    )
    with device_context:
        _fractional_max_pool3d_backward_kernel[grid](
            grad_output,
            indices,
            grad_input,
            in_n,
            in_c,
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
            block_h,
            block_w,
            USE_FP32_ACC == 1,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if USE_FP32_ACC:
        return grad_input.to(grad_output.dtype)
    return grad_input
