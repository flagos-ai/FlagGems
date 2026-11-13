# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``upsample_nearest1d_backward``：可调一维 launch 配置。

公共实现把 ``block_size`` 硬编码为 256 并固定 ``num_warps=4``。GCU400 的
SIP 数量与向量宽度和其它后端不同，这里把这两个 launch 参数提取为编译期
可调常量，由迭代探针按本后端实测选出，计算逻辑与 dtype/stride 处理完全
沿用公共 Triton kernel，不改变算子语义。
"""

import importlib
import logging
import math
from typing import Optional, Sequence

import torch
import triton

from flag_gems.runtime import torch_device_fn

# flag_gems.ops 的 __init__ 用同名函数遮蔽了子模块，按完整路径取模块。
_public = importlib.import_module("flag_gems.ops.upsample_nearest1d_backward")

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；仅影响 kernel 的 launch 形态。
BLOCK_SIZE = 256
NUM_WARPS = 4
FORCE_PUBLIC = 0
# --- TUNING PARAMS END ---


def _impl(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float],
    grad_input: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, channels, input_w, output_w = _public._validate_args(
        grad_output, output_size, input_size
    )
    if grad_input is None:
        grad_input = torch.empty(
            (batch, channels, input_w),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
    else:
        if grad_input.device != grad_output.device:
            raise RuntimeError(
                f"Expected grad_input on {grad_output.device}, "
                f"but got {grad_input.device}"
            )
        if grad_input.dtype != grad_output.dtype:
            raise RuntimeError(
                f"Expected grad_input dtype {grad_output.dtype}, "
                f"but got {grad_input.dtype}"
            )
        grad_input.resize_((batch, channels, input_w))

    if grad_input.numel() == 0:
        return grad_input

    backward_scale = _public._as_float32(
        float(scales) if scales is not None and scales > 0.0 else output_w / input_w
    )
    max_contributors = min(output_w, max(1, math.ceil(backward_scale) + 1))
    batch_loop = batch <= 32
    numel = channels * input_w if batch_loop else grad_input.numel()

    with torch_device_fn.device(grad_output.device):
        _public._upsample_nearest1d_backward_kernel[(triton.cdiv(numel, BLOCK_SIZE),)](
            grad_output,
            grad_input,
            numel,
            channels,
            output_w,
            input_w,
            *grad_output.stride(),
            *grad_input.stride(),
            backward_scale,
            BATCH=batch,
            BATCH_LOOP=batch_loop,
            IS_FP64=grad_output.dtype == torch.float64,
            IS_UINT8=grad_output.dtype == torch.uint8,
            MAX_CONTRIBUTORS=max_contributors,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
    return grad_input


def upsample_nearest1d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_GCU400 UPSAMPLE_NEAREST1D_BACKWARD")
    if FORCE_PUBLIC:
        return _public.upsample_nearest1d_backward(
            grad_output, output_size, input_size, scales
        )
    return _impl(grad_output, output_size, input_size, scales, None)


def upsample_nearest1d_backward_grad_input(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    scales: Optional[float] = None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_GCU400 UPSAMPLE_NEAREST1D_BACKWARD_GRAD_INPUT")
    if FORCE_PUBLIC:
        return _public.upsample_nearest1d_backward_grad_input(
            grad_output, output_size, input_size, scales, grad_input=grad_input
        )
    return _impl(grad_output, output_size, input_size, scales, grad_input)
