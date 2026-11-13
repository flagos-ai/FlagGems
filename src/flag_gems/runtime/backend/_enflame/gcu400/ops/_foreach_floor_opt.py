# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_floor`` / ``_foreach_floor_`` 大 shape 路径。

公共 foreach unary 执行器已经具备 int64 metadata 缓存、kernel launch replay
以及 p4 向量化打包；自建一维融合 kernel 每次重建 metadata 且缺少对齐提示，
实测只有公共路径的 40%。因此这里只把「大张量」分支隔离到本文件：沿用公共
``@triton.jit`` kernel 与缓存工具，但使用 floor 专属的 BLOCK 起点、p4 开关
与 fp32 提升策略，避免原地改动共享的 ``_foreach_unary.py`` 影响同族算子。

小张量与非浮点/非连续等形态直接回退公共 ``apply_unary``，dtype policy 与
异常语义完全不变；主计算路径始终是 Triton kernel。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import (
    _device_meta,
    _launch_executor_kernel,
    _pack_ok,
    foreach_unary_kernel,
    foreach_unary_kernel_p4,
    same_dtype,
    tl_dtype,
)

from . import _foreach_unary as shared_unary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 迭代驱动按轮改写；全部是启动期配置，不改变算子语义。
LARGE_BLOCK_START = 65536
LARGE_TENSOR_STRATEGY = 2
PACK4_MODE = 0
FP32_PROMOTE = 1
USE_SHARED_PATH = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


@triton.jit
def _floor_promote_fn(x):
    """与公共 floor_func 一致：先提升 fp32 再取整，store 时降回原 dtype。"""
    return tl.floor(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _floor_native_fn(x):
    """低精度直接取整；用于验证 GCU 后端是否接受 fp16/bf16 的 tl.floor。"""
    return tl.floor(x)


def _floor_op():
    """构造仅用于大 shape 路径的 unary 表行，不修改共享 UNARY_OPS。"""
    fn = _floor_promote_fn if FP32_PROMOTE else _floor_native_fn
    return shared_unary.UnaryOp(
        "floor", fn, same_dtype, shared_unary.NO_BOOL_NO_COMPLEX
    )


def _launch_large_block_group(op, tensors, inplace):
    """放大 BLOCK 把二维 grid 的第二维压回 255 上限内，保留公共缓存与向量化。

    metadata 布局与公共执行器完全一致，因此可直接复用 ``_device_meta``
    （带 memoisation）和 ``_launch_executor_kernel``（带 launch replay）。
    """
    dtype = tensors[0].dtype
    out_dtype = op.out_dtype_fn(dtype)
    if inplace:
        if out_dtype != dtype:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(dtype).replace('torch.', '').capitalize()}"
            )
        destinations = tensors
    else:
        destinations = [torch.empty_like(t, dtype=out_dtype) for t in tensors]

    numels = [t.numel() for t in tensors]
    max_numel = max(numels)
    tensor_count = len(tensors)
    block = LARGE_BLOCK_START
    while -(-max_numel // block) > shared_unary.GCU400_GRID_Y_LIMIT:
        block *= 2

    pointer_values = (
        [t.data_ptr() for t in tensors] + [d.data_ptr() for d in destinations] + numels
    )
    meta = _device_meta(pointer_values, tensors[0].device)
    grid = (tensor_count, -(-max_numel // block))
    pack_ok = PACK4_MODE and _pack_ok(
        (dtype, destinations[0].dtype), max_numel, pointer_values
    )
    kernel = foreach_unary_kernel_p4 if pack_ok else foreach_unary_kernel
    _launch_executor_kernel(
        kernel,
        grid,
        tensor_count,
        meta,
        op.fn,
        tl_dtype(dtype),
        tl_dtype(destinations[0].dtype),
        block,
    )
    return None if inplace else destinations


def _run_floor(self, inplace):
    """小 shape 与不支持的形态走公共执行器；大 shape 走专属参数分支。"""
    if FALLBACK_TO_PUBLIC or USE_SHARED_PATH:
        return shared_unary.apply_unary("floor", self, inplace=inplace)
    try:
        needs_chunking = shared_unary._needs_chunking(self)
    except AttributeError:
        # 非 Tensor 元素：回退公共执行器，对齐 ATen 的 TypeError/RuntimeError。
        return shared_unary.apply_unary("floor", self, inplace=inplace)
    if not needs_chunking:
        return shared_unary.apply_unary("floor", self, inplace=inplace)
    if LARGE_TENSOR_STRATEGY == 2:
        return _launch_large_block_group(_floor_op(), self, inplace)
    return shared_unary._apply_unary_chunked(_floor_op(), self, inplace)


def _foreach_floor(self):
    logger.debug("GEMS_GCU400 _FOREACH_FLOOR")
    return _run_floor(self, False)


def _foreach_floor_(self):
    logger.debug("GEMS_GCU400 _FOREACH_FLOOR_")
    return _run_floor(self, True)
