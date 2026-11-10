# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 _foreach_sub / _foreach_sub_ 的 .Scalar 与 .ScalarList。

公共 foreach 执行器用二维 grid 承载 tensor 列表，GCU400 的 ``grid.y``
上限 255 在大 shape 下会 ``out of resource``。浮点连续列表改走
``_foreach_flat`` 的一维固定 grid / metadata 融合 kernel；非浮点、
非连续或广播等形态仍委托公共执行器保持 dtype policy 与异常语义。
"""

import logging

from flag_gems.ops._foreach_binary import _apply_scalar

from ._foreach_flat import (
    FALLBACK,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    OP_SUB,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；均为编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


def _foreach_sub_Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_SUB_SCALAR")

    def fallback():
        return _apply_scalar("sub", self, scalar, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_SUB,
        MODE_SCALAR,
        False,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_sub_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_SUB_SCALARLIST")

    def fallback():
        return _apply_scalar("sub", self, scalars, inplace=False)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_SUB,
        MODE_SCALAR_LIST,
        False,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_sub__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_SUB__SCALAR")

    def fallback():
        return _apply_scalar("sub", self, scalar, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalar,
        OP_SUB,
        MODE_SCALAR,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_sub__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_SUB__SCALARLIST")

    def fallback():
        return _apply_scalar("sub", self, scalars, inplace=True)

    if FALLBACK_TO_PUBLIC:
        return fallback()
    result = run_binary(
        self,
        scalars,
        OP_SUB,
        MODE_SCALAR_LIST,
        True,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result
