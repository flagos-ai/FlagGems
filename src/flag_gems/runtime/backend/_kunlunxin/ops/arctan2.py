# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _arctan2_kernel(input, other):
    input_f32 = input.to(tl.float32)
    other_f32 = other.to(tl.float32)
    result = tl_extra_shim.atan2(input_f32, other_f32)

    # XPU atan2 returns zero for atan2(+/-0, negative), losing the quadrant.
    input_bits = input_f32.to(tl.int32, bitcast=True)
    other_bits = other_f32.to(tl.int32, bitcast=True)
    signed_pi = tl.where(
        input_bits < 0, -3.141592653589793, 3.141592653589793
    )
    negative_other = (other_f32 < 0.0) | (
        (other_f32 == 0.0) & (other_bits < 0)
    )
    result = tl.where((input_f32 == 0.0) & negative_other, signed_pi, result)
    is_nan = (input_f32 != input_f32) | (other_f32 != other_f32)
    return tl.where(is_nan, float("nan"), result)


def arctan2(input, other):
    logger.debug("GEMS_KUNLUNXIN ARCTAN2")
    return _arctan2_kernel(input, other)


def arctan2_(input, other):
    logger.debug("GEMS_KUNLUNXIN ARCTAN2_")
    _arctan2_kernel(input, other, out0=input)
    return input
