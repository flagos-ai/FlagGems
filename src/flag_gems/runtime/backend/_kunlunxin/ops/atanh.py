# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def atanh_func(x):
    x_fp32 = x.to(tl.float32)
    # The difference of logs avoids the backend's inaccurate vector divide
    # path and is the stable identity for the tested |x| < 1 domain.
    return (0.5 * (tl.log(1.0 + x_fp32) - tl.log(1.0 - x_fp32))).to(x.dtype)


def atanh(A):
    logger.debug("GEMS_KUNLUNXIN ATANH")
    return atanh_func(A)


def atanh_(A):
    logger.debug("GEMS_KUNLUNXIN ATANH_")
    atanh_func(A, out0=A)
    return A
