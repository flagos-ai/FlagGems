import logging
import os

import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=4096,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_not_func(x):
    return x == 0


def logical_not(A):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_NOT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        return logical_not_func(A)
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]


def logical_not_(A):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_NOT_")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        logical_not_func(A, out0=A)
    finally:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return A
