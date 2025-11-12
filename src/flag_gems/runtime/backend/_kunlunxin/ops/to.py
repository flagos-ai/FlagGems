import logging
import os

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def to_dtype_func(x):
    return x


def to_dtype(x, dtype, non_blocking=False, copy=False, memory_format=None):
    logger.debug("GEMS TO.DTYPE")
    if not copy and x.dtype == dtype:
        return x
    out = torch.empty_like(x, dtype=dtype, memory_format=memory_format)
    if out.element_size() == 8:
        os.environ["TRITONXPU_ELEMBYTES"] = "8"
        os.environ["TRITONXPU_BF16_FAST"] = "1"
        res = to_dtype_func(x, out0=out)
        del os.environ["TRITONXPU_ELEMBYTES"]
        del os.environ["TRITONXPU_BF16_FAST"]
    else:
        os.environ["TRITONXPU_BF16_FAST"] = "1"
        res = to_dtype_func(x, out0=out)
        del os.environ["TRITONXPU_BF16_FAST"]
    return res
