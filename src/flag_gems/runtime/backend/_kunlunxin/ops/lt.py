import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS LT")
    import os

    import torch

    container = [
        torch.Size([64, 64]),
        torch.Size([4096, 4096]),
        torch.Size([64, 512, 512]),
    ]
    if A.shape in container:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    res = lt_func(A, B)
    if "TRITONXPU_COMPARE_FUSION" in os.environ:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
    return res


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS LT SCALAR")
    return lt_func_scalar(A, B)
