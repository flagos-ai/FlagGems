import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcdiv_kernel(x, t1, t2, value):
    return x + value * (t1 / t2)


def addcdiv(inp, tensor1, tensor2, value=1.0, out=None):
    logger.debug("GEMS ADDCDIV FORWARD")

    if out is None:
        out = torch.empty_like(inp)

    addcdiv_kernel(inp, tensor1, tensor2, value, out0=out)

    return out
