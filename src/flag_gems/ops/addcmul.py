import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcmul_forward(x, t1, t2, value):
    return x + value * t1 * t2


def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS ADDCMUL FORWARD")
    if out is not None:
        broadcast_shape = torch.broadcast_shapes(
            inp.shape, tensor1.shape, tensor2.shape
        )
        if list(out.shape) != list(broadcast_shape):
            out.resize_(broadcast_shape)
        addcmul_forward(inp, tensor1, tensor2, value, out0=out)
        return out
    else:
        return addcmul_forward(inp, tensor1, tensor2, value)


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS ADDCMUL OUT FORWARD")
    if out is not None:
        broadcast_shape = torch.broadcast_shapes(
            inp.shape, tensor1.shape, tensor2.shape
        )
        if list(out.shape) != list(broadcast_shape):
            out.resize_(broadcast_shape)
        addcmul_forward(inp, tensor1, tensor2, value, out0=out)
        return out
    return addcmul_forward(inp, tensor1, tensor2, value)
