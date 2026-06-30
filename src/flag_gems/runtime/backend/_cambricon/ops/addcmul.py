import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcmul_forward(x, t1, t2, value):
    return x + value * t1 * t2


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    logger.debug("GEMS_CAMBRICON ADDCMUL_OUT")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    if list(out.shape) != list(broadcast_shape):
        out.resize_(broadcast_shape)
    addcmul_forward(inp, tensor1, tensor2, value, out0=out)
    return out


def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS_CAMBRICON ADDCMUL FORWARD")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    dtype = torch.promote_types(
        inp.dtype, torch.promote_types(tensor1.dtype, tensor2.dtype)
    )
    out = torch.empty(broadcast_shape, device=inp.device, dtype=dtype)
    return addcmul_out(inp, tensor1, tensor2, value=value, out=out)
