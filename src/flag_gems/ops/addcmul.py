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


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    logger.debug("GEMS ADDCMUL_OUT")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    if list(out.shape) != list(broadcast_shape):
        out.resize_(broadcast_shape)
    addcmul_forward(inp, tensor1, tensor2, value, out0=out)
    return out


def addcmul(inp, tensor1, tensor2, *, value=1.0):
    """Functional entry; keep alongside ``addcmul.out`` for dispatch coverage."""
    logger.debug("GEMS ADDCMUL")
    broadcast_shape = torch.broadcast_shapes(inp.shape, tensor1.shape, tensor2.shape)
    dtype = torch.promote_types(
        inp.dtype, torch.promote_types(tensor1.dtype, tensor2.dtype)
    )
    out = torch.empty(broadcast_shape, device=inp.device, dtype=dtype)
    return addcmul_out(inp, tensor1, tensor2, value=value, out=out)
