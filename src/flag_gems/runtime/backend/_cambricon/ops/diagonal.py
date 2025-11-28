import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS_CAMBRICON diagonal backward")
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    copy_func.instantiate(grad_output.ndim)(grad_output, out0=diag)
    return grad_input
