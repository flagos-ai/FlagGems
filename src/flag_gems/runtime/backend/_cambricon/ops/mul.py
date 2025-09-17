import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


def mul(A, B):
    logger.debug("GEMS_CAMBRICON MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.shape != B.shape:
            A, B = torch.broadcast_tensors(A, B)
            A = A.clone()
            B = B.clone()
        return mul_func(A, B)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_CAMBRICON MUL_")
    if isinstance(B, torch.Tensor):
        if A.shape != B.shape:
            A, B = torch.broadcast_tensors(A, B)
            A = A.clone()
            B = B.clone()
        return mul_func(A, B, out0=A)
    else:
        return mul_func_scalar(A, B, out0=A)
