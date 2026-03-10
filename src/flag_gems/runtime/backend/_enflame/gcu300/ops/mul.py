import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


def mul(A, B):
    logger.debug("GEMS MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return mul_func(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return mul_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return mul_func_scalar(B, A)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS MUL_")
    if isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return mul_func(A, B, out0=A)
    else:
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return mul_func_scalar(A, B, out0=A)
