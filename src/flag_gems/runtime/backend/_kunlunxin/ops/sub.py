import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


def sub(A, B, *, alpha=1):
    logger.debug("GEMS SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
