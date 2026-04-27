import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func_no_alpha(x, y):
    return x - y


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
    if alpha == 1 and isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return sub_func_no_alpha(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS SUB_")
    if isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return sub_func(A, B, alpha, out0=A)
    else:
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
