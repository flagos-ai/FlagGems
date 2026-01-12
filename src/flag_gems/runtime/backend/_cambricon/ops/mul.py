import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y, inplace):
    return x * y


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def mul_func_scalar(x, y, inplace):
    return x * y


def mul(A, B):
    logger.debug("GEMS_CAMBRICON MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.device != B.device:
            if A.dim() == 0:
                assert A.device == torch.device("cpu"), "expect scalar tensor on cpu"
                A = A.to(B.device)
            elif B.dim() == 0:
                assert B.device == torch.device("cpu"), "expect scalar tensor on cpu"
                B = B.to(A.device)
        return mul_func(A, B, False)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B, False)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A, False)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_CAMBRICON MUL_")
    if isinstance(B, torch.Tensor):
        if B.device != A.device and B.dim() == 0:
            assert B.device == torch.device("cpu"), "expect scalar tensor on cpu"
            B = B.to(A.device)
        return mul_func(A, B, True, out0=A)
    else:
        return mul_func_scalar(A, B, True, out0=A)
