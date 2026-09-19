import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def add_relu_func(x, y, alpha):
    # relu(x + alpha*y) = max(0, x + alpha*y); single maximum instruction.
    return tl.maximum(x + y * alpha, 0)


def _add_relu(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADD_RELU")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_relu_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_relu_func(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_relu_func(A, B, alpha)
    else:
        return torch.tensor(max(0, A + B * alpha))


def _add_relu_(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN ADD_RELU_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        add_relu_func(A, B, alpha, out0=A)
        return A
    else:
        raise ValueError("Unreachable.")
