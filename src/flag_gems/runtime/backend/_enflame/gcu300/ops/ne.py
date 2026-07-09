import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


def _is_float64_scalar(*args):
    return any(
        isinstance(a, torch.Tensor) and a.dtype == torch.float64 and a.ndim == 0
        for a in args
    )


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_ENFLAME NE")
    if _is_float64_scalar(A, B):
        dev = A.device if isinstance(A, torch.Tensor) else B.device
        A_cpu = A.cpu() if isinstance(A, torch.Tensor) else A
        B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
        return torch.ne(A_cpu, B_cpu).to(dev)
    return ne_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne_scalar(A, B):
    logger.debug("GEMS_ENFLAME NE_SCALAR")
    if _is_float64_scalar(A):
        return torch.ne(A.cpu(), B).to(A.device)
    return ne_func_scalar(A, B)
