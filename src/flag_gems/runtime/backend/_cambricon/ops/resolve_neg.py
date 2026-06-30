import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_kernel(x, inplace):
    return -x


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_CAMBRICON RESOLVE_NEG")
    return neg_kernel(A, False) if A.is_neg() else A
