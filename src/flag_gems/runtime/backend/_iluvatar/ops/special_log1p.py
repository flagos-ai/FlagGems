import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_log1p_func(x):
    return tl.log(x + 1.0)


def special_log1p(A):
    logger.debug("ILUVATAR GEMS SPECIAL_LOG1P")
    if isinstance(A, torch.Tensor):
        return special_log1p_func(A)
    else:
        # Scalar
        return torch.log(torch.tensor(A + 1.0))
