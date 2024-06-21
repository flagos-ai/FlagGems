import logging

import torch
import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def isinf_func(x):
    return tl.math.isinf(x.to(tl.float32))


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def isinf(A):
    logging.debug("GEMS ISINF")
    return isinf_func(A)
