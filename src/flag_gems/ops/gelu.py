import logging

import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def gelu_none(x):
    scale = 0.7071067811
    output = 0.5 * x * (1 + tl.math.erf(x * scale))
    return output


@pointwise_dynamic
@triton.jit
def gelu_tanh(x):
    output = (
        0.5
        * x
        * (
            1
            + tl.math.tanh(
                x * 0.79788456 * (1 + 0.044715 * tl.math.pow(x.to(tl.float32), 2))
            )
        )
    )
    return output


@elementwise_type_promotion_wrapper(
    type_promoting_args=("A"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def gelu(A, *, approximate="none"):
    logging.debug("GEMS GELU")
    if approximate == "tanh":
        return gelu_tanh(A)
    else:
        return gelu_none(A)
