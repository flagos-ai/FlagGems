import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def isnan_func(x):
    return tl.math.isnan(x.to(tl.float32))


def isnan(A):
    logging.debug("GEMS ISNAN")
    O = isnan_func(A)
    return O
