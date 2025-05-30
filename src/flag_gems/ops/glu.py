import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

exp = tl_extra_shim.exp


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b

    return result


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logging.debug("GLU FORWARD")
    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)

    return out
