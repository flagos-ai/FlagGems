import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def frac_func(x):
    """
    Match torch.frac semantics:
      out = x - trunc(x)   (trunc rounds toward 0)

    Implement trunc with floor:
      trunc(x) = floor(x)            if x >= 0
      trunc(x) = -floor(-x) (ceil)   if x < 0
    """
    xf = x.to(tl.float32)
    trunc_f = tl.where(xf < 0.0, -tl.floor(-xf), tl.floor(xf))
    return xf - trunc_f


# frac(Tensor self) -> Tensor
def frac(A):
    logger.debug("GEMS FRAC")
    return frac_func(A)


# frac_(Tensor(a!) self) -> Tensor(a!)
def frac_(A):
    logger.debug("GEMS FRAC_")
    frac_func(A, out0=A)
    return A


# frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def frac_out(A, out):
    logger.debug("GEMS FRAC_OUT")
    return frac_func(A, out0=out)
