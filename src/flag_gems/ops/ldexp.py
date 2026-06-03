import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# ldexp(x, exp) returns x * 2**exp. PyTorch's aten::ldexp.Tensor takes both
# arguments as tensors (the exponent is typically an integer or float tensor
# that gets type-promoted). We compute in float32 for fp16/bf16 inputs to
# preserve precision when `exp` is large, then pointwise_dynamic casts the
# result back according to type promotion.
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_func(x, exp):
    x_f32 = x.to(tl.float32)
    e_f32 = exp.to(tl.float32)
    return x_f32 * tl.math.exp2(e_f32)


def ldexp(self, other):
    logger.debug("GEMS LDEXP")
    return ldexp_func(self, other)


def ldexp_out(self, other, out):
    logger.debug("GEMS LDEXP_OUT")
    ldexp_func(self, other, out0=out)
    return out
