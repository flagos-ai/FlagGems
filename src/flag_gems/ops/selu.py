import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def selu_kernel(x):
    x_f32 = x.to(tl.float32)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x_neg = tl.minimum(x_f32, 0.0)
    neg_part = alpha * (tl.exp(x_neg) - 1.0)
    out_f32 = tl.where(x_f32 > 0.0, x_f32, neg_part)
    return (scale * out_f32).to(x.dtype)


def selu(*args, **kwargs):
    logger.debug("GEMS SELU")
    x = None
    if len(args) > 0:
        x = args[0]
    elif "input" in kwargs:
        x = kwargs["input"]
    elif "self" in kwargs:
        x = kwargs["self"]
    else:
        raise TypeError("selu() missing required argument 'input' (pos 1)")

    if not isinstance(x, torch.Tensor):
        raise TypeError("selu() expected a torch.Tensor as input")

    if not x.is_floating_point():
        raise TypeError("selu() expected a floating point tensor")

    return selu_kernel(x)
