import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)
exp = tl_extra_shim.exp
log = tl_extra_shim.log


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def adaptive_attention_span_forward(x):
    """
    Adaptive Attention Span forward pass.
    This implements a softplus-like function commonly used in attention mechanisms
    to compute adaptive attention spans.
    Formula: softplus(x) = log(1 + exp(x))
    """
    return log(1.0 + exp(x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def adaptive_attention_span_backward(y, dy):
    """
    Adaptive Attention Span backward pass.
    Derivative of softplus: sigmoid(x) = 1 / (1 + exp(-x))
    Since y = softplus(x), we have: dy/dx = sigmoid(x) = y / (1 + y)
    """
    return dy * (y / (1.0 + y))


class Adaptive_Attention_Span(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS ADAPTIVE_ATTENTION_SPAN FORWARD")
        if A.requires_grad is True:
            out = adaptive_attention_span_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = adaptive_attention_span_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS ADAPTIVE_ATTENTION_SPAN BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = adaptive_attention_span_backward(out, out_grad)
        return in_grad


def adaptive_attention_span(A):
    return Adaptive_Attention_Span.apply(A)