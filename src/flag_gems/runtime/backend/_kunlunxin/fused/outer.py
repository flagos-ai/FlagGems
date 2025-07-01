import logging

import torch

from ..ops import mul, mv_cluster

logger = logging.getLogger(__name__)


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("GEMS OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        inp1 = inp[:, None]
        weight1 = weight[None, :]
        inp1 = inp1.contiguous()
        weight1 = weight1.contiguous()
        out = mul(inp1, weight1)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS OUTER VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_grad = mv_cluster(out_grad, weight)
        weight_grad = mv_cluster(out_grad.t().contiguous(), inp)

        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
