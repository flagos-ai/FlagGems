import torch
import triton.language as tl
import logging
from ..utils import libentry
from .mul import mul
from .mm import mm


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logging.debug("GEMS OUTER")
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
        logging.debug("GEMS OUTER VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_shape = inp.shape
        inp_grad_mid = mm(out_grad, weight[:, None])
        inp_grad = inp_grad_mid.reshape(inp_shape)

        weight_shape = weight.shape
        inp = inp[None, :]
        inp = inp.contiguous()
        weight_grad_mid = mm(inp, out_grad)
        weight_grad = weight_grad_mid.reshape(weight_shape)

        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
