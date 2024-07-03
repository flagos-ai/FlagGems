import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logging.debug("GEMS LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        rms_norm_kernel[M,](y, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)
        return y


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
