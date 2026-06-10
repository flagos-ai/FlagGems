import builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r

    mean = tl.sum(x, axis=0) / N

    # Compute variance
    _var = tl.where(mask, x - mean, 0.0)
    _var = _var * _var
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = w * x_hat + b
    y = y.to(Y.dtype.element_ty)
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_tile(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    # mask = tl.arange(0, BLOCK_SIZE) < N
    # cols = tl.arange(0, BLOCK_SIZE)
    # x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    # r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    # x += r

    # mean = tl.sum(x, axis=0) / N
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _sum += x

    mean = tl.sum(_sum) / N

    # Compute variance
    # _var = tl.where(mask, x - mean, 0.0)
    # _var = _var * _var
    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _var = tl.where(mask, x - mean, 0.0)
        _var = _var * _var
        _var_base += _var

    var = tl.sum(_var_base, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + cols * y_stride_c, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = builtins.min(
            64 * 64, triton.next_power_of_2(N)
        )  # core_num * buffer_size_limit
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        with torch.cuda.device(x.device):
            if N > 64 * 64:
                skip_layer_norm_kernel_tile[M,](
                    y,
                    x,
                    residual,
                    weight,
                    bias,
                    N,
                    1,
                    N,
                    1,
                    N,
                    1,
                    N,
                    eps,
                    BLOCK_SIZE,
                    isCloseUnrollControl=True,
                )
            else:
                skip_layer_norm_kernel[M,](
                    y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
                )
        return y

        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            skip_layer_norm_kernel[M,](
                y,
                x,
                residual,
                weight,
                bias,
                N,
                1,
                N,
                1,
                N,
                1,
                N,
                eps,
                BLOCK_SIZE,
                isCloseUnrollControl=True,
            )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
