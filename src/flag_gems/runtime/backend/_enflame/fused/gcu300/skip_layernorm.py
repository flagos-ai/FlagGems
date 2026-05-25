import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ...gcu300.utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


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
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += (pid * BLOCK_SIZE_M - 1) * y_stride_r
    X += (pid * BLOCK_SIZE_M - 1) * x_stride_r
    R += (pid * BLOCK_SIZE_M - 1) * r_stride_r
    for i in range(BLOCK_SIZE_M):
        if pid * BLOCK_SIZE_M + i < M:
            Y += y_stride_r
            X += x_stride_r
            R += r_stride_r

            mask = tl.arange(0, BLOCK_SIZE_N) < N
            cols = tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

            x += r

            mean = tl.sum(x, axis=0) / N

            # Compute variance
            _var = tl.where(mask, x - mean, 0.0)
            _var = _var * _var
            var = tl.sum(_var, axis=0) / N
            rstd = 1 / tl.sqrt(var + eps)

            w = tl.load(W + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0).to(
                tl.float32
            )
            b = tl.load(B + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0).to(
                tl.float32
            )

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

        BLOCK_SIZE_N = triton.next_power_of_2(N)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        grid = (M,)
        BLOCK_SIZE_M = 1
        if M > 65535:
            grid = (MAX_GRID_DIM,)
            BLOCK_SIZE_M = (M + MAX_GRID_DIM - 1) // MAX_GRID_DIM
        with torch_device_fn.device(x.device):
            skip_layer_norm_kernel[grid](
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
                M,
                N,
                eps,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
