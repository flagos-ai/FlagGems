import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

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
    x_stride_r,  # pointer increment for moving 1 row
    x_stride_c,  # pointer increment for moving 1 column
    r_stride_r,  # pointer increment for moving 1 row in residual
    r_stride_c,  # pointer increment for moving 1 column in residual
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    loops = tl.cdiv(N, BLOCK_SIZE)

    # Initialize accumulators for sum of x and sum of squared x (for mean/variance)
    sum_x = tl.zeros((), dtype=tl.float32)  # Explicitly specify as float32
    sum_sq = tl.zeros((), dtype=tl.float32)  # For variance calculation

    # Pointer offsets for current row (based on program ID)
    X += pid * x_stride_r
    R += pid * r_stride_r
    Y += pid * y_stride_r

    # This partitioning is special: need to load entire row data for computation, so N is required
    # First pass: compute sum(x) and sum(x²) in one traversal (optimized from 2 passes)
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load input and residual, compute x = X + residual
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        # Accumulate sum of x and sum of x squared
        sum_x += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute mean and variance from accumulated sums
    mean = sum_x / N
    var = (sum_sq / N) - (mean * mean)  # Equivalent to E[x²] - (E[x])²
    rstd = 1 / tl.sqrt(var + eps)  # Reciprocal of standard deviation

    # Second pass: compute final output (y = w * (x-mean)/std + b)
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load weights and biases
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        # Re-load x (X + residual) for normalization
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        # Apply layer norm and linear transformation
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        # Cast to output dtype and store
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + cols * y_stride_c, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS_ASCEND SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = min(math.prod(x.shape[:dim]), 65535)
        N = math.prod(normalized_shape)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            skip_layer_norm_kernel[M,](
                y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
