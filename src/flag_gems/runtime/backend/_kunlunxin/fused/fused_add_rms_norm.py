import builtins
import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r
    # write back to residual
    tl.store(R + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(X.dtype.element_ty) * w
    # write back to input
    tl.store(X + cols * x_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel_tile(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weight
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * x_stride_r
    R += pid * r_stride_r

    # var = tl.sum(x * x / N, axis=0)
    # rrms = 1 / tl.sqrt(var + eps)

    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _var_base += x * x / N
    var = tl.sum(_var_base)
    rrms = 1 / tl.sqrt(var + eps)

    # w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    # y = (x * rrms).to(Y.dtype.element_ty) * w
    # tl.store(Y + cols * y_stride_c, y, mask=mask)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        w = tl.load(W + cols, mask, other=0.0)
        y = (x * rrms).to(X.dtype.element_ty) * w
        # write back to residual and input
        tl.store(R + cols * r_stride_c, x, mask=mask)
        tl.store(X + cols * x_stride_c, y, mask=mask)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug("GEMS FUSED_ADD_RMS_NORM FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = builtins.min(
        64 * 128, triton.next_power_of_2(N)
    )  # core_num * buffer_size_limit
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        if N > 64 * 128:
            fused_add_rms_norm_kernel_tile[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        else:
            fused_add_rms_norm_kernel[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
    return x, residual
