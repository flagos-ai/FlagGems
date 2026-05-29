import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

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
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = ext.program_id(0)
    X += (pid * BLOCK_SIZE_M - 1) * x_stride_r
    R += (pid * BLOCK_SIZE_M - 1) * r_stride_r
    for i in range(BLOCK_SIZE_M):
        if pid * BLOCK_SIZE_M + i < M:
            X += x_stride_r
            R += r_stride_r

            mask = tl.arange(0, BLOCK_SIZE_N) < N
            cols = tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

            x += r
            # write back to residual
            tl.store(R + cols * r_stride_c, x, mask=mask)

            var = tl.sum(x * x / N, axis=0)
            rrms = 1 / tl.sqrt(var + eps)

            w = tl.load(W + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0)
            y = (x * rrms).to(X.dtype.element_ty) * w
            # write back to input
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

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    grid = (M,)
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_M = 1
    if M > 65535:
        grid = (128,)
        BLOCK_SIZE_M = (M + 128 - 1) // 128

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[grid](
            x, residual, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    return x, residual
