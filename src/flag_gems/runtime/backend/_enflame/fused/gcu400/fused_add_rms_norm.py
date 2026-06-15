import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 16384


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    row_input_ptr = input_ptr + pid * in_stride_r
    row_residual_ptr = residual_ptr + pid * r_stride_r

    # Pass 1: add residual and store back, accumulate sum of squares
    var_acc = tl.zeros([BLOCK_SIZE], dtype=cdtype)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(row_input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
        r = tl.load(row_residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)
        x += r
        tl.store(row_residual_ptr + cols * r_stride_c, x, mask=mask)
        var_acc += x * x

    var = tl.sum(var_acc, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Pass 2: apply RMS normalization with weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(row_residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = (x * rrms * w).to(cdtype)
        tl.store(row_input_ptr + cols * in_stride_c, y, mask=mask)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[M,](
            x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
    return x, residual
