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
def rms_norm_kernel(
    Y,  # output tensor (GMEM)
    INV_RMS,  # inverse rms output (GMEM)
    X,  # input tensor (GMEM)
    W,  # weight tensor (GMEM)
    y_stride_r,  # output row stride
    y_stride_c,  # output col stride
    x_stride_r,  # input row stride
    x_stride_c,  # input col stride
    N,  # number of elements to normalize
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,  # chunk size matching TPU vector unit width
):
    pid = tle.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    # Phase 1: accumulate sum of squares across chunks
    # Each chunk: DMA load from GMEM -> LMEM (sg.dma.ld)
    #            vector fmul (x * x) -> cross-lane reduction
    sum_x2 = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        sum_x2 += tl.sum(x * x)

    # Compute inverse rms: 1/sqrt(mean(x^2) + eps)
    # Maps to: div -> add -> sg.sfu.rsqrt
    var = sum_x2 / N
    rrms = tl.rsqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    # Phase 2: normalize with rsqrt and apply weight
    # Maps to: DMA load -> vector fmul (x * rrms) -> fmul (result * w) -> DMA store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols * y_stride_c, y, mask=mask)



class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), 256)
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

        with torch_device_fn.device(x.device):
            rms_norm_kernel[M,](y, inv_rms, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
