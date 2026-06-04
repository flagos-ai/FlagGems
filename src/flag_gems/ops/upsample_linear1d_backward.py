import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    NC,
    W_in,
    W_out,
    scale,
    bias,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_w = tl.program_id(1)

    base_in = pid_nc * W_in
    base_out = pid_nc * W_out

    offs_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (pid_nc < NC) & (offs_w < W_out)

    offs_w_f = offs_w.to(tl.float32)

    src = offs_w_f * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    lower = tl.floor(src).to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)
    w0 = 1.0 - t
    w1 = t

    grad_out = tl.load(grad_output_ptr + base_out + offs_w, mask=mask).to(tl.float32)

    g0 = grad_out * w0
    g1 = grad_out * g1

    tl.atomic_add(grad_input_ptr + base_in + lower, g0, mask=mask)
    tl.atomic_add(grad_input_ptr + base_in + upper, g1, mask=mask)


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales: float = None,
):
    logger.debug("GEMS UPSAMPLE LINEAR1D BACKWARD")
    assert grad_output.ndim == 3, "grad_output must be [N, C, W]"
    assert grad_output.device.type == flag_gems.device

    N, C, W_out = grad_output.shape
    W_in = input_size[-1]
    NC = N * C

    go = grad_output.contiguous().view(NC, W_out)
    gi = torch.zeros((NC, W_in), device=grad_output.device, dtype=torch.float32)

    if align_corners:
        if W_out > 1:
            scale_val = (W_in - 1.0) / (W_out - 1.0)
        else:
            scale_val = 0.0
        bias_val = 0.0
    else:
        if scales is not None:
            real_scale = 1.0 / scales
        else:
            real_scale = W_in / W_out
        scale_val = real_scale
        bias_val = 0.5 * real_scale - 0.5

    BLOCK_SIZE = 256
    grid = (NC, triton.cdiv(W_out, BLOCK_SIZE))

    upsample_linear1d_backward_kernel[grid](
        go,
        gi,
        NC,
        W_in,
        W_out,
        scale_val,
        bias_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if gi.dtype == grad_output.dtype:
        return gi.view(N, C, W_in)
    result = torch.empty((N, C, W_in), device=grad_output.device, dtype=grad_output.dtype)
    result.copy_(gi.view(N, C, W_in))
    return result
