import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def upsample_linear1d_kernel(
    input_ptr,
    output_ptr,
    NC,
    W_in,
    W_out,
    align_corners,
    scale_ac,
    scale_nc,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_w = tl.program_id(1)

    base_in = pid_nc * W_in
    base_out = pid_nc * W_out

    offs_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (pid_nc < NC) & (offs_w < W_out)

    offs_w_f = offs_w.to(tl.float32)

    src = tl.where(
        align_corners != 0,
        offs_w_f * scale_ac,
        (offs_w_f + 0.5) * scale_nc - 0.5,
    )

    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    lower = tl.floor(src).to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)
    w0 = 1.0 - t
    w1 = t

    x0 = tl.load(input_ptr + base_in + lower, mask=mask)
    x1 = tl.load(input_ptr + base_in + upper, mask=mask)

    x0_f = x0.to(tl.float32)
    x1_f = x1.to(tl.float32)

    out = w0 * x0_f + w1 * x1_f

    out = out.to(x0.dtype)

    tl.store(output_ptr + base_out + offs_w, out, mask=mask)


def upsample_linear1d(
    self: torch.Tensor,
    output_size,
    align_corners: bool,
    scales: float = None,
):
    logger.debug("GEMS UPSAMPLE LINEAR1D")
    assert self.ndim == 3, "Input must be [N, C, W]"
    assert self.is_cuda

    N, C, W_in = self.shape
    NC = N * C

    if output_size is not None:
        W_out = int(
            output_size[0] if isinstance(output_size, (list, tuple)) else output_size
        )
    else:
        assert scales is not None
        W_out = int(math.floor(W_in * scales))

    inp = self.contiguous().view(NC, W_in)
    out = torch.empty((NC, W_out), device=self.device, dtype=self.dtype)

    if align_corners:
        scale_ac = (W_in - 1) / (W_out - 1) if W_out > 1 else 0.0
        scale_nc = 0.0
    else:
        scale_nc = 1.0 / scales if scales is not None else W_in / W_out
        scale_ac = 0.0

    BLOCK_SIZE = 256
    grid = (NC, triton.cdiv(W_out, BLOCK_SIZE))

    upsample_linear1d_kernel[grid](
        inp,
        out,
        NC,
        W_in,
        W_out,
        int(align_corners),
        float(scale_ac),
        float(scale_nc),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(N, C, W_out)
