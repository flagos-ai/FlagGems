import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def upsample_linear1d_kernel(
    input_ptr,
    output_ptr,
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

    # Use modulo wrap to keep all indices in [0, W_out).
    # On KunlunXin, masked tl.store does not suppress writes for masked-out
    # threads without TRITONXPU_STORE_MASK_SIM=1, causing corruption of
    # adjacent channel data.  The modulo wrap means tail-block threads simply
    # re-write already-computed values to valid positions — harmless.
    offs_w = (pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) % W_out

    offs_w_f = offs_w.to(tl.float32)

    src = offs_w_f * scale + bias

    # Clamp source position to [0, W_in - 1]
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    # For non-negative src, int truncation equals floor
    lower = src.to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)

    t = src - lower.to(tl.float32)
    w0 = 1.0 - t
    w1 = t

    # No mask needed: all offsets are within [0, W_in - 1] and [0, W_out - 1]
    x0 = tl.load(input_ptr + base_in + lower)
    x1 = tl.load(input_ptr + base_in + upper)

    x0_f = x0.to(tl.float32)
    x1_f = x1.to(tl.float32)

    out = w0 * x0_f + w1 * x1_f

    out = out.to(x0.dtype)
    tl.store(output_ptr + base_out + offs_w, out)


def upsample_linear1d(
    self: torch.Tensor,
    output_size,
    align_corners: bool,
    scales: float = None,
):
    logger.debug("GEMS UPSAMPLE LINEAR1D KUNLUNXIN")
    assert self.ndim == 3, "Input must be [N, C, W]"

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

    with torch_device_fn.device(self.device):
        upsample_linear1d_kernel[grid](
            inp,
            out,
            NC,
            W_in,
            W_out,
            scale_val,
            bias_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out.view(N, C, W_out)
