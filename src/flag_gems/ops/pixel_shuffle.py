import logging
from math import prod

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Pixel Shuffle: (*, C*r^2, H, W) -> (*, C, H*r, W*r)
# Direct index mapping kernel - each output element reads from the correct
# input position without intermediate tensors.
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],
)
@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    C,
    H,
    W,
    R,
    C_out,
    H_out,
    W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Output layout: (N, C_out, H_out, W_out)
    ow = offsets % W_out
    tmp = offsets // W_out
    oh = tmp % H_out
    tmp2 = tmp // H_out
    c_out = tmp2 % C_out
    n = tmp2 // C_out

    # Map to input: h_in = oh // R, w_in = ow // R
    h_in = oh // R
    dh = oh % R
    w_in = ow // R
    dw = ow % R

    # Input channel: c_in = c_out * R * R + dh * R + dw
    c_in = c_out * R * R + dh * R + dw

    # Input linear index
    in_idx = n * (C * H * W) + c_in * (H * W) + h_in * W + w_in

    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


def _check_pixel_shuffle(input, upscale_factor):
    r = int(upscale_factor)
    if r <= 0:
        raise RuntimeError(
            f"pixel_shuffle expects a positive upscale_factor, but got {r}"
        )
    if input.ndim < 3:
        raise RuntimeError(
            "pixel_shuffle expects input to have at least 3 dimensions, "
            f"but got input with {input.ndim} dimension(s)"
        )

    C, H, W = input.shape[-3:]
    r2 = r * r
    if C % r2 != 0:
        raise RuntimeError(
            "pixel_shuffle expects its input's 'channel' dimension to be divisible "
            f"by the square of upscale_factor, but input.size(-3)={C} is not "
            f"divisible by {r2}"
        )

    output_shape = (*input.shape[:-3], C // r2, H * r, W * r)
    return r, C, H, W, output_shape


def _flatten_input(input, C, H, W):
    batch = prod(input.shape[:-3])
    return input.contiguous().reshape(batch, C, H, W)


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor):
    try:
        return torch._C._overlaps(left, right)
    except AttributeError:
        return True


def _launch_pixel_shuffle(input, output, r, C, H, W):
    n_elements = output.numel()
    if n_elements == 0:
        return output

    C_out = C // (r * r)
    H_out = H * r
    W_out = W * r
    input = input.contiguous()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        pixel_shuffle_kernel[grid](
            input,
            output,
            n_elements,
            C,
            H,
            W,
            r,
            C_out,
            H_out,
            W_out,
        )
    return output


def pixel_shuffle(input, upscale_factor):
    logger.debug("GEMS PIXEL_SHUFFLE")
    r, C, H, W, output_shape = _check_pixel_shuffle(input, upscale_factor)
    input = _flatten_input(input, C, H, W)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    return _launch_pixel_shuffle(input, output, r, C, H, W)


def pixel_shuffle_out(input, upscale_factor, out):
    logger.debug("GEMS PIXEL_SHUFFLE_OUT")
    r, C, H, W, output_shape = _check_pixel_shuffle(input, upscale_factor)
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError("Expected out tensor to be on the same device as input")

    input_flat = _flatten_input(input, C, H, W)
    if _tensors_overlap(input, out):
        input_flat = input_flat.clone()

    if tuple(out.shape) != tuple(output_shape):
        out.resize_(output_shape)

    if out.is_contiguous():
        return _launch_pixel_shuffle(input_flat, out, r, C, H, W)

    out_contiguous = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    _launch_pixel_shuffle(input_flat, out_contiguous, r, C, H, W)
    out.copy_(out_contiguous)
    return out
