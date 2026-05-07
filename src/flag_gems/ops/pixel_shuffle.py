import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    R,
    C_out,
    H_out,
    W_out,
    s_in_n,
    s_in_c,
    s_in_h,
    s_in_w,
    s_out_n,
    s_out_c,
    s_out_h,
    s_out_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs32 = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs32 < n_elements
    offsets = tl.cast(offs32, tl.int64)

    N64 = tl.cast(N, tl.int64)  # noqa: F841
    C64 = tl.cast(C_out, tl.int64)
    H64 = tl.cast(H, tl.int64)  # noqa: F841
    W64 = tl.cast(W, tl.int64)  # noqa: F841
    R64 = tl.cast(R, tl.int64)
    H_out64 = tl.cast(H_out, tl.int64)
    W_out64 = tl.cast(W_out, tl.int64)

    s_in_n64 = tl.cast(s_in_n, tl.int64)
    s_in_c64 = tl.cast(s_in_c, tl.int64)
    s_in_h64 = tl.cast(s_in_h, tl.int64)
    s_in_w64 = tl.cast(s_in_w, tl.int64)

    s_out_n64 = tl.cast(s_out_n, tl.int64)
    s_out_c64 = tl.cast(s_out_c, tl.int64)
    s_out_h64 = tl.cast(s_out_h, tl.int64)
    s_out_w64 = tl.cast(s_out_w, tl.int64)

    wo = offsets % W_out64
    tmp = offsets // W_out64
    ho = tmp % H_out64
    tmp = tmp // H_out64
    co = tmp % C64
    no = tmp // C64

    rh = ho % R64
    h = ho // R64
    rw = wo % R64
    w = wo // R64

    cin = co * (R64 * R64) + rh * R64 + rw

    in_idx = no * s_in_n64 + cin * s_in_c64 + h * s_in_h64 + w * s_in_w64
    out_idx = no * s_out_n64 + co * s_out_c64 + ho * s_out_h64 + wo * s_out_w64

    val = tl.load(in_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + out_idx, val, mask=mask)


def _check_and_get_shapes_strides(x: torch.Tensor, upscale_factor: int):
    if x.dim() != 4:
        raise RuntimeError(
            f"pixel_shuffle expects a 4D tensor (N, C, H, W), but got {x.dim()}D"
        )
    if upscale_factor <= 0:
        raise RuntimeError("upscale_factor must be > 0")
    N, C_in, H, W = x.shape
    r2 = upscale_factor * upscale_factor
    if C_in % r2 != 0:
        raise RuntimeError(
            f"Input channel dimension {C_in} is not divisible by upscale_factor^2={r2}"
        )
    C_out = C_in // r2
    H_out = H * upscale_factor
    W_out = W * upscale_factor
    in_strides = x.stride()
    return (N, C_in, H, W, C_out, H_out, W_out, in_strides)


def _launch_pixel_shuffle_kernel(
    x: torch.Tensor, out: torch.Tensor, upscale_factor: int
):
    N, C_in, H, W = x.shape
    C_out = C_in // (upscale_factor * upscale_factor)
    H_out = H * upscale_factor
    W_out = W * upscale_factor

    s_in_n, s_in_c, s_in_h, s_in_w = x.stride()
    s_out_n, s_out_c, s_out_h, s_out_w = out.stride()

    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        pixel_shuffle_kernel[grid](
            x,
            out,
            N,
            C_out,
            H,
            W,
            upscale_factor,
            H_out,
            W_out,
            s_in_n,
            s_in_c,
            s_in_h,
            s_in_w,
            s_out_n,
            s_out_c,
            s_out_h,
            s_out_w,
            n_elements,
            BLOCK_SIZE=1024,
        )


def pixel_shuffle(input, upscale_factor):
    logger.debug("GEMS PIXEL_SHUFFLE")
    r = int(upscale_factor)
    N, C, H, W, C_out, H_out, W_out, _ = _check_and_get_shapes_strides(
        input, r
    )

    output = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    _launch_pixel_shuffle_kernel(input, output, r)
    return output


def pixel_shuffle_out(input, upscale_factor, out):
    logger.debug("GEMS PIXEL_SHUFFLE_OUT")
    r = int(upscale_factor)
    N, C, H, W, C_out, H_out, W_out, _ = _check_and_get_shapes_strides(
        input, r
    )

    expected_shape = (N, C_out, H_out, W_out)
    assert out.shape == expected_shape, f"out must have shape {expected_shape}"
    assert out.dtype == input.dtype, "out dtype must match input dtype"
    assert out.device == input.device, "out device must match input device"

    _launch_pixel_shuffle_kernel(input, out, r)
    return out
