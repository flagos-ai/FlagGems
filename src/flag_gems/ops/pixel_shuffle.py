import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    N,
    C_out,
    H,
    W,
    R,
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
    offs = tl.cast(offs32, tl.int64)

    C64 = tl.cast(C_out, tl.int64)
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

    wo = offs % W_out64
    tmp = offs // W_out64
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


def _check_and_get_shapes(x: torch.Tensor, upscale_factor: int):
    if x.dim() != 4:
        raise RuntimeError(
            f"pixel_shuffle expects a 4D tensor (N, C, H, W), but got {x.dim()}D"
        )
    if upscale_factor <= 0:
        raise RuntimeError("pixel_shuffle: upscale_factor must be > 0")
    N, C_in, H, W = x.shape
    r2 = upscale_factor * upscale_factor
    if C_in % r2 != 0:
        raise RuntimeError(
            f"Input channel dimension {C_in} is not divisible by upscale_factor^2={r2}"
        )
    C_out = C_in // r2
    H_out = H * upscale_factor
    W_out = W * upscale_factor
    return N, C_in, H, W, C_out, H_out, W_out


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
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
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


def pixel_shuffle(self: torch.Tensor, upscale_factor: int):
    logger.debug("GEMS PIXEL_SHUFFLE")
    if not isinstance(upscale_factor, int):
        raise TypeError("pixel_shuffle: upscale_factor must be an integer")

    if not self.is_cuda:
        return torch.ops.aten.pixel_shuffle.default(self, upscale_factor)

    N, C_in, H, W, C_out, H_out, W_out = _check_and_get_shapes(
        self, upscale_factor
    )
    out = torch.empty(
        (N, C_out, H_out, W_out),
        dtype=self.dtype,
        device=self.device,
        layout=self.layout,
    )
    _launch_pixel_shuffle_kernel(self, out, upscale_factor)
    return out


def pixel_shuffle_out(self: torch.Tensor, upscale_factor: int, out: torch.Tensor):
    logger.debug("GEMS PIXEL_SHUFFLE_OUT")
    if not isinstance(upscale_factor, int):
        raise TypeError("pixel_shuffle_out: upscale_factor must be an integer")

    if not self.is_cuda or not out.is_cuda:
        return torch.ops.aten.pixel_shuffle.out(self, upscale_factor, out=out)

    N, C_in, H, W, C_out, H_out, W_out = _check_and_get_shapes(
        self, upscale_factor
    )
    expected_shape = (N, C_out, H_out, W_out)
    if tuple(out.shape) != expected_shape:
        raise RuntimeError(
            "pixel_shuffle_out: out tensor has incorrect shape, "
            f"expected {expected_shape} but got {tuple(out.shape)}"
        )
    if out.dtype != self.dtype:
        raise RuntimeError("pixel_shuffle_out: out dtype must match input dtype")
    _launch_pixel_shuffle_kernel(self, out, upscale_factor)
    return out
