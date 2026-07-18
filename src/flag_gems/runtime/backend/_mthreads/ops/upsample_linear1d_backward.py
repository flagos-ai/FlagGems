import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SCALE2_BLOCK_W = 256
_GENERIC_BLOCK_W = 512


@libentry()
@triton.jit
def _upsample_linear1d_backward_scale2_kernel(
    grad_out,
    grad_in,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    UPSAMPLE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    x_in = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (row < rows) & (x_in < IN_W)
    grad_base = grad_out + row * OUT_W

    if UPSAMPLE:
        x_even = x_in * 2
        grad_prev = tl.load(
            grad_base + x_even - 1, mask=mask & (x_even > 0), other=0.0
        ).to(tl.float32)
        grad_even = tl.load(grad_base + x_even, mask=mask, other=0.0).to(tl.float32)
        grad_odd = tl.load(grad_base + x_even + 1, mask=mask, other=0.0).to(tl.float32)
        grad_next = tl.load(
            grad_base + x_even + 2,
            mask=mask & (x_even + 2 < OUT_W),
            other=0.0,
        ).to(tl.float32)
        result = 0.25 * (grad_prev + grad_next) + 0.75 * (grad_even + grad_odd)
        result += tl.where(x_in == 0, 0.25 * grad_even, 0.0)
        result += tl.where(x_in == IN_W - 1, 0.25 * grad_odd, 0.0)
    else:
        result = 0.5 * tl.load(grad_base + x_in // 2, mask=mask, other=0.0).to(
            tl.float32
        )

    tl.store(grad_in + row * IN_W + x_in, result, mask=mask)


@libentry()
@triton.jit
def _upsample_linear1d_backward_kernel(
    grad_out,
    grad_in,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    x_in = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (row < rows) & (x_in < IN_W)
    x_in_f = x_in.to(tl.float32)

    if ALIGN_CORNERS:
        if IN_W > 1:
            center = x_in_f * ((OUT_W - 1) + 0.0) / ((IN_W - 1) + 0.0)
        else:
            center = tl.zeros((BLOCK_W,), dtype=tl.float32)
    else:
        center = (x_in_f + 0.5) / SCALE - 0.5

    base = tl.floor(center).to(tl.int32)
    grad_base = grad_out + row * OUT_W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for delta in tl.static_range(-WINDOW, WINDOW + 1):
        x_out = base + delta
        valid = mask & (x_out >= 0) & (x_out < OUT_W)
        x_out_f = x_out.to(tl.float32)
        if ALIGN_CORNERS:
            if OUT_W > 1:
                x_real = x_out_f * ((IN_W - 1) + 0.0) / ((OUT_W - 1) + 0.0)
            else:
                x_real = tl.zeros((BLOCK_W,), dtype=tl.float32)
        else:
            x_real = (x_out_f + 0.5) * SCALE - 0.5

        x0_f = tl.floor(x_real)
        weight1 = x_real - x0_f
        weight0 = 1.0 - weight1
        x0 = tl.maximum(x0_f, 0.0).to(tl.int32)
        x1 = tl.minimum(x0_f + 1.0, (IN_W - 1) + 0.0).to(tl.int32)
        grad = tl.load(grad_base + x_out, mask=valid, other=0.0).to(tl.float32)

        same = x0 == x1
        is_x0 = x_in.to(tl.int32) == x0
        is_x1 = x_in.to(tl.int32) == x1
        acc += tl.where(same & is_x0, grad * (weight0 + weight1), 0.0)
        acc += tl.where((~same) & is_x0, grad * weight0, 0.0)
        acc += tl.where((~same) & is_x1, grad * weight1, 0.0)

    tl.store(grad_in + row * IN_W + x_in, acc, mask=mask)


def _scale_value(in_w, out_w, align_corners, scale_factors):
    if align_corners:
        return (in_w - 1) / (out_w - 1) if out_w > 1 else 0.0
    if scale_factors is not None:
        scale_factor = (
            scale_factors[0]
            if isinstance(scale_factors, (list, tuple))
            else scale_factors
        )
        if scale_factor > 0:
            return 1.0 / scale_factor
    return in_w / out_w


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS UPSAMPLE_LINEAR1D_BACKWARD")

    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError(
            f"expected input_size with 1 to 3 dimensions, got {input_size}"
        )

    if output_size is not None:
        out_w = output_size[0]
    else:
        if scale_factors is None:
            raise ValueError("output_size or scale_factors must be provided")
        scale_factor = (
            scale_factors[0]
            if isinstance(scale_factors, (list, tuple))
            else scale_factors
        )
        out_w = int(in_w * scale_factor)

    if grad_output.shape[-1] != out_w:
        raise ValueError(
            f"expected grad_output width {out_w}, got {grad_output.shape[-1]}"
        )

    grad_out = grad_output.contiguous().view(n * c, out_w)
    if in_w == out_w:
        return grad_out.clone().view(n, c, in_w)

    grad_in = torch.empty(
        (n * c, in_w), device=grad_output.device, dtype=grad_output.dtype
    )
    scale = _scale_value(in_w, out_w, align_corners, scale_factors)
    with torch_device_fn.device(grad_output.device):
        if not align_corners and out_w == 2 * in_w and math.isclose(scale, 0.5):
            grid = (n * c, triton.cdiv(in_w, _SCALE2_BLOCK_W))
            _upsample_linear1d_backward_scale2_kernel[grid](
                grad_out,
                grad_in,
                n * c,
                in_w,
                out_w,
                True,
                _SCALE2_BLOCK_W,
                num_warps=4,
                num_stages=1,
            )
        elif not align_corners and in_w == 2 * out_w and math.isclose(scale, 2.0):
            grid = (n * c, triton.cdiv(in_w, _SCALE2_BLOCK_W))
            _upsample_linear1d_backward_scale2_kernel[grid](
                grad_out,
                grad_in,
                n * c,
                in_w,
                out_w,
                False,
                _SCALE2_BLOCK_W,
                num_warps=4,
                num_stages=1,
            )
        else:
            window = 2
            if out_w > 2 * in_w:
                window = math.ceil(out_w / in_w) + 2
            grid = (n * c, triton.cdiv(in_w, _GENERIC_BLOCK_W))
            _upsample_linear1d_backward_kernel[grid](
                grad_out,
                grad_in,
                n * c,
                in_w,
                out_w,
                scale,
                align_corners,
                window,
                _GENERIC_BLOCK_W,
                num_warps=4,
                num_stages=1,
            )

    return grad_in.view(n, c, in_w)
