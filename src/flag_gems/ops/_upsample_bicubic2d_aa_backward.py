import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

device = device.name

logger = logging.getLogger(__name__)


def bicubic_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


@triton.jit
def compute_bicubic_weight(pos, span_size, start_minus_center, invscale, a):
    """Compute bicubic weight for position pos in span."""
    w = tl.abs((pos + start_minus_center + 0.5) * invscale)
    weight = tl.where(
        pos < span_size,
        tl.where(
            w < 1.0,
            ((a + 2) * w - (a + 3)) * w * w + 1,
            tl.where(w < 2.0, (((w - 5) * w + 8) * w - 4) * a, 0.0),
        ),
        0.0,
    )
    return weight


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bicubic2d_aa"),
    key=["N", "C", "IH", "IW"],
)
@triton.jit
def upsample_bicubic2d_aa_backward_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Backward kernel for upsampling (reciprocal_scale < 1)."""
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    iw = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % IW
    ih = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % IH

    support_w = 2.0
    support_h = 2.0
    invscale_w = 1.0
    invscale_h = 1.0
    a = -0.5

    for n in range(0, N, 1):
        for c in range(0, C, 1):
            grad_acc = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)

            # Iterate over all output pixels
            for oh in range(0, OH, 1):
                # Compute span for this output pixel
                center_h = (oh + 0.5) * reciprocal_scale_h
                span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
                span_size_h = (
                    tl.minimum(center_h + support_h + 0.5, IH) - span_start_h
                ).to(tl.int32)
                start_minus_center_h = span_start_h - center_h

                # Check if current ih is in this span
                pos_h = ih - span_start_h
                in_span_h = (pos_h >= 0) & (pos_h < span_size_h)

                # Compute weight for h dimension
                weight_h = compute_bicubic_weight(
                    pos_h, span_size_h, start_minus_center_h, invscale_h, a
                )

                # Compute normalization factor for h
                weight_h_total = tl.zeros((BLOCK_Y,), dtype=tl.float32)
                for k in range(0, 5, 1):
                    w_k = compute_bicubic_weight(
                        k, span_size_h, start_minus_center_h, invscale_h, a
                    )
                    weight_h_total += w_k
                weight_h_total = tl.where(weight_h_total != 0, weight_h_total, 1.0)
                weight_h_normalized = weight_h / weight_h_total

                for ow in range(0, OW, 1):
                    # Compute span for this output pixel
                    center_w = (ow + 0.5) * reciprocal_scale_w
                    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
                    span_size_w = (
                        tl.minimum(center_w + support_w + 0.5, IW) - span_start_w
                    ).to(tl.int32)
                    start_minus_center_w = span_start_w - center_w

                    # Check if current iw is in this span
                    pos_w = iw - span_start_w
                    in_span_w = (pos_w >= 0) & (pos_w < span_size_w)

                    # Compute weight for w dimension
                    weight_w = compute_bicubic_weight(
                        pos_w, span_size_w, start_minus_center_w, invscale_w, a
                    )

                    # Compute normalization factor for w
                    weight_w_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                    for k in range(0, 5, 1):
                        w_k = compute_bicubic_weight(
                            k, span_size_w, start_minus_center_w, invscale_w, a
                        )
                        weight_w_total += w_k
                    weight_w_total = tl.where(weight_w_total != 0, weight_w_total, 1.0)
                    weight_w_normalized = weight_w / weight_w_total

                    # Combined weight and mask
                    weight = weight_h_normalized[:, None] * weight_w_normalized[None, :]
                    valid = in_span_h[:, None] & in_span_w[None, :]

                    # Load grad_output and accumulate
                    grad_offset = (n * C + c) * OH * OW + oh * OW + ow
                    grad_val = tl.load(ptr_grad_output + grad_offset).to(tl.float32)
                    grad_acc += tl.where(valid, grad_val * weight, 0.0)

            # Store gradient
            offset_i = ((n * C + c) * IH + ih[:, None]) * IW + iw[None, :]
            tl.store(
                ptr_grad_input + offset_i, grad_acc.to(ptr_grad_input.type.element_ty)
            )


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bicubic2d_aa"),
    key=["N", "C", "IH", "IW"],
)
@triton.jit
def general_bicubic2d_aa_backward_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Backward kernel for general case (including downsampling)."""
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    iw = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % IW
    ih = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % IH

    support_w = 2 * reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 2.0
    support_h = 2 * reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 2.0

    invscale_w = 1.0 / reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 1.0
    invscale_h = 1.0 / reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 1.0

    interpolate_size_h = (support_h + 0.5).to(tl.int32) * 2 + 1
    interpolate_size_w = (support_w + 0.5).to(tl.int32) * 2 + 1

    a = -0.5

    for n in range(0, N, 1):
        for c in range(0, C, 1):
            grad_acc = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)

            # Iterate over all output pixels
            for oh in range(0, OH, 1):
                # Compute span for this output pixel
                center_h = (oh + 0.5) * reciprocal_scale_h
                span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
                span_size_h = (
                    tl.minimum(center_h + support_h + 0.5, IH) - span_start_h
                ).to(tl.int32)
                start_minus_center_h = span_start_h - center_h

                # Check if current ih is in this span
                pos_h = ih - span_start_h
                in_span_h = (pos_h >= 0) & (pos_h < span_size_h)

                # Compute weight for h dimension
                weight_h = compute_bicubic_weight(
                    pos_h, span_size_h, start_minus_center_h, invscale_h, a
                )

                # Compute normalization factor for h
                weight_h_total = tl.zeros((BLOCK_Y,), dtype=tl.float32)
                for k in range(0, interpolate_size_h, 1):
                    w_k = compute_bicubic_weight(
                        k, span_size_h, start_minus_center_h, invscale_h, a
                    )
                    weight_h_total += w_k
                weight_h_total = tl.where(weight_h_total != 0, weight_h_total, 1.0)
                weight_h_normalized = weight_h / weight_h_total

                for ow in range(0, OW, 1):
                    # Compute span for this output pixel
                    center_w = (ow + 0.5) * reciprocal_scale_w
                    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
                    span_size_w = (
                        tl.minimum(center_w + support_w + 0.5, IW) - span_start_w
                    ).to(tl.int32)
                    start_minus_center_w = span_start_w - center_w

                    # Check if current iw is in this span
                    pos_w = iw - span_start_w
                    in_span_w = (pos_w >= 0) & (pos_w < span_size_w)

                    # Compute weight for w dimension
                    weight_w = compute_bicubic_weight(
                        pos_w, span_size_w, start_minus_center_w, invscale_w, a
                    )

                    # Compute normalization factor for w
                    weight_w_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                    for k in range(0, interpolate_size_w, 1):
                        w_k = compute_bicubic_weight(
                            k, span_size_w, start_minus_center_w, invscale_w, a
                        )
                        weight_w_total += w_k
                    weight_w_total = tl.where(weight_w_total != 0, weight_w_total, 1.0)
                    weight_w_normalized = weight_w / weight_w_total

                    # Combined weight and mask
                    weight = weight_h_normalized[:, None] * weight_w_normalized[None, :]
                    valid = in_span_h[:, None] & in_span_w[None, :]

                    # Load grad_output and accumulate
                    grad_offset = (n * C + c) * OH * OW + oh * OW + ow
                    grad_val = tl.load(ptr_grad_output + grad_offset).to(tl.float32)
                    grad_acc += tl.where(valid, grad_val * weight, 0.0)

            # Store gradient
            offset_i = ((n * C + c) * IH + ih[:, None]) * IW + iw[None, :]
            tl.store(
                ptr_grad_input + offset_i, grad_acc.to(ptr_grad_input.type.element_ty)
            )


def _upsample_bicubic2d_aa_backward(
    grad_output: torch.Tensor,
    output_size: List[int],
    input_size: List[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS UPSAMPLE BICUBIC2D AA BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(input_size) == 4, "The len of input_size must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    N, C, IH, IW = input_size
    OH, OW = output_size

    reciprocal_scale_h = bicubic_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bicubic_reciprocal_scale(IW, OW, align_corners, scales_w)

    # Allocate grad_input
    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )

    # Handle empty tensor case
    if grad_input.numel() == 0:
        return grad_input

    grad_output = grad_output.contiguous()

    grid = lambda META: (
        triton.cdiv(IW, META["BLOCK_X"]),
        triton.cdiv(IH, META["BLOCK_Y"]),
    )

    # Use general kernel for downsampling (reciprocal_scale >= 1) or upsample kernel
    kernel = (
        general_bicubic2d_aa_backward_kernel
        if (reciprocal_scale_w >= 1.0) or (reciprocal_scale_h >= 1.0)
        else upsample_bicubic2d_aa_backward_kernel
    )

    with torch_device_fn.device(grad_output.device):
        kernel[grid](
            grad_input,
            grad_output,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            reciprocal_scale_h,
            reciprocal_scale_w,
        )

    return grad_input
