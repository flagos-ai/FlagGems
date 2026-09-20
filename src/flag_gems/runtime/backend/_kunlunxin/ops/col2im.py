import logging
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=1, num_warps=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=1, num_warps=4),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def col2im_kernel(
    input_ptr,
    output_ptr,
    # Input tensor info
    in_stride_n,
    in_stride_ck,
    in_stride_l,
    # Output tensor info
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Shapes
    batch_size,
    channels,
    out_h,
    out_w,
    L_h,
    L_w,
    # Kernel parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Tiling
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Each program handles one (batch, channel) slice and a block of output positions.
    #
    # Kunlunxin XPU note: tl.load(ptr, mask=...) is NOT reliable on this backend
    # when the address is out-of-bounds even if mask=False -- a real load may
    # still be issued, and the masked-out lane may end up reading garbage memory.
    # The fix here is to (1) clamp the load address into the valid range so the
    # backing read is always in-bounds, and (2) gate the contribution with a
    # pure-arithmetic tl.where instead of relying on the load mask.
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    n_idx = pid_nc // channels
    c_idx = pid_nc % channels

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Accumulator for output values
    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Base pointer to input for this batch
    input_base_ptr = input_ptr + n_idx * in_stride_n

    # Iterate over kernel positions
    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            # l_h * stride_h = h + padding_h - kh * dilation_h
            # l_w * stride_w = w + padding_w - kw * dilation_w
            h_num = h_out_offsets[:, None] + padding_h - kh * dilation_h
            w_num = w_out_offsets[None, :] + padding_w - kw * dilation_w

            # Check divisibility by stride
            h_valid = (h_num % stride_h) == 0
            w_valid = (w_num % stride_w) == 0

            # Compute l_h and l_w
            l_h = h_num // stride_h
            l_w = w_num // stride_w

            # Check bounds for l_h and l_w
            l_h_valid = (l_h >= 0) & (l_h < L_h)
            l_w_valid = (l_w >= 0) & (l_w < L_w)

            # Also explicitly mask BLOCK tail lanes that exceed out_h/out_w --
            # otherwise their accumulator entries can leak into vectorized
            # register groups on XPU.
            h_in_range = h_out_offsets[:, None] < out_h
            w_in_range = w_out_offsets[None, :] < out_w

            # Combined arithmetic-only mask (no addressing semantics).
            valid_mask = (
                h_valid & w_valid & l_h_valid & l_w_valid & h_in_range & w_in_range
            )

            # c_k = c * kernel_h * kernel_w + kh * kernel_w + kw
            c_k = c_idx * kernel_h * kernel_w + kh * kernel_w + kw

            # ---- Key 1: clamp l_h / l_w into [0, L-1] so the load address is
            # always inside the input tensor.  Whether the value should
            # actually contribute is decided by valid_mask below, NOT by the
            # load mask.
            safe_l_h = tl.maximum(tl.minimum(l_h, L_h - 1), 0)
            safe_l_w = tl.maximum(tl.minimum(l_w, L_w - 1), 0)
            safe_l_idx = safe_l_h * L_w + safe_l_w
            input_offset = c_k * in_stride_ck + safe_l_idx * in_stride_l

            # ---- Key 2: tl.load WITHOUT mask, bypassing the unreliable
            # masked-load path on XPU.
            input_val = tl.load(input_base_ptr + input_offset)

            # ---- Key 3: arithmetic mask via tl.where (no memory semantics,
            # consistent across backends).
            zero = tl.zeros_like(input_val)
            input_val = tl.where(valid_mask, input_val, zero)

            # Accumulate
            sum_acc += input_val

    # Store output (out_mask is fine here -- store-side masks are honored).
    out_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    out_offset = (
        h_out_offsets[:, None] * out_stride_h + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(
        out_base_ptr + out_offset,
        sum_acc.to(output_ptr.type.element_ty),
        mask=out_mask,
    )


def _parse_col2im_params(output_size, kernel_size, dilation, padding, stride):
    """Parse and validate col2im parameters."""

    def _to_pair(val, name):
        if isinstance(val, int):
            return val, val
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return tuple(val)
        raise ValueError(f"Invalid {name}: {val}")

    out_h, out_w = _to_pair(output_size, "output_size")
    kernel_h, kernel_w = _to_pair(kernel_size, "kernel_size")
    dilation_h, dilation_w = _to_pair(dilation, "dilation")
    padding_h, padding_w = _to_pair(padding, "padding")
    stride_h, stride_w = _to_pair(stride, "stride")

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"stride must be positive, got ({stride_h}, {stride_w})")
    if padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, got ({padding_h}, {padding_w})"
        )
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(f"dilation must be positive, got ({dilation_h}, {dilation_w})")

    return (
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        padding_h,
        padding_w,
        stride_h,
        stride_w,
    )


def col2im(
    input: torch.Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> torch.Tensor:
    """
    Combines an array of sliding local blocks into a large containing tensor.

    Kunlunxin XPU override of the generic col2im -- adopts a
    "safe-address + arithmetic-mask" strategy in the kernel because the
    XPU backend does not honor tl.load(ptr, mask=...) for out-of-bounds
    addresses.  Default-branch behavior is unchanged.
    """
    logger.debug("GEMS_KUNLUNXIN COL2IM")

    # Parse parameters
    (
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        padding_h,
        padding_w,
        stride_h,
        stride_w,
    ) = _parse_col2im_params(output_size, kernel_size, dilation, padding, stride)

    # Input shape validation
    if input.dim() != 3:
        raise ValueError(f"Expected 3D input, got {input.dim()}D")

    batch_size, ck, L = input.shape

    # L_h = (out_h + 2*padding_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1
    L_h = (out_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    L_w = (out_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    expected_L = L_h * L_w

    if L != expected_L:
        raise ValueError(
            f"Input size mismatch: expected L={expected_L} (L_h={L_h}, L_w={L_w}), got L={L}"
        )

    kernel_size_total = kernel_h * kernel_w
    if ck % kernel_size_total != 0:
        raise ValueError(
            f"Input dimension 1 ({ck}) must be divisible by kernel_size ({kernel_size_total})"
        )
    channels = ck // kernel_size_total

    # Make input contiguous
    input = input.contiguous()

    # Allocate output
    output = torch.empty(
        (batch_size, channels, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )

    if output.numel() == 0:
        return output

    # Launch kernel
    grid = lambda meta: (  # noqa: E731
        batch_size * channels,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    col2im_kernel[grid](
        input,
        output,
        # Input strides
        input.stride(0),
        input.stride(1),
        input.stride(2),
        # Output strides
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        # Shapes
        batch_size,
        channels,
        out_h,
        out_w,
        L_h,
        L_w,
        # Kernel parameters
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )

    return output
