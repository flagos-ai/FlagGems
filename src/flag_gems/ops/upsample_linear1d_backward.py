import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

device = device.name
logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=2, num_warps=8),
    ],
    key=["OL"],
)
@triton.jit
def upsample_linear1d_forward_kernel(
    input_ptr,
    output_ptr,
    N,
    C,
    IL,
    OL,
    scale,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for upsample_linear1d.
    Each program handles one (n, c) pair and iterates over output length.
    """
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc % C

    input_offset = (n * C + c) * IL
    output_offset = (n * C + c) * OL

    for block_start in range(0, OL, BLOCK_SIZE):
        o_idx = block_start + tl.arange(0, BLOCK_SIZE)
        mask = o_idx < OL

        o_idx_f32 = o_idx.to(tl.float32)

        if ALIGN_CORNERS:
            if OL > 1:
                src_idx = o_idx_f32 * ((IL - 1) / (OL - 1))
            else:
                src_idx = tl.zeros_like(o_idx_f32)
        else:
            src_idx = (o_idx_f32 + 0.5) * scale - 0.5
            src_idx = tl.maximum(src_idx, 0.0)

        src_idx_floor = tl.math.floor(src_idx).to(tl.int32)
        src_idx_floor = tl.minimum(src_idx_floor, IL - 1)
        src_idx_ceil = tl.minimum(src_idx_floor + 1, IL - 1)

        lambda1 = src_idx - src_idx_floor.to(tl.float32)
        lambda0 = 1.0 - lambda1

        val_floor = tl.load(input_ptr + input_offset + src_idx_floor, mask=mask, other=0.0)
        val_ceil = tl.load(input_ptr + input_offset + src_idx_ceil, mask=mask, other=0.0)

        val_floor = val_floor.to(tl.float32)
        val_ceil = val_ceil.to(tl.float32)

        result = val_floor * lambda0 + val_ceil * lambda1

        tl.store(output_ptr + output_offset + o_idx, result.to(output_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=2, num_warps=8),
    ],
    key=["OL"],
)
@triton.jit
def upsample_linear1d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    C,
    IL,
    OL,
    scale,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for upsample_linear1d.
    Each program handles one (n, c) pair and iterates over output length.
    Uses atomic add to accumulate gradients to input positions.
    """
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc % C

    grad_output_offset = (n * C + c) * OL
    grad_input_offset = (n * C + c) * IL

    for block_start in range(0, OL, BLOCK_SIZE):
        o_idx = block_start + tl.arange(0, BLOCK_SIZE)
        mask = o_idx < OL

        o_idx_f32 = o_idx.to(tl.float32)

        if ALIGN_CORNERS:
            if OL > 1:
                src_idx = o_idx_f32 * ((IL - 1) / (OL - 1))
            else:
                src_idx = tl.zeros_like(o_idx_f32)
        else:
            src_idx = (o_idx_f32 + 0.5) * scale - 0.5
            src_idx = tl.maximum(src_idx, 0.0)

        src_idx_floor = tl.math.floor(src_idx).to(tl.int32)
        src_idx_floor = tl.minimum(src_idx_floor, IL - 1)
        src_idx_ceil = tl.minimum(src_idx_floor + 1, IL - 1)

        lambda1 = src_idx - src_idx_floor.to(tl.float32)
        lambda0 = 1.0 - lambda1

        grad_out = tl.load(grad_output_ptr + grad_output_offset + o_idx, mask=mask, other=0.0)
        grad_out = grad_out.to(tl.float32)

        contrib0 = grad_out * lambda0
        contrib1 = grad_out * lambda1

        tl.atomic_add(grad_input_ptr + grad_input_offset + src_idx_floor, contrib0, mask=mask)
        tl.atomic_add(grad_input_ptr + grad_input_offset + src_idx_ceil, contrib1, mask=mask)


def upsample_linear1d_forward(
    input: torch.Tensor,
    output_size: List[int],
    align_corners: bool,
    scales: Optional[float] = None,
) -> torch.Tensor:
    """
    Forward pass for upsample_linear1d.
    """
    logger.debug("GEMS UPSAMPLE_LINEAR1D FORWARD")

    assert input.device.type == device
    assert input.ndim == 3, "input must be 3D tensor (N, C, IL)"

    N, C, IL = input.shape
    OL = output_size[0]

    if scales is not None:
        scale = float(IL) / float(OL)
    else:
        scale = float(IL) / float(OL)

    input = input.contiguous()
    output = torch.empty((N, C, OL), dtype=input.dtype, device=input.device)

    if output.numel() == 0:
        return output

    grid = (N * C,)

    with torch_device_fn.device(input.device):
        upsample_linear1d_forward_kernel[grid](
            input,
            output,
            N,
            C,
            IL,
            OL,
            scale,
            ALIGN_CORNERS=align_corners,
        )

    return output


def upsample_linear1d_backward_impl(
    grad_output: torch.Tensor,
    output_size: List[int],
    input_size: List[int],
    align_corners: bool,
    scales: Optional[float] = None,
) -> torch.Tensor:
    """
    Backward pass for upsample_linear1d.
    """
    logger.debug("GEMS UPSAMPLE_LINEAR1D_BACKWARD")

    assert grad_output.device.type == device
    assert grad_output.ndim == 3, "grad_output must be 3D tensor (N, C, OL)"

    N, C, OL = grad_output.shape
    IL = input_size[2]

    if scales is not None:
        scale = float(IL) / float(OL)
    else:
        scale = float(IL) / float(OL)

    grad_output = grad_output.contiguous()
    grad_input = torch.zeros(
        (N, C, IL), dtype=torch.float32, device=grad_output.device
    )

    if grad_output.numel() == 0 or IL == 0:
        return grad_input.to(grad_output.dtype)

    grid = (N * C,)

    with torch_device_fn.device(grad_output.device):
        upsample_linear1d_backward_kernel[grid](
            grad_output,
            grad_input,
            N,
            C,
            IL,
            OL,
            scale,
            ALIGN_CORNERS=align_corners,
        )

    return grad_input.to(grad_output.dtype)


class UpsampleLinear1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size, align_corners, scales=None):
        ctx.save_for_backward(input)
        ctx.output_size = output_size
        ctx.align_corners = align_corners
        ctx.scales = scales
        return upsample_linear1d_forward(input, output_size, align_corners, scales)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        output_size = ctx.output_size
        align_corners = ctx.align_corners
        scales = ctx.scales

        input_size = list(input.shape)
        grad_input = upsample_linear1d_backward_impl(
            grad_output, output_size, input_size, align_corners, scales
        )
        return grad_input, None, None, None


def upsample_linear1d(
    input: torch.Tensor,
    output_size: List[int],
    align_corners: bool,
    scales: Optional[float] = None,
) -> torch.Tensor:
    """
    Upsamples the input using linear interpolation in 1D.

    Args:
        input: Input tensor of shape (N, C, L)
        output_size: List with output length [OL]
        align_corners: Whether to align corners
        scales: Optional scale factor

    Returns:
        Upsampled tensor of shape (N, C, OL)
    """
    return UpsampleLinear1D.apply(input, output_size, align_corners, scales)


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size: List[int],
    input_size: List[int],
    align_corners: bool,
    scales: Optional[float] = None,
) -> torch.Tensor:
    """
    Backward pass for upsample_linear1d (standalone function).

    This can be called directly when needed for explicit backward computation.

    Args:
        grad_output: Gradient of the output tensor, shape (N, C, OL)
        output_size: List with output length [OL]
        input_size: List with input dimensions [N, C, IL]
        align_corners: Whether corners are aligned
        scales: Optional scale factor

    Returns:
        grad_input: Gradient of the input tensor, shape (N, C, IL)
    """
    return upsample_linear1d_backward_impl(
        grad_output, output_size, input_size, align_corners, scales
    )
