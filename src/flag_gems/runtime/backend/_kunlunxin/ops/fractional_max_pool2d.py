import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    alpha_height,
    alpha_width,
):
    output_column = tl.program_id(0)
    row_id = tl.program_id(1)
    channel = tl.program_id(2)
    batch = row_id // output_height
    output_row = row_id % output_height
    sample_offset = (batch * input_channels + channel) * 2
    sample_height = tl.load(random_samples_ptr + sample_offset).to(tl.float32)
    sample_width = tl.load(random_samples_ptr + sample_offset + 1).to(tl.float32)

    start_height = (
        (output_row.to(tl.float32) + sample_height) * alpha_height
    ).to(tl.int32) - (sample_height * alpha_height).to(tl.int32)
    start_width = (
        (output_column.to(tl.float32) + sample_width) * alpha_width
    ).to(tl.int32) - (sample_width * alpha_width).to(tl.int32)
    start_height = tl.where(
        output_row == output_height - 1,
        input_height - kernel_height,
        start_height,
    )
    start_width = tl.where(
        output_column == output_width - 1,
        input_width - kernel_width,
        start_width,
    )

    max_value = tl.full((), -float("inf"), tl.float32)
    max_index = tl.full((), -1, tl.int64)
    for kernel_row in tl.static_range(0, kernel_height):
        for kernel_column in tl.static_range(0, kernel_width):
            input_row = start_height + kernel_row
            input_column = start_width + kernel_column
            input_offset = (
                ((batch * input_channels + channel) * input_height + input_row)
                * input_width
                + input_column
            )
            value = tl.load(input_ptr + input_offset).to(tl.float32)
            update = value > max_value
            max_value = tl.where(update, value, max_value)
            max_index = tl.where(
                update,
                input_row * input_width + input_column,
                max_index,
            )

    output_offset = (
        ((batch * input_channels + channel) * output_height + output_row)
        * output_width
        + output_column
    )
    tl.store(output_ptr + output_offset, max_value)
    tl.store(indices_ptr + output_offset, max_index)


@libentry()
@triton.jit
def _fractional_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    output_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    input_column = tl.program_id(0)
    row_id = tl.program_id(1)
    channel = tl.program_id(2)
    batch = row_id // input_height
    input_row = row_id % input_height
    target_index = input_row * input_width + input_column
    output_elements = output_height * output_width
    gradient = tl.zeros((), dtype=tl.float32)

    for block in tl.static_range(0, output_blocks):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < output_elements
        safe_offsets = tl.where(mask, offsets, tl.zeros_like(offsets))
        base = (batch * input_channels + channel) * output_elements
        indices = tl.load(indices_ptr + base + safe_offsets)
        values = tl.load(grad_output_ptr + base + safe_offsets).to(tl.float32)
        indices = tl.where(mask, indices, -1)
        values = tl.where(mask, values, 0.0)
        gradient += tl.sum(
            tl.where(indices == target_index, values, 0.0), axis=0
        )

    output_offset = (
        ((batch * input_channels + channel) * input_height + input_row)
        * input_width
        + input_column
    )
    tl.store(grad_input_ptr + output_offset, gradient)


def _parse_size(value):
    if isinstance(value, (int, float)):
        return value, value
    return value[0], value[1]


def fractional_max_pool2d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=True,
    _random_samples=None,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL2D")
    if isinstance(output_ratio, torch.Tensor) and _random_samples is None:
        _random_samples = output_ratio
        output_ratio = None
    assert input.dim() == 4, f"Expected 4D input, got {input.dim()}D"
    input = input.contiguous()
    batch_size, channels, input_height, input_width = input.shape
    kernel_height, kernel_width = _parse_size(kernel_size)
    if output_size is not None:
        output_height, output_width = _parse_size(output_size)
    elif output_ratio is not None:
        ratio_height, ratio_width = _parse_size(output_ratio)
        output_height = int(input_height * ratio_height)
        output_width = int(input_width * ratio_width)
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    assert output_height + kernel_height - 1 <= input_height
    assert output_width + kernel_width - 1 <= input_width

    if _random_samples is None:
        _random_samples = torch.rand(
            batch_size,
            channels,
            2,
            device=input.device,
            dtype=input.dtype,
        )
    else:
        assert _random_samples.shape == (batch_size, channels, 2)
        _random_samples = _random_samples.to(dtype=input.dtype).contiguous()

    output = torch.empty(
        (batch_size, channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (batch_size, channels, output_height, output_width),
        device=input.device,
        dtype=torch.int64,
    )
    alpha_height = (
        (input_height - kernel_height) / (output_height - 1)
        if output_height > 1
        else 0.0
    )
    alpha_width = (
        (input_width - kernel_width) / (output_width - 1)
        if output_width > 1
        else 0.0
    )
    grid = (output_width, batch_size * output_height, channels)
    with torch_device_fn.device(input.device):
        _fractional_max_pool2d_forward_kernel[grid](
            input,
            output,
            indices,
            _random_samples.reshape(batch_size * channels, 2),
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            alpha_height,
            alpha_width,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    if return_indices:
        return output, indices
    return output


def fractional_max_pool2d_backward(
    grad_output,
    input,
    kernel_size,
    output_size,
    indices,
):
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL2D_BACKWARD")
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    batch_size, channels, input_height, input_width = input.shape
    output_height, output_width = _parse_size(output_size)
    grad_input = torch.empty_like(input)
    block = 32
    output_blocks = triton.cdiv(output_height * output_width, block)
    grid = (input_width, batch_size * input_height, channels)
    with torch_device_fn.device(input.device):
        _fractional_max_pool2d_backward_kernel[grid](
            grad_output,
            indices,
            grad_input,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            output_blocks,
            BLOCK=block,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return grad_input
