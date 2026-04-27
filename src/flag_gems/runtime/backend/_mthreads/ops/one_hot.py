import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)


@libentry()
@triton.jit
def one_hot_kernel_16(
    input_ptr,
    output_ptr,
    num_elements,
    actual_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_base = offsets * actual_classes

    class_offsets = tl.arange(0, 16)
    out_offsets = out_base[:, None] + class_offsets[None, :]
    values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
    valid_classes = class_offsets < actual_classes
    combined_mask = mask[:, None] & valid_classes[None, :]
    tl.store(output_ptr + out_offsets, values, mask=combined_mask)


@libentry()
@triton.jit
def one_hot_kernel_32(
    input_ptr,
    output_ptr,
    num_elements,
    actual_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_base = offsets * actual_classes

    class_offsets = tl.arange(0, 32)
    out_offsets = out_base[:, None] + class_offsets[None, :]
    values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
    valid_classes = class_offsets < actual_classes
    combined_mask = mask[:, None] & valid_classes[None, :]
    tl.store(output_ptr + out_offsets, values, mask=combined_mask)


@libentry()
@triton.jit
def one_hot_kernel_64(
    input_ptr,
    output_ptr,
    num_elements,
    actual_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_base = offsets * actual_classes

    class_offsets = tl.arange(0, 64)
    out_offsets = out_base[:, None] + class_offsets[None, :]
    values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
    valid_classes = class_offsets < actual_classes
    combined_mask = mask[:, None] & valid_classes[None, :]
    tl.store(output_ptr + out_offsets, values, mask=combined_mask)


@libentry()
@triton.jit
def one_hot_set_one_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that only writes 1s to the correct positions.
    Output tensor should be pre-initialized with zeros.
    """
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_offsets = offsets * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS ONE_HOT")

    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    # Only compute max when necessary (num_classes=-1)
    if num_classes == -1:
        # Only compute max to infer num_classes
        maxv = int(tensor.max().item())
        num_classes = maxv + 1

    # Validate that all indices are non-negative
    if (tensor < 0).any():
        raise RuntimeError("Class values must be non-negative.")

    # Validate that all indices are within num_classes range
    if (tensor >= num_classes).any():
        raise RuntimeError("Class values must be smaller than num_classes.")

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    # CPU tensor handling
    if tensor.device.type == "cpu":
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    # Flatten input for kernel processing
    flat_input = tensor.contiguous().view(-1)
    num_elements = flat_input.numel()

    # Choose kernel based on num_classes
    with torch_device_fn.device(tensor.device):
        if num_classes <= 16:
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = 128
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_kernel_16[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_classes <= 32:
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = 128
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_kernel_32[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif num_classes <= 64:
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = 128
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_kernel_64[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # For large num_classes, use zeros + set ones
            out = torch.zeros(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_set_one_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(*tensor.shape, num_classes)
