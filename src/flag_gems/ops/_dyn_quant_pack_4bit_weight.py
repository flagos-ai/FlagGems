import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _dyn_quant_pack_4bit_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    quant_tmp_ptr,
    num_groups,
    group_size,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for dynamic 4-bit weight quantization and packing.

    This kernel performs:
    1. Split input into groups of `group_size` elements
    2. Compute per-group scale (absmax / 7.0 for symmetric quantization)
    3. Quantize elements to 4-bit (0-15)
    4. Pack two 4-bit values into one byte
    """
    pid = tl.program_id(0)
    group_id = pid

    if group_id >= num_groups:
        return

    # Compute base pointer for this group
    x_base = x_ptr + group_id * group_size
    qweight_base = qweight_ptr + group_id * (group_size // 2)
    scales_base = scales_ptr + group_id
    zeros_base = zeros_ptr + group_id
    quant_tmp_base = quant_tmp_ptr + group_id * group_size

    # Load elements for this group
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < group_size
    x_vals = tl.load(x_base + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute scale using absmax
    absmax = tl.max(tl.abs(x_vals))
    # For symmetric 4-bit quantization, range is [-7, 7]
    scale = absmax / 7.0

    # Store scale and zero point (zero = 8 for symmetric)
    tl.store(scales_base, scale)
    tl.store(zeros_base, tl.constexpr(8))

    # Quantize to 4-bit using floor(x + 0.5) for rounding
    x_quant = tl.math.floor(x_vals / scale + 0.5)
    x_quant = tl.clamp(x_quant, -7.0, 7.0)
    x_quant_int = x_quant.to(tl.int8)

    # Store quantized values to temporary buffer
    tl.store(quant_tmp_base + col_offsets, x_quant_int, mask=mask)

    # Split into low and high nibbles
    half = group_size // 2
    pack_offsets = tl.arange(0, BLOCK_SIZE // 2)
    pack_mask = pack_offsets < half

    # Load quantized values for packing
    low_vals = tl.load(quant_tmp_base + pack_offsets, mask=pack_mask, other=0).to(tl.int8)
    high_vals = tl.load(quant_tmp_base + half + pack_offsets, mask=pack_mask, other=0).to(tl.int8)

    # Pack: (high << 4) | (low & 0x0F)
    packed = (high_vals << 4) | (low_vals & 0x0F)
    tl.store(qweight_base + pack_offsets, packed.to(tl.uint8), mask=pack_mask)


def _dyn_quant_pack_4bit_weight(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic 4-bit weight quantization and packing.

    This function performs dynamic quantization of weights to 4-bit format.
    It uses symmetric quantization with per-group scales.

    Args:
        x: Input tensor (weight matrix), supports fp16, bf16, fp32
        group_size: Size of quantization groups (default: 128)

    Returns:
        qweight: Quantized and packed weights (uint8 tensor)
        scales: Per-group scales (float32 tensor)
        zeros: Per-group zero points (uint8 tensor, always 8 for symmetric)
    """
    logger.debug("GEMS _DYN_QUANT_PACK_4BIT_WEIGHT")

    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.ndim >= 2, "Input must be at least 2D"

    # Flatten the input for quantization (process last dimension as groups)
    original_shape = x.shape
    num_elements = x.shape[-1]
    num_groups = x.numel() // group_size

    assert num_elements % group_size == 0, (
        f"Last dimension {num_elements} must be divisible by group_size {group_size}"
    )

    # Reshape to process all elements as groups
    x_flat = x.reshape(-1)

    # Output shapes
    qweight_shape = list(original_shape)
    qweight_shape[-1] = num_elements // 2  # Pack 2 elements into 1 byte
    qweight = torch.empty(qweight_shape, dtype=torch.uint8, device=x.device)

    scales_shape = list(original_shape[:-1])
    scales_shape.append(num_elements // group_size)
    scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    zeros_shape = scales_shape
    zeros = torch.zeros(zeros_shape, dtype=torch.uint8, device=x.device)

    # Temporary buffer for quantized values
    quant_tmp = torch.empty(x_flat.shape, dtype=torch.int8, device=x.device).contiguous()

    # Kernel configuration
    BLOCK_SIZE = triton.next_power_of_2(group_size)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    grid = (num_groups,)

    _dyn_quant_pack_4bit_kernel[grid](
        x_flat,
        qweight.reshape(-1),
        scales.reshape(-1),
        zeros.reshape(-1),
        quant_tmp,
        num_groups,
        group_size,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return qweight, scales, zeros


def dyn_quant_pack_4bit_weight(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public wrapper for dynamic 4-bit weight quantization."""
    return _dyn_quant_pack_4bit_weight(x, group_size)