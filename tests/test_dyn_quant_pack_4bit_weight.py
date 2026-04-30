import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def native_dyn_quant_pack_4bit_weight(x: torch.Tensor, group_size: int = 128):
    """Native PyTorch implementation for reference."""
    original_shape = x.shape
    num_elements = x.shape[-1]
    num_groups = x.numel() // group_size

    # Flatten for processing
    x_flat = x.reshape(-1)

    # Output shapes
    qweight_shape = list(original_shape)
    qweight_shape[-1] = num_elements // 2
    qweight = torch.empty(qweight_shape, dtype=torch.uint8, device=x.device)

    scales_shape = list(original_shape[:-1])
    scales_shape.append(num_elements // group_size)
    scales = torch.empty(scales_shape, dtype=torch.float32, device=x.device)

    zeros = torch.full(scales_shape, 8, dtype=torch.uint8, device=x.device)

    # Process each group
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = x_flat[start:end].float()

        # Compute scale using absmax
        absmax = group.abs().max()
        scale = absmax / 7.0

        # Store scale
        scales.reshape(-1)[g] = scale

        # Quantize to 4-bit (-7 to 7)
        group_quant = torch.round(group / scale).clamp(-7, 7).to(torch.int8)

        # Pack two 4-bit values into one byte
        for i in range(group_size // 2):
            low = group_quant[i * 2].item()
            high = group_quant[i * 2 + 1].item() if i * 2 + 1 < group_size else low
            packed = (high << 4) | (low & 0x0F)
            qweight.reshape(-1)[g * (group_size // 2) + i] = packed

    return qweight, scales, zeros


# Test shapes for 4bit quantization
DYN_QUANT_4BIT_SHAPES = [
    (16, 128),
    (32, 256),
    (64, 512),
    (128, 1024),
    (1, 128),
    (8, 256),
]


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize("shape", DYN_QUANT_4BIT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [64, 128])
def test_accuracy_dyn_quant_pack_4bit_weight(shape, dtype, group_size):
    if shape[-1] % group_size != 0:
        pytest.skip(f"Shape {shape} not divisible by group_size {group_size}")

    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_qweight, ref_scales, ref_zeros = native_dyn_quant_pack_4bit_weight(
        x, group_size
    )

    with flag_gems.use_gems():
        res_qweight, res_scales, res_zeros = flag_gems._dyn_quant_pack_4bit_weight(
            x, group_size
        )

    # Compare scales (main tolerance for quantization)
    utils.gems_assert_close(res_scales, ref_scales, dtype=torch.float32, atol=1e-2)

    # Compare zeros (should always be 8)
    utils.gems_assert_equal(res_zeros, ref_zeros)
