from typing import Optional, Tuple

import pytest
import torch

import flag_gems
from flag_gems.utils.device_info import get_device_capability

device = flag_gems.device

M = [1, 40, 164, 512, 3454, 12027, 38594]
N = [128, 896, 2048, 8192]
# Test parameters
SHAPES = [(m, n) for m in M for n in N]
BLOCK_SIZES = [64, 128]
SCALE_FMTS = [None, "ue8m0"]


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


def torch_act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation for act_quant.
    Performs block-wise FP8 quantization.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, "Last dim must be divisible by block_size"

    FP8_MAX = 448.0
    FP8_MAX_INV = 1.0 / 448.0

    N = x.size(-1)
    x_2d = x.view(-1, N)
    M = x_2d.size(0)
    n_blocks = N // block_size

    # Reshape to (M, n_blocks, block_size) for block-wise processing
    x_blocked = x_2d.view(M, n_blocks, block_size).float()

    # Compute amax per block (M, n_blocks)
    amax = x_blocked.abs().amax(dim=-1)
    amax = torch.clamp(amax, min=1e-4)

    if scale_fmt is not None:
        # Use fast bit manipulation to match Triton implementation
        scale_raw = amax * FP8_MAX_INV

        # fast_log2_ceil: extract exponent and check mantissa bits
        bits_x = scale_raw.view(torch.int32)
        exp_x = (bits_x >> 23) & 0xFF
        man_bits = bits_x & ((1 << 23) - 1)
        log2_ceil = (exp_x - 127 + (man_bits != 0).int()).int()

        # fast_pow2: reconstruct power of 2 from exponent
        bits_scale = (log2_ceil + 127) << 23
        scale = bits_scale.view(torch.float32)
    else:
        scale = amax * FP8_MAX_INV

    # Quantize: y = clamp(x / scale, -FP8_MAX, FP8_MAX)
    y_blocked = x_blocked * (1.0 / scale.unsqueeze(-1))
    y_blocked = torch.clamp(y_blocked, -FP8_MAX, FP8_MAX)

    # Convert to FP8
    y = y_blocked.view(M, N).to(torch.float8_e4m3fn)
    y = y.view(x.shape)
    s = scale.to(torch.float32).view(*x.shape[:-1], n_blocks)

    return y, s


@pytest.mark.act_quant
# https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py#L188
@pytest.skip("Do not support fp8e4nv when capability < 89", not is_support_fp8e4nv())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("scale_fmt", SCALE_FMTS)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
def test_act_quant_accuracy(shape, block_size, scale_fmt, dtype):
    if shape[-1] % block_size != 0:
        pytest.skip(f"Shape {shape} not divisible by block_size {block_size}")

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)

    # Reference implementation
    ref_y, ref_s = torch_act_quant(x, block_size=block_size, scale_fmt=scale_fmt)

    # FlagGems implementation
    res_y, res_s = flag_gems.act_quant_triton(
        x, block_size=block_size, scale_fmt=scale_fmt
    )

    torch.testing.assert_close(ref_y.float(), res_y.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_s, res_s, rtol=1e-5, atol=1e-5)
