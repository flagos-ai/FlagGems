import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# FP16 representable range
FP16_MAX = 65504.0
FP16_MIN = -65504.0


def reference_saturate_weight_to_fp16(x):
    # torch._saturate_weight_to_fp16 has no usable CPU implementation, so the
    # golden reference is a plain clamp to the fp16 representable range.
    return torch.clamp(x, FP16_MIN, FP16_MAX)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_saturate_weight_to_fp16(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_saturate_weight_to_fp16_extreme_values(dtype):
    # Test with values that exceed fp16 range
    inp = torch.tensor(
        [
            [65504.0, 65505.0, 70000.0, 100000.0],
            [-65504.0, -65505.0, -70000.0, -100000.0],
            [0.0, 1.0, -1.0, 3.14159],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.saturate_weight_to_fp16
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_saturate_weight_to_fp16_boundary(dtype):
    # Test exact boundary values
    inp = torch.tensor(
        [FP16_MAX, FP16_MIN, FP16_MAX + 1, FP16_MIN - 1, 0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = reference_saturate_weight_to_fp16(ref_inp)
    res_out = flag_gems._saturate_weight_to_fp16(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
