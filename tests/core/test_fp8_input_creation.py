# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Ascend FP8 input construction without device-side casts."""

import pytest
import torch

import flag_gems
from tests import kernelgen_test_utils as tu

pytestmark = [
    pytest.mark.fp8_input_creation,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend",
        reason="Ascend FP8 input construction regression",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 5)])
@pytest.mark.parametrize("value_range", tu.REQUIRED_RANGES + [["0", "0"]])
def test_fp8_input_has_target_storage_and_requested_values(dtype, shape, value_range):
    result = tu.make_input(dtype, shape, value_range)
    target = torch.device(flag_gems.device)
    if target.index is None:
        target = torch.device(target.type, torch.npu.current_device())
    assert result.device == target
    assert result.dtype == dtype
    assert tuple(result.shape) == shape
    assert result.is_contiguous()
    assert not result.requires_grad
    assert result.grad_fn is None
    values = result.cpu().float()
    assert torch.isfinite(values).all()
    low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
    assert (values >= low).all()
    assert (values <= high).all()
    if low == high:
        assert (values == low).all()


@pytest.mark.parametrize(
    "dtype,scenario",
    tu.special_value_cases([torch.float8_e4m3fn, torch.float8_e5m2]),
)
def test_fp8_special_input_preserves_encoded_values(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    expected = torch.tensor(payloads[scenario], dtype=torch.float32).to(dtype)
    result = tu.make_special_input(dtype, scenario)
    target = torch.device(flag_gems.device)
    if target.index is None:
        target = torch.device(target.type, torch.npu.current_device())
    assert result.device == target
    assert result.dtype == dtype
    assert result.shape == expected.shape
    assert result.is_contiguous()
    assert not result.requires_grad
    assert result.grad_fn is None
    # Compare encoded bytes so NaN, infinity and signed zero stay intact.
    assert torch.equal(result.cpu().view(torch.uint8), expected.view(torch.uint8))
