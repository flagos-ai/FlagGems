import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_2d_with_bias(dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    # Test 2D input with bias
    # Common MLP hidden layer sizes to verify correctness with realistic tensor shapes
    batch_size = 16
    in_features = 128
    out_features = 64

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_2d_without_bias(dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    # Test 2D input without bias
    # Common MLP hidden layer sizes to verify correctness without bias term
    batch_size = 16
    in_features = 128
    out_features = 64

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.linear(input_tensor, weight)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_3d_with_bias(dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    # Test 3D input (batch > 1 in leading dims) with bias
    # Multi-dimensional batch input to verify batch flattening logic handles >2 dims
    batch1 = 4
    batch2 = 8
    in_features = 128
    out_features = 64

    input_tensor = torch.randn(
        (batch1, batch2, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_1d_with_bias(dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    # Test 1D input (single sample) with bias
    # Minimum-dimensional input to verify unsqueeze/squeeze logic for 1D inputs
    in_features = 128
    out_features = 64

    input_tensor = torch.randn((in_features,), dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)
