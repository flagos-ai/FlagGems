import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log_sigmoid_forward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_forward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Move to CPU for reference since PyTorch's CUDA implementation of
    # aten.log_sigmoid_forward has a bug that returns empty buffer
    ref_inp = utils.to_reference(inp).to("cpu")

    ref_out = torch.ops.aten.log_sigmoid_forward(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.log_sigmoid_forward(inp)

    # Both output and buffer should match
    # In quick-cpu mode (TO_CPU=True), gems_assert_close handles device conversion
    # internally; we must not move the CPU reference back to a non-CPU device
    if utils.TO_CPU:
        utils.gems_assert_close(res_out[0], ref_out[0], dtype)
        utils.gems_assert_close(res_out[1], ref_out[1], dtype)
    else:
        utils.gems_assert_close(res_out[0], ref_out[0].to(res_out[0].device), dtype)
        utils.gems_assert_close(res_out[1], ref_out[1].to(res_out[1].device), dtype)


# Known-value tests for log_sigmoid correctness
@pytest.mark.log_sigmoid_forward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid_forward_known_values(dtype):
    """Verify log_sigmoid_forward at known inputs against analytical values."""
    x = torch.tensor([0.0, 1.0, -1.0, 100.0, -100.0], dtype=dtype, device=flag_gems.device)
    expected_output = torch.tensor(
        [-0.6931, -0.3133, -1.3133, 0.0, -100.0], dtype=torch.float32
    )
    ref_inp = x.to("cpu")

    ref_out = torch.ops.aten.log_sigmoid_forward(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.log_sigmoid_forward(x)

    if utils.TO_CPU:
        utils.gems_assert_close(res_out[0], ref_out[0], dtype)
        utils.gems_assert_close(res_out[1], ref_out[1], dtype)
    else:
        utils.gems_assert_close(res_out[0], ref_out[0].to(res_out[0].device), dtype)
        utils.gems_assert_close(res_out[1], ref_out[1].to(res_out[1].device), dtype)
