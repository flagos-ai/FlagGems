# LEAKY_RELU operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.leaky_relu import leaky_relu as gems_leaky_relu
from flag_gems.experimental_ops.leaky_relu import leaky_relu_out as gems_leaky_relu_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("negative_slope", [0.0, 0.01, 0.2])
def test_leaky_relu_tensor(shape, dtype, negative_slope):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.leaky_relu(ref_input, negative_slope)
    with flag_gems.use_gems():
        act_out = gems_leaky_relu(input_tensor, negative_slope)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("negative_slope", [0.0, 0.01, 0.2])
def test_leaky_relu_out(shape, dtype, negative_slope):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    ref_out_buf = torch.empty_like(ref_input)
    ref_out = torch.ops.aten.leaky_relu.out(ref_input, negative_slope, out=ref_out_buf)
    with flag_gems.use_gems():
        act_out_buf = torch.empty_like(input_tensor)
        act_out = gems_leaky_relu_out(input_tensor, negative_slope, act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)


@pytest.mark.leaky_relu
def test_perf_aten_leaky_relu():
    # Define input generation logic matching the operator arguments
    def leaky_relu_input_fn(shape, dtype, device):
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield input_tensor,

        # Yield negative_slope as a parameter
        for negative_slope in [0.0, 0.01, 0.2]:
            yield input_tensor, negative_slope

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=leaky_relu_input_fn,
        op_name="leaky_relu",
        torch_op=torch.ops.aten.leaky_relu,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
