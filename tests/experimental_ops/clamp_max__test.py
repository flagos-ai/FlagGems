# CLAMP_MAX_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.clamp_max_ import clamp_max_ as gems_clamp_max_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.clamp_max_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("max_value", [-0.25, 0.0, 0.75])
def test_clamp_max__scalar(shape, dtype, max_value):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.clamp_max_(ref_input, max_value)
    with flag_gems.use_gems():
        act_out = gems_clamp_max_(act_input, max_value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.clamp_max_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_clamp_max__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    max_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()
    ref_max = max_tensor.clone()
    act_max = max_tensor.clone()

    ref_out = torch.ops.aten.clamp_max_(ref_input, ref_max)
    with flag_gems.use_gems():
        act_out = gems_clamp_max_(act_input, act_max)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.clamp_max_
def test_perf_aten_clamp_max_():
    # Define input generation logic matching the operator arguments
    def clamp_max__input_fn(shape, dtype, device):
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        max_value = torch.rand(
            1, dtype=dtype, device=flag_gems.device
        )  # Scalar max value
        yield input_tensor, max_value

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=clamp_max__input_fn,
        op_name="clamp_max_",
        torch_op=torch.ops.aten.clamp_max_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
