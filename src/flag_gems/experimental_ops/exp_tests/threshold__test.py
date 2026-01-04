# THRESHOLD_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.threshold_ import threshold_ as gems_threshold_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.threshold_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("threshold", [0.0, 0.25, -0.75, 0])
@pytest.mark.parametrize("value", [0.0, -0.5, 1.0, 2])
def test_threshold__tensor_inplace(shape, dtype, threshold, value):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.threshold_(ref_input, threshold, value)

    with flag_gems.use_gems():
        act_out = gems_threshold_(act_input, threshold, value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.threshold_
def test_perf_aten_threshold_():
    # Define input generation logic matching the operator arguments
    def threshold__input_fn(shape, dtype, device):
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        threshold = 0.0  # Example threshold
        value = 1.0  # Example value
        yield input_tensor, threshold, value

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=threshold__input_fn,
        op_name="threshold_",
        torch_op=torch.ops.aten.threshold_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
