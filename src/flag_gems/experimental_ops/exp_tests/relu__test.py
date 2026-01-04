# RELU_ operator test

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
from flag_gems.experimental_ops.relu_ import relu_ as gems_relu_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.relu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.relu_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_relu_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.relu_
def test_perf_aten_relu_():
    # Define input generation logic matching the operator arguments
    def relu__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=relu__input_fn,
        op_name="relu_",
        torch_op=torch.ops.aten.relu_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
