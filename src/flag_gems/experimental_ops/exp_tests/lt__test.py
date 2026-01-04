# LT_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
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
from flag_gems.experimental_ops.lt_ import lt__Scalar, lt__Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.lt_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lt__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    ref_other = other_tensor.clone()

    ref_out = torch.ops.aten.lt_(ref_input, ref_other)

    with flag_gems.use_gems():
        act_out = lt__Tensor(input_tensor, other_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.lt_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other", [-1.0, 0.0, 1.5])
def test_lt__scalar(shape, dtype, other):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.lt_(ref_input, other)

    with flag_gems.use_gems():
        act_out = lt__Scalar(input_tensor, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.lt_
def test_perf_aten_lt_():
    # Define input generation logic matching the operator arguments
    def lt__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark - using lt__Tensor for performance test
    def lt__Tensor_wrapper(self, other):
        return lt__Tensor(self, other)

    bench = GenericBenchmark(
        input_fn=lt__input_fn,
        op_name="lt_",
        torch_op=torch.ops.aten.lt_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    # Replace the op with lt__Tensor
    bench.op = lt__Tensor_wrapper

    return bench.run()
