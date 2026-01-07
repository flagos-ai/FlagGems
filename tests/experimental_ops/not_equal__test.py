# NOT_EQUAL_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.not_equal_ import (
    not_equal__Scalar as gems_not_equal__Scalar,
)
from flag_gems.experimental_ops.not_equal_ import (
    not_equal__Tensor as gems_not_equal__Tensor,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import gems_assert_close


except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.not_equal_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other", [0.0, 1.0, -2.5])
def test_not_equal__scalar(shape, dtype, other):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.not_equal_(ref_input, other)
    with flag_gems.use_gems():
        act_out = gems_not_equal__Scalar(input_tensor, other)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.not_equal_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_not_equal__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    ref_other = other_tensor.clone()
    ref_out = torch.ops.aten.not_equal_(ref_input, ref_other)
    with flag_gems.use_gems():
        act_out = gems_not_equal__Tensor(input_tensor, other_tensor)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.not_equal_
def test_perf_aten_not_equal_():
    # Define input generation logic matching the operator arguments
    def not_equal__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=not_equal__input_fn,
        op_name="not_equal_",
        torch_op=torch.ops.aten.not_equal_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
