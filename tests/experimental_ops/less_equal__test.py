# LESS_EQUAL_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.less_equal_ import (
    less_equal__Scalar as gems_less_equal__Scalar,
)
from flag_gems.experimental_ops.less_equal_ import (
    less_equal__Tensor as gems_less_equal__Tensor,
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


@pytest.mark.less_equal_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_less_equal__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    ref_other = other_tensor.clone()
    act_input = input_tensor.clone()
    act_other = other_tensor.clone()

    ref_out = torch.ops.aten.less_equal_(ref_input, ref_other)

    with flag_gems.use_gems():
        act_out = gems_less_equal__Tensor(act_input, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.less_equal_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [-0.5, 0.0, 1.0])
def test_less_equal__scalar(shape, dtype, scalar):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.less_equal_(ref_input, scalar)

    with flag_gems.use_gems():
        act_out = gems_less_equal__Scalar(act_input, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.less_equal_
def test_perf_aten_less_equal_():
    # Define input generation logic matching the operator arguments
    def less_equal__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=less_equal__input_fn,
        op_name="less_equal_",
        torch_op=torch.ops.aten.less_equal_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
