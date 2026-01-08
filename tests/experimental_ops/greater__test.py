# GREATER_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.greater_ import greater__Scalar as gems_greater__Scalar
from flag_gems.experimental_ops.greater_ import greater__Tensor as gems_greater__Tensor

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


@pytest.mark.greater_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_greater__tensor(shape, dtype):
    base_self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base_other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = base_self.clone()
    ref_other = base_other.clone()
    ref_out = torch.ops.aten.greater_(ref_self, ref_other)

    act_self = base_self.clone()
    act_other = base_other.clone()
    with flag_gems.use_gems():
        act_out = gems_greater__Tensor(act_self, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.greater_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [-0.5, 0.0, 1.0])
def test_greater__scalar(shape, dtype, scalar):
    base_self = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = base_self.clone()
    ref_out = torch.ops.aten.greater_(ref_self, scalar)

    act_self = base_self.clone()
    with flag_gems.use_gems():
        act_out = gems_greater__Scalar(act_self, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.greater_
def test_perf_aten_greater_():
    # Define input generation logic matching the operator arguments
    def greater__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=greater__input_fn,
        op_name="greater_",
        torch_op=torch.ops.aten.greater_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
