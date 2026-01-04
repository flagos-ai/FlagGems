# FILL_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.fill_ import fill__Scalar, fill__Tensor

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


@pytest.mark.fill_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, 1.25, -3.5])
def test_fill__scalar(shape, dtype, value):
    x_ref = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_act = x_ref.clone()

    ref_out = torch.ops.aten.fill_(x_ref, value)

    with flag_gems.use_gems():
        act_out = fill__Scalar(x_act, value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.fill_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, 2.0, -1.5])
def test_fill__tensor(shape, dtype, value):
    x_ref = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_act = x_ref.clone()

    val_ref = torch.tensor(value, dtype=dtype, device=flag_gems.device)
    val_act = torch.tensor(value, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.fill_(x_ref, val_ref)

    with flag_gems.use_gems():
        act_out = fill__Tensor(x_act, val_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.fill_
def test_perf_aten_fill_():
    # Define input generation logic matching the operator arguments
    def fill__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        value = 1.0  # Scalar value to fill
        yield inp, value

    # Initialize benchmark - using fill__Scalar for performance test
    def fill__Scalar_wrapper(self, value):
        return fill__Scalar(self, value)

    bench = GenericBenchmark(
        input_fn=fill__input_fn,
        op_name="fill_",
        torch_op=torch.ops.aten.fill_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    # Replace the op with fill__Scalar
    bench.op = fill__Scalar_wrapper

    return bench.run()
