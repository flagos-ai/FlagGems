# GREATER_EQUAL_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.greater_equal_ import (
    greater_equal__Scalar as gems_greater_equal__scalar,
)
from flag_gems.experimental_ops.greater_equal_ import (
    greater_equal__Tensor as gems_greater_equal__tensor,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.greater_equal_
@pytest.mark.parametrize(
    "self_shape,other_shape",
    [
        ((2, 3), (2, 3)),
        ((2, 3), (1, 3)),
        ((128, 256), (1, 256)),
        ((128, 256), (128, 1)),
        ((1024, 1024), (1, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_greater_equal__tensor(self_shape, other_shape, dtype):
    self_tensor = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self_tensor)
    ref_other = to_reference(other_tensor)
    ref_out = torch.ops.aten.greater_equal_(ref_self, ref_other)

    act_self = self_tensor.clone()
    act_other = other_tensor.clone()
    with flag_gems.use_gems():
        act_out = gems_greater_equal__tensor(act_self, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.greater_equal_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [-0.5, 0.0, 1.25])
def test_greater_equal__scalar(shape, dtype, scalar):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self_tensor)
    ref_out = torch.ops.aten.greater_equal_(ref_self, scalar)

    act_self = self_tensor.clone()
    with flag_gems.use_gems():
        act_out = gems_greater_equal__scalar(act_self, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.greater_equal_
def test_perf_aten_greater_equal_():
    # Define input generation logic matching the operator arguments
    def greater_equal__input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=greater_equal__input_fn,
        op_name="greater_equal_",
        torch_op=torch.ops.aten.greater_equal_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
