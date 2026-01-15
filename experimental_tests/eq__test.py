# EQ_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.eq_ import eq__Scalar as gems_eq__Scalar
from flag_gems.experimental_ops.eq_ import eq__Tensor as gems_eq__Tensor

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


@pytest.mark.eq_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [-0.5, 0.0, 1.0])
def test_eq__scalar(shape, dtype, scalar):
    x_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x_base)
    act_x = x_base.clone()

    ref_out = torch.ops.aten.eq_(ref_x, scalar)

    with flag_gems.use_gems():
        act_out = gems_eq__Scalar(act_x, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eq_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other_mode", ["same", "row", "col", "scalar0d"])
def test_eq__tensor(shape, dtype, other_mode):
    x_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if other_mode == "same":
        other_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif other_mode == "row":
        other_base = torch.randn((1, shape[1]), dtype=dtype, device=flag_gems.device)
    elif other_mode == "col":
        other_base = torch.randn((shape[0], 1), dtype=dtype, device=flag_gems.device)
    elif other_mode == "scalar0d":
        other_base = torch.randn((), dtype=dtype, device=flag_gems.device)
    else:
        raise ValueError("invalid other_mode")

    ref_x = to_reference(x_base)
    act_x = x_base.clone()
    ref_other = to_reference(other_base)
    act_other = other_base.clone()

    ref_out = torch.ops.aten.eq_(ref_x, ref_other)

    with flag_gems.use_gems():
        # Use Scalar variant for 0-d tensors, Tensor variant for multi-element tensors
        if other_mode == "scalar0d":
            act_out = gems_eq__Scalar(act_x, act_other)
        else:
            act_out = gems_eq__Tensor(act_x, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.eq_
def test_perf_aten_eq_():
    # Define input generation logic matching the operator arguments
    def eq__input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=eq__input_fn,
        op_name="eq_",
        torch_op=torch.ops.aten.eq_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
