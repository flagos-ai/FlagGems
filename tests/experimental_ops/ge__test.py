# GE_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.ge_ import ge__Scalar as gems_ge__Scalar
from flag_gems.experimental_ops.ge_ import ge__Tensor as gems_ge__Tensor

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU

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


@pytest.mark.ge_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ge__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_other = to_reference(other_tensor)

    ref_out = torch.ops.aten.ge_(ref_input, ref_other)

    with flag_gems.use_gems():
        act_out = gems_ge__Tensor(input_tensor, other_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.ge_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other", [-0.5, 0.0, 1.25])
def test_ge__scalar(shape, dtype, other):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.ge_(ref_input, other)

    with flag_gems.use_gems():
        act_out = gems_ge__Scalar(input_tensor, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.ge_
def test_perf_aten_ge_():
    # Define input generation logic matching the operator arguments
    def ge__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=ge__input_fn,
        op_name="ge_",
        torch_op=torch.ops.aten.ge_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
