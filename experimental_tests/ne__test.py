# NE_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.ne_ import ne__Scalar as gems_ne__Scalar
from flag_gems.experimental_ops.ne_ import ne__Tensor as gems_ne__Tensor

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


@pytest.mark.ne_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other", [0.0, 1.0, -0.5])
def test_ne__scalar(shape, dtype, other):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)

    ref_out = torch.ops.aten.ne_(ref_input, other)

    with flag_gems.use_gems():
        act_out = gems_ne__Scalar(input_tensor, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.ne_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other_mode", ["same", "row", "col", "scalar0d"])
def test_ne__tensor(shape, dtype, other_mode):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if other_mode == "same":
        other_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif other_mode == "row":
        other_tensor = torch.randn((1, shape[1]), dtype=dtype, device=flag_gems.device)
    elif other_mode == "col":
        other_tensor = torch.randn((shape[0], 1), dtype=dtype, device=flag_gems.device)
    elif other_mode == "scalar0d":
        other_tensor = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_other = to_reference(other_tensor)

    ref_out = torch.ops.aten.ne_(ref_input, ref_other)

    with flag_gems.use_gems():
        # Use Scalar variant for 0-d tensors, Tensor variant for multi-element tensors
        if other_mode == "scalar0d":
            act_out = gems_ne__Scalar(input_tensor, other_tensor)
        else:
            act_out = gems_ne__Tensor(input_tensor, other_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.ne_
def test_perf_aten_ne_():
    # Define input generation logic matching the operator arguments
    def ne__input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=ne__input_fn,
        op_name="ne_",
        torch_op=torch.ops.aten.ne_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
