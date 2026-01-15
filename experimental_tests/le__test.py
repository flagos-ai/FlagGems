# LE_ operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.le_ import le__Scalar, le__Tensor  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


@pytest.mark.le_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [-0.5, 0.0, 1.25])
def test_le__scalar(shape, dtype, scalar):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.le_(ref_input, scalar)
    with flag_gems.use_gems():
        act_out = le__Scalar(act_input, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.le_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_le__tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_other = to_reference(other_tensor)
    act_input = input_tensor.clone()
    act_other = other_tensor.clone()

    ref_out = torch.ops.aten.le_(ref_input, ref_other)
    with flag_gems.use_gems():
        act_out = le__Tensor(act_input, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.le_
def test_perf_aten_le_():
    # Define input generation logic matching the operator arguments
    def le__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark - using le__Tensor for performance test
    def le__Tensor_wrapper(self, other):
        return le__Tensor(self, other)

    bench = GenericBenchmark(
        input_fn=le__input_fn,
        op_name="le_",
        torch_op=torch.ops.aten.le_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    # Replace the op with le__Tensor
    bench.op = le__Tensor_wrapper

    return bench.run()
