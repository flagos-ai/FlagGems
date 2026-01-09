# MULTIPLY operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.multiply import multiply_out as gems_multiply_out
from flag_gems.experimental_ops.multiply import multiply_Scalar as gems_multiply_Scalar
from flag_gems.experimental_ops.multiply import multiply_Tensor as gems_multiply_Tensor

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
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
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.multiply
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), (2, 3)),
        ((2, 3), (1,)),
        ((128, 256), (128, 256)),
        ((128, 256), (256,)),
        ((512, 512), (1, 512)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_multiply_tensor(case, dtype):
    self_shape, other_shape = case
    self = torch.randn(self_shape, device=flag_gems.device, dtype=dtype)
    other = torch.randn(other_shape, device=flag_gems.device, dtype=dtype)

    ref_self = to_reference(self)
    ref_other = to_reference(other)

    ref_out = torch.ops.aten.multiply(ref_self, ref_other)

    with flag_gems.use_gems():
        act_out = gems_multiply_Tensor(self, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.multiply
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.0, 0.5, -1.25, 2.0])
def test_multiply_scalar(shape, dtype, scalar):
    self = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_self = to_reference(self)
    ref_out = torch.ops.aten.multiply(ref_self, scalar)

    with flag_gems.use_gems():
        act_out = gems_multiply_Scalar(self, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.multiply
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), (2, 3)),
        ((2, 3), (1,)),
        ((128, 256), (128, 256)),
        ((128, 256), (256,)),
        ((512, 512), (1, 512)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_multiply_out(case, dtype):
    self_shape, other_shape = case
    self = torch.randn(self_shape, device=flag_gems.device, dtype=dtype)
    other = torch.randn(other_shape, device=flag_gems.device, dtype=dtype)

    b_self, b_other = torch.broadcast_tensors(self, other)
    out_shape = b_self.shape

    ref_self = to_reference(self)
    ref_other = to_reference(other)
    ref_out = torch.empty(out_shape, device=flag_gems.device, dtype=dtype)
    torch.ops.aten.multiply.out(ref_self, ref_other, out=ref_out)

    act_out = torch.empty_like(ref_out)
    with flag_gems.use_gems():
        gems_multiply_out(self, other, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.multiply
def test_perf_aten_multiply():
    # Define input generation logic matching the operator arguments
    def multiply_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=multiply_input_fn,
        op_name="multiply",
        torch_op=torch.ops.aten.multiply,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
