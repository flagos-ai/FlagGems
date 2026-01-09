# XLOGY operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.xlogy import (
    xlogy_OutScalar_Other as gems_xlogy_OutScalar_Other,
)
from flag_gems.experimental_ops.xlogy import (
    xlogy_OutScalar_Self as gems_xlogy_OutScalar_Self,
)
from flag_gems.experimental_ops.xlogy import xlogy_OutTensor as gems_xlogy_OutTensor
from flag_gems.experimental_ops.xlogy import (
    xlogy_Scalar_Other as gems_xlogy_Scalar_Other,
)
from flag_gems.experimental_ops.xlogy import xlogy_Scalar_Self as gems_xlogy_Scalar_Self
from flag_gems.experimental_ops.xlogy import xlogy_Tensor as gems_xlogy_Tensor

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


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_xlogy_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.2

    ref_self = to_reference(self)
    ref_other = to_reference(other)
    ref_out = torch.ops.aten.xlogy(ref_self, ref_other)

    with flag_gems.use_gems():
        act_out = gems_xlogy_Tensor(self, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.5, 1.5, 3.0])
def test_xlogy_scalar_other(shape, dtype, scalar):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_out = torch.ops.aten.xlogy(ref_self, scalar)

    with flag_gems.use_gems():
        act_out = gems_xlogy_Scalar_Other(self, scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.0, 1.0, 2.0])
def test_xlogy_scalar_self(shape, dtype, scalar):
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.2

    ref_other = to_reference(other)
    ref_out = torch.ops.aten.xlogy(scalar, ref_other)

    with flag_gems.use_gems():
        act_out = gems_xlogy_Scalar_Self(scalar, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_xlogy_out_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.2

    ref_self = to_reference(self)
    ref_other = to_reference(other)
    ref_out = torch.empty_like(ref_self)
    torch.ops.aten.xlogy(ref_self, ref_other, out=ref_out)

    act_out = torch.empty_like(self)
    with flag_gems.use_gems():
        gems_xlogy_OutTensor(self, other, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.0, 1.0, 2.0])
def test_xlogy_out_scalar_self(shape, dtype, scalar):
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.2

    ref_other = to_reference(other)
    ref_out = torch.empty_like(ref_other)
    torch.ops.aten.xlogy(scalar, ref_other, out=ref_out)

    act_out = torch.empty_like(other)
    with flag_gems.use_gems():
        gems_xlogy_OutScalar_Self(scalar, other, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.5, 1.5, 3.0])
def test_xlogy_out_scalar_other(shape, dtype, scalar):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_out = torch.empty_like(ref_self)
    torch.ops.aten.xlogy(ref_self, scalar, out=ref_out)

    act_out = torch.empty_like(self)
    with flag_gems.use_gems():
        gems_xlogy_OutScalar_Other(self, scalar, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy
def test_perf_aten_xlogy():
    # Define input generation logic matching the operator arguments
    def xlogy_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = (
            torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.2
        )  # Ensure no zeros in inp2
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=xlogy_input_fn,
        op_name="xlogy",
        torch_op=torch.ops.aten.xlogy,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
