# XLOGY_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.xlogy_ import (
    xlogy__Scalar_Other as gems_xlogy__Scalar_Other,
)
from flag_gems.experimental_ops.xlogy_ import xlogy__Tensor as gems_xlogy__Tensor

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


@pytest.mark.xlogy_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_xlogy__tensor(shape, dtype):
    self_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other_base = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1

    ref_self = to_reference(self_base)
    ref_other = to_reference(other_base)
    ref_out = torch.ops.aten.xlogy_(ref_self, ref_other)

    act_self = self_base.clone()
    act_other = other_base.clone()
    with flag_gems.use_gems():
        act_out = gems_xlogy__Tensor(act_self, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other_scalar", [0.3, 1.0, 2.5])
def test_xlogy__scalar(shape, dtype, other_scalar):
    self_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self_base)
    ref_out = torch.ops.aten.xlogy_(ref_self, other_scalar)

    act_self = self_base.clone()
    with flag_gems.use_gems():
        act_out = gems_xlogy__Scalar_Other(act_self, other_scalar)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.xlogy_
def test_perf_aten_xlogy_():
    # Define input generation logic matching the operator arguments
    def xlogy__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = (
            torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
        )  # Ensure no zeros in inp2
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=xlogy__input_fn,
        op_name="xlogy_",
        torch_op=torch.ops.aten.xlogy_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
