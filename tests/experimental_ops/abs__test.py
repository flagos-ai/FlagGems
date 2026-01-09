# ABS_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.abs_ import abs_ as gems_abs_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

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


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.abs_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_abs__tensor(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.abs_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_abs_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(act_input, ref_input, dtype=dtype)


@pytest.mark.abs_
def test_perf_aten_abs_():
    # Define input generation logic matching the operator arguments
    def abs__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=abs__input_fn,
        op_name="abs_",
        torch_op=torch.ops.aten.abs_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
