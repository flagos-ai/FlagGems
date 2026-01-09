# ABS operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.abs import abs as gems_abs
from flag_gems.experimental_ops.abs import abs_out as gems_abs_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


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


@pytest.mark.abs
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_abs_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.abs(ref_x)
    with flag_gems.use_gems():
        act_out = gems_abs(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.abs
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_abs_out_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out_tensor = torch.empty(shape, dtype=dtype, device=ref_x.device)
    act_out_tensor = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.abs.out(ref_x, out=ref_out_tensor)
    with flag_gems.use_gems():
        act_out = gems_abs_out(x, act_out_tensor)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.abs
def test_perf_aten_abs():
    # Define input generation logic matching the operator arguments
    def abs_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=abs_input_fn,
        op_name="abs",
        torch_op=torch.ops.aten.abs,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
