# SOFTSHRINK operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.softshrink import (  # noqa: E402
    softshrink as gems_softshrink,
)
from flag_gems.experimental_ops.softshrink import (  # noqa: E402
    softshrink_out as gems_softshrink_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
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


@pytest.mark.softshrink
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.25])
def test_softshrink_tensor(shape, dtype, lambd):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.softshrink(ref_x, lambd)

    with flag_gems.use_gems():
        act_out = gems_softshrink(x, lambd)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.softshrink
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.25])
def test_softshrink_out(shape, dtype, lambd):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out_buf = torch.empty_like(ref_x)
    ref_res = torch.ops.aten.softshrink.out(ref_x, lambd, out=ref_out_buf)

    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_res = gems_softshrink_out(x, lambd, act_out_buf)

    gems_assert_close(act_res, ref_res, dtype=dtype)


@pytest.mark.softshrink
def test_perf_aten_softshrink():
    # Define input generation logic matching the operator arguments
    def softshrink_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        lambd = 0.5  # You can choose a fixed lambda for benchmarking
        yield inp, lambd

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=softshrink_input_fn,
        op_name="softshrink",
        torch_op=torch.ops.aten.softshrink,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
