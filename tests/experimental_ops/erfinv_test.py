# ERFINV operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.erfinv import erfinv as gems_erfinv  # noqa: E402
from flag_gems.experimental_ops.erfinv import (  # noqa: E402
    erfinv_out as gems_erfinv_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
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


@pytest.mark.erfinv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfinv_tensor(shape, dtype):
    # inputs in valid domain [-1, 1]
    base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
    input_tensor = (base * 1.98 - 0.99).to(dtype)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.erfinv(ref_input)

    with flag_gems.use_gems():
        act_out = gems_erfinv(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.erfinv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfinv_out(shape, dtype):
    # inputs in valid domain [-1, 1]
    base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
    input_tensor = (base * 1.98 - 0.99).to(dtype)

    ref_input = to_reference(input_tensor)
    ref_out_buf = torch.empty(shape, dtype=dtype, device=ref_input.device)
    ref_out = torch.ops.aten.erfinv.out(ref_input, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        act_out = gems_erfinv_out(input_tensor, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.erfinv
def test_perf_aten_erfinv():
    # Define input generation logic matching the operator arguments
    def erfinv_input_fn(shape, dtype, device):
        # Generate inputs in valid domain [-1, 1]
        base = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
        input_tensor = (base * 1.98 - 0.99).to(dtype)
        yield input_tensor,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=erfinv_input_fn,
        op_name="erfinv",
        torch_op=torch.ops.aten.erfinv,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
