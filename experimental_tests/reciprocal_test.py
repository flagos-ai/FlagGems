# RECIPROCAL operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.reciprocal import (  # noqa: E402
    reciprocal as gems_reciprocal,
)
from flag_gems.experimental_ops.reciprocal import (  # noqa: E402
    reciprocal_out as gems_reciprocal_out,
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


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reciprocal_tensor(shape, dtype):
    base = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.9 + 0.1
    sign = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(dtype)
    input_tensor = base * sign

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.reciprocal(ref_input)

    with flag_gems.use_gems():
        act_out = gems_reciprocal(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reciprocal_out(shape, dtype):
    base = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.9 + 0.1
    sign = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(dtype)
    input_tensor = base * sign

    ref_input = to_reference(input_tensor)
    ref_out = torch.empty_like(ref_input)
    torch.ops.aten.reciprocal.out(ref_input, out=ref_out)

    act_out = torch.empty_like(input_tensor)
    with flag_gems.use_gems():
        gems_reciprocal_out(input_tensor, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reciprocal
def test_perf_aten_reciprocal():
    # Define input generation logic matching the operator arguments
    def reciprocal_input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        base = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.9 + 0.1
        sign = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(dtype)
        input_tensor = base * sign
        yield input_tensor,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=reciprocal_input_fn,
        op_name="reciprocal",
        torch_op=torch.ops.aten.reciprocal,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
