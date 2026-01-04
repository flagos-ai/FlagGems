# HARDSHRINK operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.hardshrink import hardshrink as gems_hardshrink
from flag_gems.experimental_ops.hardshrink import hardshrink_out as gems_hardshrink_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.0])
def test_hardshrink_tensor(shape, dtype, lambd):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.hardshrink(ref_x, lambd)

    with flag_gems.use_gems():
        act_out = gems_hardshrink(x, lambd)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lambd", [0.0, 0.75, 1.5])
def test_hardshrink_out(shape, dtype, lambd):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    out_ref = torch.empty_like(ref_x)
    out_act = torch.empty_like(x)

    ref_out = torch.ops.aten.hardshrink.out(ref_x, lambd, out=out_ref)

    with flag_gems.use_gems():
        act_out = gems_hardshrink_out(x, lambd, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardshrink
def test_perf_aten_hardshrink():
    # Define input generation logic matching the operator arguments
    def hardshrink_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        lambd = 0.5  # Example lambda value, can be varied if needed
        yield inp, lambd

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=hardshrink_input_fn,
        op_name="hardshrink",
        torch_op=torch.ops.aten.hardshrink,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
