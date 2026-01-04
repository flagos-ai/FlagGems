# LIFT operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.lift import lift as gems_lift
from flag_gems.experimental_ops.lift import lift_out as gems_lift_out

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.lift
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lift_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = x.clone()
    ref_out = torch.ops.aten.lift(ref_x)

    with flag_gems.use_gems():
        act_out = gems_lift(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.lift
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lift_out_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.lift.out(ref_x, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out_buf = torch.empty_like(x)
        act_out = gems_lift_out(x, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.lift
def test_perf_aten_lift():
    # Define input generation logic matching the operator arguments
    def lift_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=lift_input_fn,
        op_name="lift",
        torch_op=torch.ops.aten.lift,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
