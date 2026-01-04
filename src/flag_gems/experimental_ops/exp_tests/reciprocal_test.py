# RECIPROCAL operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
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
from flag_gems.experimental_ops.reciprocal import reciprocal as gems_reciprocal
from flag_gems.experimental_ops.reciprocal import reciprocal_out as gems_reciprocal_out

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reciprocal_tensor(shape, dtype):
    base = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.9 + 0.1
    sign = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(dtype)
    input_tensor = base * sign

    ref_input = input_tensor.clone()
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

    ref_out = torch.empty_like(input_tensor)
    torch.ops.aten.reciprocal.out(input_tensor.clone(), out=ref_out)

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
