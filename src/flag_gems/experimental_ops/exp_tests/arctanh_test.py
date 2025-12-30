# ARCTANH operator test

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
import triton

import flag_gems
from flag_gems.experimental_ops.arctanh import arctanh as gems_arctanh, arctanh_out as gems_arctanh_out

import torch

from benchmark.performance_utils import GenericBenchmark

@pytest.mark.arctanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arctanh_tensor(shape, dtype):
    x = (torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 1.8 - 0.9).to(dtype)
    ref_x = x.clone()

    ref_out = torch.ops.aten.arctanh(ref_x)

    with flag_gems.use_gems():
        act_out = gems_arctanh(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.arctanh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arctanh_out(shape, dtype):
    x = (torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 1.8 - 0.9).to(dtype)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out_buf = torch.empty_like(ref_x)
    torch.ops.aten.arctanh.out(ref_x, out=ref_out_buf)

    act_out_buf = torch.empty_like(act_x)
    with flag_gems.use_gems():
        gems_arctanh_out(act_x, act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)

@pytest.mark.arctanh
def test_perf_aten_arctanh():
    # Define input generation logic matching the operator arguments
    def arctanh_input_fn(shape, dtype, device):
        inp = (torch.rand(shape, device=flag_gems.device, dtype=torch.float32) * 1.8 - 0.9).to(dtype)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=arctanh_input_fn,
        op_name="arctanh",
        torch_op=torch.ops.aten.arctanh,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()