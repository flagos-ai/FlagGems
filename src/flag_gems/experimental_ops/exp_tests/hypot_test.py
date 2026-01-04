# HYPOT operator test

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
from flag_gems.experimental_ops.hypot import hypot as gems_hypot
from flag_gems.experimental_ops.hypot import hypot_out as gems_hypot_out

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.hypot
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (513, 257)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hypot_tensor(shape, dtype):
    self = torch.randn(shape, device=flag_gems.device, dtype=torch.float32) * 3.0
    other = torch.randn(shape, device=flag_gems.device, dtype=torch.float32) * 3.0
    self = self.to(dtype)
    other = other.to(dtype)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out = torch.ops.aten.hypot(ref_self, ref_other)

    with flag_gems.use_gems():
        act_out = gems_hypot(self, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hypot
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (513, 257)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hypot_out(shape, dtype):
    self = torch.randn(shape, device=flag_gems.device, dtype=torch.float32) * 3.0
    other = torch.randn(shape, device=flag_gems.device, dtype=torch.float32) * 3.0
    self = self.to(dtype)
    other = other.to(dtype)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out_buf = torch.empty(shape, device=flag_gems.device, dtype=dtype)
    ref_out = torch.ops.aten.hypot.out(ref_self, ref_other, out=ref_out_buf)

    act_out_buf = torch.empty(shape, device=flag_gems.device, dtype=dtype)
    with flag_gems.use_gems():
        act_out = gems_hypot_out(self, other, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hypot
def test_perf_aten_hypot():
    # Define input generation logic matching the operator arguments
    def hypot_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=hypot_input_fn,
        op_name="hypot",
        torch_op=torch.ops.aten.hypot,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
