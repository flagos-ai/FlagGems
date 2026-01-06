# HEAVISIDE operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.heaviside import heaviside as gems_heaviside
from flag_gems.experimental_ops.heaviside import heaviside_out as gems_heaviside_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_heaviside_tensor(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = self_tensor.clone()
    ref_values = values_tensor.clone()

    ref_out = torch.ops.aten.heaviside(ref_self, ref_values)

    with flag_gems.use_gems():
        act_out = gems_heaviside(self_tensor, values_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_heaviside_out(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = self_tensor.clone()
    ref_values = values_tensor.clone()
    ref_out_buf = torch.empty_like(ref_self)

    ref_out = torch.ops.aten.heaviside.out(ref_self, ref_values, out=ref_out_buf)

    act_out_buf = torch.empty_like(self_tensor)
    with flag_gems.use_gems():
        act_out = gems_heaviside_out(self_tensor, values_tensor, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.heaviside
def test_perf_aten_heaviside():
    # Define input generation logic matching the operator arguments
    def heaviside_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=heaviside_input_fn,
        op_name="heaviside",
        torch_op=torch.ops.aten.heaviside,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
