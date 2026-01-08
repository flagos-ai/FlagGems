# DEG2RAD operator test

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


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.deg2rad import deg2rad as gems_deg2rad  # noqa: E402
from flag_gems.experimental_ops.deg2rad import (  # noqa: E402
    deg2rad_out as gems_deg2rad_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deg2rad_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = x.clone()
    ref_out = torch.ops.aten.deg2rad(ref_x)

    with flag_gems.use_gems():
        act_out = gems_deg2rad(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deg2rad_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = x.clone()
    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.deg2rad.out(ref_x, out=ref_out)

    act_out = torch.empty_like(x)
    with flag_gems.use_gems():
        gems_deg2rad_out(x, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.deg2rad
def test_perf_aten_deg2rad():
    # Define input generation logic matching the operator arguments
    def deg2rad_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=deg2rad_input_fn,
        op_name="deg2rad",
        torch_op=torch.ops.aten.deg2rad,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
