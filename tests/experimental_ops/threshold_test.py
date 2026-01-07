# THRESHOLD operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.threshold import (  # noqa: E402
    threshold as gems_threshold,
)
from flag_gems.experimental_ops.threshold import (  # noqa: E402
    threshold_out as gems_threshold_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.threshold
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("threshold", [0.0, -0.5, 0.5, 1.0])
@pytest.mark.parametrize("value", [0.0, 0.1, -1.0, 2.0])
def test_threshold_tensor(shape, dtype, threshold, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out = torch.ops.aten.threshold(ref_x, threshold, value)
    with flag_gems.use_gems():
        act_out = gems_threshold(x, threshold, value)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.threshold
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("threshold", [0.0, -0.5, 0.5, 1.0])
@pytest.mark.parametrize("value", [0.0, 0.1, -1.0, 2.0])
def test_threshold_out(shape, dtype, threshold, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.threshold.out(ref_x, threshold, value, out=ref_out_buf)
    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_threshold_out(x, threshold, value, act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.threshold
def test_perf_aten_threshold():
    # Define input generation logic matching the operator arguments
    def threshold_input_fn(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        threshold = 0.5  # Example threshold
        value = 1.0  # Example value
        yield x, threshold, value

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=threshold_input_fn,
        op_name="threshold",
        torch_op=torch.ops.aten.threshold,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
