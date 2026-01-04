# PRELU operator test

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
from flag_gems.experimental_ops.prelu import prelu as gems_prelu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.prelu
@pytest.mark.parametrize(
    "shape",
    [(2, 3), (128, 256), (512, 512), (4, 8, 16), (2, 32, 16, 16), (2, 128, 64, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_kind", ["scalar", "per_channel"])
def test_prelu_tensor(shape, dtype, weight_kind):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if weight_kind == "scalar":
        w = torch.randn((), dtype=dtype, device=flag_gems.device)
    else:
        c = shape[1]
        w = torch.randn((c,), dtype=dtype, device=flag_gems.device)

    ref_x = x.clone()
    ref_w = w.clone()

    ref_out = torch.ops.aten.prelu(ref_x, ref_w)

    with flag_gems.use_gems():
        act_out = gems_prelu(x, w)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.prelu
def test_perf_aten_prelu():
    # Define input generation logic matching the operator arguments
    def prelu_input_fn(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if len(shape) == 1:
            w = torch.randn((), dtype=dtype, device=flag_gems.device)  # Scalar weight
        else:
            w = torch.randn(
                (shape[1],), dtype=dtype, device=flag_gems.device
            )  # Per-channel weight
        yield x, w

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=prelu_input_fn,
        op_name="prelu",
        torch_op=torch.ops.aten.prelu,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
