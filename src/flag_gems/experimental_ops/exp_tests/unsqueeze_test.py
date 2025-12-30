# UNSQUEEZE operator test

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
from flag_gems.experimental_ops.unsqueeze import unsqueeze as gems_unsqueeze

import torch

from benchmark.performance_utils import GenericBenchmark

@pytest.mark.unsqueeze
@pytest.mark.parametrize(
    "shape_dim",
    [
        ((), 0),
        ((), -1),
        ((2, 3), -3),
        ((2, 3), -1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), 2),
        ((128, 256), -3),
        ((128, 256), -1),
        ((128, 256), 0),
        ((128, 256), 2),
        ((512, 512), -3),
        ((512, 512), 0),
        ((512, 512), 2),
        ((64, 32, 16), -4),
        ((64, 32, 16), -1),
        ((64, 32, 16), 0),
        ((64, 32, 16), 2),
        ((64, 32, 16), 3),
        ((4,), -2),
        ((4,), -1),
        ((4,), 0),
        ((4,), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unsqueeze_tensor(shape_dim, dtype):
    shape, dim = shape_dim
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.unsqueeze(ref_input, dim)

    with flag_gems.use_gems():
        act_out = gems_unsqueeze(input_tensor, dim)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.unsqueeze
def test_perf_aten_unsqueeze():
    # Define input generation logic matching the operator arguments
    def unsqueeze_input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        for dim in range(len(shape) + 1):  # Yielding all possible dimensions for unsqueeze
            yield inp, dim

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=unsqueeze_input_fn,
        op_name="unsqueeze",
        torch_op=torch.ops.aten.unsqueeze,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()