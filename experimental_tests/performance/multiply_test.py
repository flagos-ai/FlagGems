# MULTIPLY operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import TO_CPU, gems_assert_close


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp):
    """Move to CPU when TO_CPU is set, keep dtype/device otherwise."""
    if inp is None:
        return None
    return inp.to("cpu") if TO_CPU else inp.clone()


@pytest.mark.multiply
def test_perf_aten_multiply():
    # Define input generation logic matching the operator arguments
    def multiply_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=multiply_input_fn,
        op_name="multiply",
        torch_op=torch.ops.aten.multiply,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
