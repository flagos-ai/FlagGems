# THRESHOLD operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


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
