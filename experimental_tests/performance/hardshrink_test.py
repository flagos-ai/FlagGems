# HARDSHRINK operator test

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
    from tests.accuracy_utils import TO_CPU, gems_assert_close
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


@pytest.mark.hardshrink
def test_perf_aten_hardshrink():
    # Define input generation logic matching the operator arguments
    def hardshrink_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        lambd = 0.5  # Example lambda value, can be varied if needed
        yield inp, lambd

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=hardshrink_input_fn,
        op_name="hardshrink",
        torch_op=torch.ops.aten.hardshrink,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
