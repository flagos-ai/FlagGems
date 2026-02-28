# _UNSAFE_VIEW operator test
import os
import sys

import pytest
import torch

import flag_gems

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


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


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


@pytest.mark.unsafe_view
def test_perf_aten__unsafe_view():
    # Define input generation logic matching the operator arguments
    def _unsafe_view_input_fn(shape, dtype, device):
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Generate the output size based on the input shape
        output_size = (
            torch.prod(torch.tensor(shape)).item(),
        )  # Flatten the input shape
        yield input_tensor, output_size

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=_unsafe_view_input_fn,
        op_name="_unsafe_view",
        torch_op=torch.ops.aten._unsafe_view,
        dtypes=[torch.float32, torch.float16],
    )

    return bench.run()
