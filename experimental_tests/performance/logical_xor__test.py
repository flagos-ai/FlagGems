# LOGICAL_XOR_ operator test

import os
import sys

import pytest
import torch

import flag_gems

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

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


@pytest.mark.logical_xor_
def test_perf_aten_logical_xor_():
    # Define input generation logic matching the operator arguments
    def logical_xor__input_fn(shape, dtype, device):
        # Generate and yield inputs as required by the operator
        inp1 = (torch.rand(shape, device=flag_gems.device) > 0.5).to(dtype)
        inp2 = (torch.rand(shape, device=flag_gems.device) > 0.5).to(dtype)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=logical_xor__input_fn,
        op_name="logical_xor_",
        torch_op=torch.ops.aten.logical_xor_,
        dtypes=[torch.bool],  # Use torch.bool as per the correctness test
    )

    return bench.run()
