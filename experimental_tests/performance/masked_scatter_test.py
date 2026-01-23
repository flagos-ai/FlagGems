# MASKED_SCATTER operator test

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


@pytest.mark.masked_scatter
def test_perf_aten_masked_scatter():
    # Define input generation logic matching the operator arguments
    def masked_scatter_input_fn(shape, dtype, device):
        self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        mask = torch.rand(shape, device=flag_gems.device) > 0.75
        source = torch.randn(self.numel(), dtype=dtype, device=flag_gems.device)
        yield self, mask, source

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=masked_scatter_input_fn,
        op_name="masked_scatter",
        torch_op=torch.ops.aten.masked_scatter,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
