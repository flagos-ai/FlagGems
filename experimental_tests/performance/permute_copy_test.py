# PERMUTE_COPY operator test

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


@pytest.mark.permute_copy
def test_perf_aten_permute_copy():
    # Define input generation logic matching the operator arguments
    def permute_copy_input_fn(shape, dtype, device):
        # Generate input tensor
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Generate a random permutation of dimensions
        dims = list(range(len(shape)))
        yield inp, dims

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=permute_copy_input_fn,
        op_name="permute_copy",
        torch_op=torch.ops.aten.permute_copy,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
