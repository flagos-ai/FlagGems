# PERMUTE operator test

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


@pytest.mark.permute
def test_perf_aten_permute():
    # Define input generation logic matching the operator arguments
    def permute_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Generate all possible permutations of the dimensions
        rank = len(shape)
        dims_map = {
            1: [[0]],
            2: [[0, 1], [1, 0]],
            3: [[0, 1, 2], [0, 2, 1], [2, 0, 1]],
            4: [[0, 1, 2, 3], [0, 2, 3, 1], [3, 1, 0, 2]],
        }
        for dims in dims_map[rank]:
            yield inp, dims

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=permute_input_fn,
        op_name="permute",
        torch_op=torch.ops.aten.permute,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
