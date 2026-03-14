# LEAKY_RELU operator test

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
from flag_gems.testing import gems_assert_close


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


@pytest.mark.leaky_relu
def test_perf_aten_leaky_relu():
    # Define input generation logic matching the operator arguments
    def leaky_relu_input_fn(shape, dtype, device):
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield input_tensor,

        # Yield negative_slope as a parameter
        for negative_slope in [0.0, 0.01, 0.2]:
            yield input_tensor, negative_slope

    test_x = torch.randn((1024, 1024), dtype=torch.float16, device="cuda")
    with flag_gems.use_gems():
        actual = torch.nn.functional.leaky_relu(test_x, negative_slope=0.01)
    expected = torch.nn.functional.leaky_relu(test_x, negative_slope=0.01)
    gems_assert_close(actual, expected, torch.float16)
    
    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=leaky_relu_input_fn,
        op_name="leaky_relu",
        torch_op=torch.ops.aten.leaky_relu,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
