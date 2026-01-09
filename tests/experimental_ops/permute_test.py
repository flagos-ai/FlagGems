# PERMUTE operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.permute import permute as gems_permute

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import gems_assert_close, TO_CPU


except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.permute
@pytest.mark.parametrize(
    "shape",
    [
        (2,),
        (2, 3),
        (128, 256),
        (512, 512),
        (2, 3, 4),
        (32, 64, 16),
        (64, 32, 128),
        (2, 3, 4, 5),
        (8, 16, 32, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_permute_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    rank = len(shape)
    dims_map = {
        1: [[0]],
        2: [[0, 1], [1, 0]],
        3: [[0, 1, 2], [0, 2, 1], [2, 0, 1]],
        4: [[0, 1, 2, 3], [0, 2, 3, 1], [3, 1, 0, 2]],
    }
    for dims in dims_map[rank]:
        ref_out = torch.ops.aten.permute(ref_x, dims)
        with flag_gems.use_gems():
            act_out = gems_permute(x, dims)
        gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.permute
@pytest.mark.parametrize("shape", [(1024,), (2, 3, 4), (8, 16, 32, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_permute_negative_dims(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    rank = len(shape)
    neg_dims_map = {
        1: [[-1]],
        3: [[-1, -2, -3], [-3, -1, -2]],
        4: [[-1, -2, -3, -4], [-4, -2, -1, -3]],
    }
    for dims in neg_dims_map[rank]:
        ref_out = torch.ops.aten.permute(ref_x, dims)
        with flag_gems.use_gems():
            act_out = gems_permute(x, dims)
        gems_assert_close(act_out, ref_out, dtype=dtype)


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
