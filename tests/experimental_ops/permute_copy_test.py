# PERMUTE_COPY operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.permute_copy import permute_copy as gems_permute_copy
from flag_gems.experimental_ops.permute_copy import (
    permute_copy_out as gems_permute_copy_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import gems_assert_close


except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.permute_copy
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), [0, 1]),
        ((2, 3), [1, 0]),
        ((128, 256), [0, 1]),
        ((128, 256), [1, 0]),
        ((512, 512), [0, 1]),
        ((512, 512), [1, 0]),
        ((8, 16, 32), [0, 1, 2]),
        ((8, 16, 32), [2, 1, 0]),
        ((8, 16, 32), [1, 2, 0]),
        ((8, 16, 32), [-1, -2, -3]),
        ((64, 64, 64), [2, 0, 1]),
        ((4, 8, 16, 32), [0, 2, 3, 1]),
        ((4, 8, 16, 32), [3, 2, 1, 0]),
        ((4, 8, 16, 32), [0, 1, 2, 3]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_permute_copy_tensor(case, dtype):
    shape, dims = case
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.permute_copy(ref_x, dims)

    with flag_gems.use_gems():
        act_out = gems_permute_copy(x, dims)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.permute_copy
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3), [0, 1]),
        ((2, 3), [1, 0]),
        ((128, 256), [0, 1]),
        ((128, 256), [1, 0]),
        ((512, 512), [0, 1]),
        ((512, 512), [1, 0]),
        ((8, 16, 32), [0, 1, 2]),
        ((8, 16, 32), [2, 1, 0]),
        ((8, 16, 32), [1, 2, 0]),
        ((8, 16, 32), [-1, -2, -3]),
        ((64, 64, 64), [2, 0, 1]),
        ((4, 8, 16, 32), [0, 2, 3, 1]),
        ((4, 8, 16, 32), [3, 2, 1, 0]),
        ((4, 8, 16, 32), [0, 1, 2, 3]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_permute_copy_out(case, dtype):
    shape, dims = case
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    rank = len(shape)
    norm_dims = [d if d >= 0 else d + rank for d in dims]
    out_shape = tuple(shape[d] for d in norm_dims)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.permute_copy.out(ref_x, dims, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_permute_copy_out(x, dims, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


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
