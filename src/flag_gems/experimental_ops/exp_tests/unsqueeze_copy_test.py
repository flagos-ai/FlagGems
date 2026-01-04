# UNSQUEEZE_COPY operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.unsqueeze_copy import (
    unsqueeze_copy as gems_unsqueeze_copy,
)
from flag_gems.experimental_ops.unsqueeze_copy import (
    unsqueeze_copy_out as gems_unsqueeze_copy_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.unsqueeze_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 64), (64, 32, 16), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("where", ["zero", "neg1", "end", "minneg"])
def test_unsqueeze_copy_default(shape, dtype, where):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    n = len(shape)
    if where == "zero":
        dim = 0
    elif where == "neg1":
        dim = -1
    elif where == "end":
        dim = n
    elif where == "minneg":
        dim = -(n + 1)
    else:
        dim = 0

    ref_out = torch.ops.aten.unsqueeze_copy(ref_x, dim)

    with flag_gems.use_gems():
        act_out = gems_unsqueeze_copy(x, dim)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.unsqueeze_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 64), (64, 32, 16), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("where", ["zero", "neg1", "end", "minneg"])
def test_unsqueeze_copy_out(shape, dtype, where):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    n = len(shape)
    if where == "zero":
        dim = 0
    elif where == "neg1":
        dim = -1
    elif where == "end":
        dim = n
    elif where == "minneg":
        dim = -(n + 1)
    else:
        dim = 0

    pos = dim + n + 1 if dim < 0 else dim
    new_shape = shape[:pos] + (1,) + shape[pos:]

    ref_out_buf = torch.empty(new_shape, device=flag_gems.device, dtype=dtype)
    act_out_buf = torch.empty_like(ref_out_buf)

    ref_out = torch.ops.aten.unsqueeze_copy(ref_x, dim, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_unsqueeze_copy_out(x, dim, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.unsqueeze_copy
def test_perf_aten_unsqueeze_copy():
    # Define input generation logic matching the operator arguments
    def unsqueeze_copy_input_fn(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        n = len(shape)
        if n == 0:
            dim = 0
        else:
            dim = 0  # You can modify this to test different dimensions if needed
        yield x, dim

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=unsqueeze_copy_input_fn,
        op_name="unsqueeze_copy",
        torch_op=torch.ops.aten.unsqueeze_copy,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
