# SPECIAL_I0E operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.special_i0e import special_i0e as gems_special_i0e
from flag_gems.experimental_ops.special_i0e import (
    special_i0e_out as gems_special_i0e_out,
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


@pytest.mark.special_i0e
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_i0e_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out = torch.ops.aten.special_i0e(ref_x)
    with flag_gems.use_gems():
        act_out = gems_special_i0e(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.special_i0e
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_i0e_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    out_ref = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.special_i0e.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_special_i0e_out(x, out_act)
    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(out_act, out_ref, dtype=dtype)


@pytest.mark.special_i0e
def test_perf_aten_special_i0e():
    # Define input generation logic matching the operator arguments
    def special_i0e_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=special_i0e_input_fn,
        op_name="special_i0e",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
