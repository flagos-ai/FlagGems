# ARCSINH operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.arcsinh import arcsinh as gems_arcsinh
from flag_gems.experimental_ops.arcsinh import arcsinh_out as gems_arcsinh_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.arcsinh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arcsinh_tensor(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_out = torch.ops.aten.arcsinh(ref_x)

    with flag_gems.use_gems():
        act_out = gems_arcsinh(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.arcsinh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("aliasing", [False, True])
def test_arcsinh_out(shape, dtype, aliasing):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    if aliasing:
        ref_out = torch.ops.aten.arcsinh.out(ref_x, out=ref_x)
    else:
        ref_out_buf = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.arcsinh.out(ref_x, out=ref_out_buf)

    if aliasing:
        with flag_gems.use_gems():
            act_out = gems_arcsinh_out(x, x)
    else:
        out_buf = torch.empty_like(x)
        with flag_gems.use_gems():
            act_out = gems_arcsinh_out(x, out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.arcsinh
def test_perf_aten_arcsinh():
    # Define input generation logic matching the operator arguments
    def arcsinh_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=arcsinh_input_fn,
        op_name="arcsinh",
        torch_op=torch.ops.aten.arcsinh,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
