# SQUARE operator test

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
import triton

import flag_gems
from flag_gems.experimental_ops.square import square as gems_square, square_out as gems_square_out

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark

@pytest.mark.square
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_square_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = x.clone()
    ref_out = torch.ops.aten.square(ref_x)

    with flag_gems.use_gems():
        act_out = gems_square(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.square
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_layout", ["contiguous", "noncontiguous"])
def test_square_out(shape, dtype, out_layout):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if out_layout == "contiguous":
        ref_out_buf = torch.empty_like(x)
        act_out_buf = torch.empty_like(x)
    else:
        ref_base = torch.empty((shape[0], shape[1] * 2), dtype=dtype, device=flag_gems.device)
        act_base = torch.empty((shape[0], shape[1] * 2), dtype=dtype, device=flag_gems.device)
        ref_out_buf = ref_base[:, ::2]
        act_out_buf = act_base[:, ::2]

    ref_res = torch.ops.aten.square.out(x.clone(), out=ref_out_buf)

    with flag_gems.use_gems():
        act_res = gems_square_out(x, act_out_buf)

    gems_assert_close(act_res, ref_res, dtype=dtype)

@pytest.mark.square
def test_perf_aten_square():
    # Define input generation logic matching the operator arguments
    def square_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=square_input_fn,
        op_name="square",
        torch_op=torch.ops.aten.square,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()