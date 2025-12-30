# LEAKY_RELU_ operator test

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
from flag_gems.experimental_ops.leaky_relu_ import leaky_relu_ as gems_leaky_relu_

import torch

from benchmark.performance_utils import GenericBenchmark

@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_leaky_relu__tensor_default(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out = torch.ops.aten.leaky_relu_(ref_x)

    with flag_gems.use_gems():
        act_out = gems_leaky_relu_(act_x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("negative_slope", [0.0, 0.01, 0.2, 1.5])
def test_leaky_relu__tensor_with_slope(shape, dtype, negative_slope):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    ref_out = torch.ops.aten.leaky_relu_(ref_x, negative_slope)

    with flag_gems.use_gems():
        act_out = gems_leaky_relu_(act_x, negative_slope)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.leaky_relu_
def test_perf_aten_leaky_relu_():
    # Define input generation logic matching the operator arguments
    def leaky_relu__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=leaky_relu__input_fn,
        op_name="leaky_relu_",
        torch_op=torch.ops.aten.leaky_relu_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()