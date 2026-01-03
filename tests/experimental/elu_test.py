# ELU operator test

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
from flag_gems.experimental_ops.elu import elu as gems_elu, elu_out as gems_elu_out

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark

@pytest.mark.elu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("params", [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.5, 2.0, 0.5)])
def test_elu_tensor(shape, dtype, params):
    alpha, scale, input_scale = params
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    ref_out = torch.ops.aten.elu(ref_input, alpha, scale, input_scale)

    with flag_gems.use_gems():
        act_out = gems_elu(act_input, alpha, scale, input_scale)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.elu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("params", [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.5, 2.0, 0.5)])
def test_elu_out(shape, dtype, params):
    alpha, scale, input_scale = params
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    ref_outbuf = torch.empty_like(ref_input)
    ref_res = torch.ops.aten.elu.out(ref_input, alpha, scale, input_scale, out=ref_outbuf)

    act_outbuf = torch.empty_like(act_input)
    with flag_gems.use_gems():
        act_res = gems_elu_out(act_input, alpha, scale, input_scale, act_outbuf)

    gems_assert_close(act_outbuf, ref_outbuf, dtype=dtype)

@pytest.mark.elu
def test_perf_aten_elu():
    # Define input generation logic matching the operator arguments
    def elu_input_fn(shape, dtype, device):
        alpha = 1.0  # Example fixed parameter
        scale = 1.0  # Example fixed parameter
        input_scale = 1.0  # Example fixed parameter
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp, alpha, scale, input_scale

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=elu_input_fn,
        op_name="elu",
        torch_op=torch.ops.aten.elu,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()