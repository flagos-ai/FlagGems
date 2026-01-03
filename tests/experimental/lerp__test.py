# LERP_ operator test

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
from flag_gems.experimental_ops.lerp_ import lerp__Scalar, lerp__Tensor

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark

@pytest.mark.lerp_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight", [0.0, 0.5, -0.3, 1.0])
def test_lerp__scalar(shape, dtype, weight):
    self_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_self = self_base.clone()
    ref_end = end_base.clone()
    ref_out = torch.ops.aten.lerp_(ref_self, ref_end, weight)

    act_self = self_base.clone()
    act_end = end_base.clone()
    with flag_gems.use_gems():
        act_out = lerp__Scalar(act_self, act_end, weight)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.lerp_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lerp__tensor(shape, dtype):
    self_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight_base = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_self = self_base.clone()
    ref_end = end_base.clone()
    ref_weight = weight_base.clone()
    ref_out = torch.ops.aten.lerp_(ref_self, ref_end, ref_weight)

    act_self = self_base.clone()
    act_end = end_base.clone()
    act_weight = weight_base.clone()
    with flag_gems.use_gems():
        act_out = lerp__Tensor(act_self, act_end, act_weight)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.lerp_
def test_perf_aten_lerp_():
    # Define input generation logic matching the operator arguments
    def lerp__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2, weight

    # Initialize benchmark - using lerp__Tensor for performance test
    def lerp__Tensor_wrapper(self, end, weight):
        return lerp__Tensor(self, end, weight)
    
    bench = GenericBenchmark(
        input_fn=lerp__input_fn,
        op_name="lerp_",
        torch_op=torch.ops.aten.lerp_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )
    
    # Replace the op with lerp__Tensor
    bench.op = lerp__Tensor_wrapper

    return bench.run()