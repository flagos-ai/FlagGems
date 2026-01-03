# ADDCMUL_ operator test

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
from flag_gems.experimental_ops.addcmul_ import addcmul_ as gems_addcmul_

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark

@pytest.mark.addcmul_
@pytest.mark.parametrize(
    "self_shape,t1_shape,t2_shape",
    [
        ((2, 3), (2, 3), (2, 3)),
        ((2, 3), (2, 1), (1, 3)),
        ((128, 256), (128, 256), (128, 1)),
        ((128, 256), (1, 256), (128, 256)),
        ((64, 128), (64, 1), (1, 128)),
        ((4, 8, 16), (1, 8, 1), (4, 1, 16)),
        ((512, 512), (512, 512), (512, 512)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [1, 0.5, 2, -1.5])
def test_addcmul__inplace(self_shape, t1_shape, t2_shape, dtype, value):
    self_tensor = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(t1_shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(t2_shape, dtype=dtype, device=flag_gems.device)

    ref_self = self_tensor.clone()
    ref_t1 = t1.clone()
    ref_t2 = t2.clone()
    ref_out = torch.ops.aten.addcmul_(ref_self, ref_t1, ref_t2, value=value)

    act_self = self_tensor.clone()
    act_t1 = t1.clone()
    act_t2 = t2.clone()
    with flag_gems.use_gems():
        act_out = gems_addcmul_(act_self, act_t1, act_t2, value=value)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.addcmul_
def test_perf_aten_addcmul_():
    # Define input generation logic matching the operator arguments
    def addcmul__input_fn(shape, dtype, device):
        self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield self_tensor, t1, t2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=addcmul__input_fn,
        op_name="addcmul_",
        torch_op=torch.ops.aten.addcmul_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()