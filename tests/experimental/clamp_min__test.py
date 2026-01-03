# CLAMP_MIN_ operator test

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
from flag_gems.experimental_ops.clamp_min_ import clamp_min_ as gems_clamp_min_

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark

@pytest.mark.clamp_min_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_val", [0.0, -0.5, 1.5])
def test_clamp_min__scalar(shape, dtype, min_val):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = inp.clone()
    act_inp = inp.clone()

    ref_out = torch.ops.aten.clamp_min_(ref_inp, min_val)
    with flag_gems.use_gems():
        act_out = gems_clamp_min_(act_inp, min_val)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.clamp_min_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_clamp_min__tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    min_t = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = inp.clone()
    ref_min = min_t.clone()
    act_inp = inp.clone()
    act_min = min_t.clone()

    ref_out = torch.ops.aten.clamp_min_.Tensor(ref_inp, ref_min)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.clamp_min_.Tensor(act_inp, act_min)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.clamp_min_
def test_perf_aten_clamp_min_():
    # Define input generation logic matching the operator arguments
    def clamp_min__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        min_val = torch.tensor(0.0, dtype=dtype, device=flag_gems.device)  # Example min_val
        yield inp, min_val

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=clamp_min__input_fn,
        op_name="clamp_min_",
        torch_op=torch.ops.aten.clamp_min_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()