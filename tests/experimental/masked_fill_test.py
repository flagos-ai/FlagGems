# MASKED_FILL operator test

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
from flag_gems.experimental_ops.masked_fill import masked_fill_Scalar, masked_fill_Tensor, masked_fill_Scalar_out, masked_fill_Tensor_out

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark

@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, -1.25, 2.5])
def test_masked_fill_scalar(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.5)
    ref_x = x.clone()
    ref_mask = mask.clone()
    ref_out = torch.ops.aten.masked_fill.Scalar(ref_x, ref_mask, value)
    with flag_gems.use_gems():
        act_out = masked_fill_Scalar(x, mask, value)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [1.0, -3.0, 0.75])
def test_masked_fill_tensor(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.5)
    val = torch.tensor(value, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_mask = mask.clone()
    ref_val = val.clone()
    ref_out = torch.ops.aten.masked_fill.Tensor(ref_x, ref_mask, ref_val)
    with flag_gems.use_gems():
        act_out = masked_fill_Tensor(x, mask, val)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, -2.0, 3.5])
def test_masked_fill_scalar_out(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.5)
    ref_x = x.clone()
    ref_mask = mask.clone()
    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.masked_fill.Scalar_out(ref_x, ref_mask, value, out=ref_out_buf)
    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = masked_fill_Scalar_out(x, mask, value, out=act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [1.5, -0.5, 4.0])
def test_masked_fill_tensor_out(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.5)
    val = torch.tensor(value, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_mask = mask.clone()
    ref_val = val.clone()
    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.masked_fill.Tensor_out(ref_x, ref_mask, ref_val, out=ref_out_buf)
    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = masked_fill_Tensor_out(x, mask, val, out=act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.masked_fill
def test_perf_aten_masked_fill():
    # Define input generation logic matching the operator arguments
    def masked_fill_input_fn(shape, dtype, device):
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        mask = (torch.rand(shape, device=flag_gems.device) > 0.5)
        value = torch.tensor(1.0, dtype=dtype, device=flag_gems.device)  # Example scalar value
        yield x, mask, value

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=masked_fill_input_fn,
        op_name="masked_fill",
        torch_op=torch.ops.aten.masked_fill,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()