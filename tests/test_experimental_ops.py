import pytest
import torch
import flag_gems
from tests.accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    to_reference,
)  
  
# Test shapes for layer_norm (normalized_shape should match last dimensions)
LAYER_NORM_SHAPES = [
    ((16, 32), (32,)),      # 2D with last dim normalized
    ((8, 16, 32), (16, 32)),  # 3D with last 2 dims normalized
    ((4, 8, 16, 32), (16, 32)),  # 4D with last 2 dims normalized
]
  
@pytest.mark.layer_norm
@pytest.mark.parametrize("shape, normalized_shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_accuracy_layer_norm(shape, normalized_shape, dtype, eps):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
      
    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)
      
    ref_out = torch.layer_norm(ref_inp, normalized_shape, weight=ref_weight, bias=ref_bias, eps=eps)
    with flag_gems.use_gems():
        res_out = flag_gems.experimental.generated_ops.layer_norm(inp, normalized_shape, weight=weight, bias=bias, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)

@pytest.mark.layer_norm
@pytest.mark.parametrize("shape, normalized_shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layer_norm_no_weight_bias(shape, normalized_shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
      
    ref_out = torch.layer_norm(ref_inp, normalized_shape)
    with flag_gems.use_gems():
        res_out = flag_gems.experimental.generated_ops.layer_norm(inp, normalized_shape)

    gems_assert_close(res_out, ref_out, dtype)