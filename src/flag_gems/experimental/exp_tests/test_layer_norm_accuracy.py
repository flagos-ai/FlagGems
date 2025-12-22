import os
import sys

import pytest
import torch

import flag_gems

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
except ImportError:
    # Fallback values when running outside pytest
    FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)

    def to_reference(x, upcast=False):
        # Simple fallback - move to CPU
        return x.to("cpu")


# Test shapes relevant for layer normalization
LAYER_NORM_SHAPES = [
    (16, 512),  # Typical transformer hidden size
    (32, 1024),  # Larger hidden size
    (4, 128, 1024),  # Larger batch and hidden
    (1, 2048),  # Single sample, large hidden
]

NORMALIZED_SHAPES = [
    (512,),  # Normalize last dimension
    (1024,),  # Large models
    (64, 768),  # Normalize last 2 dimensions
    (2048,),
]

SHAPE_PAIRS = [
    ((16, 512), (512,)),
    ((32, 1024), (1024,)),
    ((4, 128, 1024), (1024,)),
    ((1, 2048), (2048,)),
]


@pytest.mark.layer_norm
@pytest.mark.parametrize("input_shape,normalized_shape", SHAPE_PAIRS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("eps", [1e-6])
def test_accuracy_layer_norm(input_shape, normalized_shape, dtype, eps):
    # Create input tensor
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    # Test with weight and bias
    weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    # Reference computation
    ref_out = torch.nn.functional.layer_norm(
        ref_inp, normalized_shape, weight=ref_weight, bias=ref_bias, eps=eps
    )

    # FlagGems computation
    with flag_gems.use_gems():
        res_out = flag_gems.experimental.layer_norm(
            inp, normalized_shape, weight=weight, bias=bias, eps=eps
        )

    res_out = res_out.cpu()
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.layer_norm
@pytest.mark.parametrize("input_shape,normalized_shape", SHAPE_PAIRS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layer_norm_no_weight_bias(input_shape, normalized_shape, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.layer_norm(
        ref_inp, normalized_shape, weight=None, bias=None
    )

    with flag_gems.use_gems():
        res_out = flag_gems.experimental.layer_norm(
            inp, normalized_shape, weight=None, bias=None
        )

    res_out = res_out.cpu()
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
