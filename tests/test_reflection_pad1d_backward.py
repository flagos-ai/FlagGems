import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# 2D/3D shapes covering small to medium width; pad1d operates on last dim
REFLECTION_PAD1D_SHAPES = [(2, 3), (1, 5), (4, 10), (1, 8, 16)]
# Asymmetric and symmetric padding combinations
REFLECTION_PAD1D_PADDING = [(1, 1), (0, 2), (2, 1), (1, 2)]


@pytest.mark.reflection_pad1d_backward
@pytest.mark.parametrize("shape", REFLECTION_PAD1D_SHAPES)
@pytest.mark.parametrize("padding", REFLECTION_PAD1D_PADDING)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad1d_backward(shape, padding, dtype):
    if padding[0] >= shape[-1] or padding[1] >= shape[-1]:
        pytest.skip("padding values must be less than input width")
    if shape[-1] < 2:
        pytest.skip("input width must be at least 2")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    padded_out = torch.ops.aten.reflection_pad1d(inp, padding)
    grad_output = torch.ones_like(padded_out)
    ref_grad = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.reflection_pad1d_backward(ref_grad, ref_inp, padding)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad1d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)
