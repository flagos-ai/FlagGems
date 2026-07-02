import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Expand sizes: (input shape, target size)
# Tests common expand scenarios: (2,1)->(2,3), (1,3)->(5,3), etc.
EXPAND_SIZES = [
    ((2, 1), (2, 3)),
    ((1, 3), (5, 3)),
    ((2, 1, 3), (2, 4, 3)),
    ((1, 1, 1), (2, 3, 4)),
    ((1,), (2, 3, 4)),
    ((1, 2), (1, 2)),  # No change
    ((2, 1), (2, -1)),  # Keep dimension
]


@pytest.mark.expand
@pytest.mark.parametrize("shape_expand_sizes", EXPAND_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand(shape_expand_sizes, dtype):
    input_shape, expand_size = shape_expand_sizes
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.expand(ref_inp, expand_size)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.expand(inp, expand_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.expand_
@pytest.mark.parametrize("shape_expand_sizes", EXPAND_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_(shape_expand_sizes, dtype):
    input_shape, expand_size = shape_expand_sizes
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.expand(expand_size)
    with flag_gems.use_gems():
        res_out = inp.expand(expand_size)

    utils.gems_assert_close(res_out, ref_out, dtype)
    # Verify original input data was not modified (expand_ returns a view)
    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.expand
@pytest.mark.parametrize(
    "input_shape,expand_size",
    [
        ((1, 3), (3,)),
        ((3,), (-1, 3)),
        ((3,), (-2, 3)),
        ((2, 3), (2, 4)),
    ],
)
def test_expand_invalid_sizes(input_shape, expand_size):
    inp = torch.randn(input_shape, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.expand(inp, expand_size)


@pytest.mark.expand
def test_expand_zero_size_singleton_stride():
    inp = torch.randn((1,), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.expand(ref_inp, (0,))
    with flag_gems.use_gems():
        res_out = torch.ops.aten.expand(inp, (0,))

    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
