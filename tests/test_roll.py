import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ROLL_SHIFTS_DIMS = [
    (1, 0),
    (-1, 0),
    (2, -1),
    (3, 1),
]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("shifts_dims", ROLL_SHIFTS_DIMS)
def test_roll_single_dim(shape, dtype, shifts_dims):
    shifts, dims = shifts_dims
    ndim = len(shape)
    # Adjust dims if it's out of range for this shape
    if dims >= ndim or dims < -ndim:
        pytest.skip(f"dims {dims} out of range for shape {shape}")

    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


ROLL_MULTI_DIMS = [
    ((1, 2), (0, 1)),
    ((-1, 1), (0, -1)),
    ((2, -2), (-2, -1)),
]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shifts_dims", ROLL_MULTI_DIMS)
def test_roll_multi_dims(shape, dtype, shifts_dims):
    shifts, dims = shifts_dims
    ndim = len(shape)
    # Check all dims are valid for this shape
    for d in dims:
        if d >= ndim or d < -ndim:
            pytest.skip(f"dims {d} out of range for shape {shape}")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


ROLL_FLATTEN_SHIFTS = [1, -1, 5, -3]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shifts", ROLL_FLATTEN_SHIFTS)
def test_roll_flatten(shape, dtype, shifts):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    # Roll without specifying dims (flatten case)
    ref_out = torch.roll(ref_inp, shifts)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_roll_with_non_dense_input(shape, dtype):
    if len(shape) < 2:
        pytest.skip("Need at least 2D for non-dense test")

    shape_dilated = tuple(item * 2 for item in shape)
    inp = torch.randn(shape_dilated, dtype=dtype, device=flag_gems.device)[::2, ::2]
    ref_inp = utils.to_reference(inp, False)

    shifts = 2
    dims = 0

    ref_out = torch.roll(ref_inp, shifts, dims)
    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)
