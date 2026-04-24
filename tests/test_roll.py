import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ROLL_SINGLE_DIM_CASES = [
    (1, 0),
    (-1, 0),
    (2, -1),
    (3, 1),
]

ROLL_MULTI_DIM_CASES = [
    ((1, 2), (0, 1)),
    ((-1, 1), (0, -1)),
    ((2, -2), (-2, -1)),
    ((1, 2), (0, 0)),
]

ROLL_FLATTEN_SHIFTS = [1, -1, 5, -3]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(
            0, 2, shape, device=flag_gems.device, dtype=torch.int32
        ).to(torch.bool)
    if dtype in utils.ALL_FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("shifts,dims", ROLL_SINGLE_DIM_CASES)
def test_roll_single_dim(shape, dtype, shifts, dims):
    ndim = len(shape)
    if dims >= ndim or dims < -ndim:
        pytest.skip(f"dims {dims} out of range for shape {shape}")

    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)
    ref_out = torch.roll(ref_inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize("shape", [s for s in utils.POINTWISE_SHAPES if len(s) >= 2])
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
@pytest.mark.parametrize("shifts,dims", ROLL_MULTI_DIM_CASES)
def test_roll_multi_dims(shape, dtype, shifts, dims):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts, dims)
    ref_out = torch.roll(ref_inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
@pytest.mark.parametrize("shifts", ROLL_FLATTEN_SHIFTS)
def test_roll_flatten(shape, dtype, shifts):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts)
    ref_out = torch.roll(ref_inp, shifts)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_roll_with_non_dense_input(dtype):
    base = _make_input((8, 10, 12), dtype)
    inp = base[:, ::2, :].transpose(0, 2)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts=(2, -3), dims=(0, 2))
    ref_out = torch.roll(ref_inp, shifts=(2, -3), dims=(0, 2))

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
def test_roll_empty_tensor():
    inp = torch.randn((2, 0, 3), device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts=5, dims=1)
    ref_out = torch.roll(ref_inp, shifts=5, dims=1)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
def test_roll_scalar_tensor():
    inp = torch.tensor(5, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.roll(inp, shifts=3)
    ref_out = torch.roll(ref_inp, shifts=3)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize(
    "kwargs, error_type",
    [
        ({"shifts": (1, 2)}, RuntimeError),
        ({"shifts": 1, "dims": (0, 1)}, RuntimeError),
        ({"shifts": (1, 2, 3), "dims": (0, 1)}, RuntimeError),
        ({"shifts": 1, "dims": 3}, IndexError),
        ({"shifts": 1, "dims": -4}, IndexError),
    ],
)
def test_roll_invalid_args(kwargs, error_type):
    inp = torch.randn((2, 3, 4), device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(error_type):
            torch.roll(inp, **kwargs)
