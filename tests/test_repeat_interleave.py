import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

REPEAT_INTERLEAVE_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

REPEAT_INTERLEAVE_DIM = [-1, 0, None]


@pytest.mark.repeat_interleave_self_int
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES + [(1,)])
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_interleave_self_int(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = 2
    ref_inp = utils.to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave_self_int
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_interleave_self_int_non_contiguous(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)[::2]
    repeats = 2
    ref_inp = utils.to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave_tensor
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", [torch.int32])
def test_repeat_interleave_tensor(shape, dtype):
    repeats = torch.randint(0, 30, shape, dtype=dtype, device=flag_gems.device)
    ref_repeats = utils.to_reference(repeats)
    ref_out = torch.repeat_interleave(ref_repeats)

    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(repeats)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave_self_tensor
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_interleave_self_tensor(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = torch.randint(0, 30, (shape[dim],), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_repeats = utils.to_reference(repeats)

    ref_out = torch.repeat_interleave(ref_inp, ref_repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    utils.gems_assert_equal(res_out, ref_out)
