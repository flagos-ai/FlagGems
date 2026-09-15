import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils

DTYPES = utils.FLOAT_DTYPES


# Core shapes exercised by SHAPES (mirrors worktree CI branch).
SHAPES = utils.REDUCTION_SHAPES + [(1, 8192), (32, 50257)]

# ``_aminmax`` / ``_aminmax_out`` start with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register
# them directly on the MarkGenerator so ``@pytest.mark._aminmax`` and
# ``-m _aminmax`` both work.
setattr(
    pytest.mark,
    "_aminmax",
    MarkDecorator(Mark("_aminmax", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_aminmax_out",
    MarkDecorator(Mark("_aminmax_out", (), {}, _ispytest=True), _ispytest=True),
)


@pytest.mark._aminmax
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test__aminmax(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark._aminmax
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test__aminmax_zero(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark._aminmax
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test__aminmax_inf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_flat = inp.float().flatten()
    inp_flat[0] = float("inf")
    inp_flat[1] = float("-inf")
    inp = inp_flat.reshape(shape).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min, equal_nan=True)
    utils.gems_assert_equal(res_max, ref_max, equal_nan=True)


@pytest.mark._aminmax
@pytest.mark.parametrize("shape", SHAPES)
def test__aminmax_int(shape):
    inp = torch.randint(-100, 100, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark._aminmax
def test__aminmax_scalar():
    inp = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark._aminmax_out
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test__aminmax_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min = torch.empty((), dtype=dtype, device=ref_inp.device)
    ref_max = torch.empty((), dtype=dtype, device=ref_inp.device)
    torch.ops.aten._aminmax.out(ref_inp, out0=ref_min, out1=ref_max)

    min_out = torch.empty((), dtype=dtype, device=flag_gems.device)
    max_out = torch.empty((), dtype=dtype, device=flag_gems.device)
    flag_gems._aminmax_out(inp, out0=min_out, out1=max_out)

    utils.gems_assert_equal(min_out, ref_min)
    utils.gems_assert_equal(max_out, ref_max)
