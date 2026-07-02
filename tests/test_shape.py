import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shape operator: dtype-independent, returns tensor dimensions only.
SHAPE_SHAPES = [(2, 3, 5), (10, 20), (5,), (1, 1, 1, 1), (100,)]


@pytest.mark.shape
@pytest.mark.parametrize("shape", SHAPE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_shape(shape, dtype):
    from flag_gems.ops.shape import shape as shape_fn

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = shape_fn(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.tensor(
        list(ref_inp.shape), dtype=torch.int64, device=ref_inp.device
    )
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.shape
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_shape_scalar(dtype):
    from flag_gems.ops.shape import shape as shape_fn

    inp = torch.tensor(3.14, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = shape_fn(inp)
    ref_out = torch.tensor([], dtype=torch.int64, device=inp.device)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.shape_
def test_shape_():
    from flag_gems.ops.shape import shape_ as shape__fn

    inp = torch.randn((2, 3, 5), device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = shape__fn(inp)
    ref_inp = utils.to_reference(inp)
    utils.gems_assert_equal(res_out, ref_inp)
