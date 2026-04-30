import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DIAGONAL = list(zip(utils.POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = utils.unsqueeze_tensor(inp, 2)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Extra 2D coverage (KernelGen tril path): non-square last two dims, same style as above.
SHAPES_2D_SMALL = [
    (1, 1),
    (8, 16),
    (32, 24),
]


@pytest.mark.tril
@pytest.mark.parametrize("shape", SHAPES_2D_SMALL)
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_2d_nonsquare(shape, diagonal, dtype):
    inp = torch.randn(*shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)
    utils.gems_assert_close(res_out, ref_out, dtype)
