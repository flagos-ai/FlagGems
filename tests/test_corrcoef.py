import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# corrcoef treats rows as variables and columns as observations, so the
# input must be a 2D matrix. These shapes cover a single-variable degenerate
# case, square matrices, and tall/wide matrices.
CORRCOEF_SHAPES = [
    (1, 8),
    (2, 4),
    (4, 16),
    (8, 64),
    (16, 256),
    (32, 1024),
    (64, 128),
    (128, 512),
]


@pytest.mark.corrcoef
@pytest.mark.parametrize("shape", CORRCOEF_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_corrcoef(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_out = torch.corrcoef(inp)

    ref_out = torch.corrcoef(ref_inp).to(dtype)

    # The correlation matrix is NxN; use a larger reduce_dim tolerance since
    # the matrix product accumulates over n_cols observations.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[1])


@pytest.mark.corrcoef
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_corrcoef_1d(dtype):
    # A 1D input represents a single variable; the correlation coefficient of a
    # variable with itself is 1.0.
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.corrcoef(inp)

    ref_out = torch.corrcoef(utils.to_reference(inp, upcast=True)).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.corrcoef
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_corrcoef_non_contiguous(dtype):
    # Verify corrcoef handles non-contiguous inputs by working on a strided
    # slice of a larger matrix.
    base = torch.randn(64, 256, dtype=dtype, device=flag_gems.device)
    inp = base[::2, ::2]
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_out = torch.corrcoef(inp)

    ref_out = torch.corrcoef(ref_inp).to(dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.shape[1])
