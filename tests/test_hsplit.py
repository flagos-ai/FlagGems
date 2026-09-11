import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hsplit
@pytest.mark.parametrize(
    "shape",
    [(128,), (64, 128), (32, 64, 128), (16, 32, 64, 128)],
)
# View operation supports all dtypes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("sections", [2, 4])
def test_hsplit_int(shape, dtype, sections):
    """Test hsplit.int accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.hsplit(ref_inp, sections)
    with flag_gems.use_gems():
        res_out = torch.hsplit(inp, sections)

    assert len(res_out) == len(
        ref_out
    ), f"hsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.hsplit
@pytest.mark.parametrize(
    "shape",
    [(128,), (64, 128), (32, 64, 128)],
)
# View operation supports all dtypes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("indices", [[32], [16, 48], [20, 40, 80]])
def test_hsplit_array(shape, dtype, indices):
    """Test hsplit.array accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.hsplit(ref_inp, indices)
    with flag_gems.use_gems():
        res_out = torch.hsplit(inp, indices)

    assert len(res_out) == len(
        ref_out
    ), f"hsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)
