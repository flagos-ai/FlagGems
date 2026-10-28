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
    res_out = flag_gems.hsplit(inp, sections)

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
    res_out = flag_gems.hsplit(inp, indices)

    assert len(res_out) == len(
        ref_out
    ), f"hsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.hsplit
@pytest.mark.parametrize(
    "shape",
    [(64, 128), (32, 64, 128)],
)
# View operation supports all dtypes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "indices",
    [
        [20, 40, 80],  # past the split dim: trailing section is empty
        [20, 40, 128],  # exactly at the boundary
        [70],  # single index past the dim
        [10, 10, 20],  # duplicate index
        [30, 10],  # unsorted index
        [-1],  # negative index wraps
    ],
)
def test_hsplit_array_out_of_range(shape, dtype, indices):
    """hsplit tolerates indices torch accepts: out of range, duplicate, unsorted."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.hsplit(ref_inp, indices)
    res_out = flag_gems.hsplit(inp, indices)

    assert len(res_out) == len(
        ref_out
    ), f"hsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        assert (
            res.shape == ref.shape
        ), f"hsplit section {i} shape mismatch: {tuple(res.shape)} vs {tuple(ref.shape)}"
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)
