import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.dsplit
@pytest.mark.parametrize(
    "shape",
    [(8, 16, 32), (32, 64, 128), (16, 32, 64, 128)],
)
# View operation supports all dtypes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("sections", [2, 4])
def test_dsplit_int(shape, dtype, sections):
    """Test dsplit.int accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.dsplit(ref_inp, sections)
    res_out = flag_gems.dsplit(inp, sections)

    assert len(res_out) == len(
        ref_out
    ), f"dsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.dsplit
@pytest.mark.parametrize(
    "shape",
    [(32, 64, 128), (16, 32, 64, 128)],
)
# View operation supports all dtypes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("indices", [[32], [16, 48], [20, 40, 80]])
def test_dsplit_array(shape, dtype, indices):
    """Test dsplit.array accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.dsplit(ref_inp, indices)
    res_out = flag_gems.dsplit(inp, indices)

    assert len(res_out) == len(
        ref_out
    ), f"dsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.dsplit
@pytest.mark.parametrize(
    "shape",
    [(16, 32, 64, 128), (32, 64, 128)],
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
def test_dsplit_array_out_of_range(shape, dtype, indices):
    """dsplit tolerates indices torch accepts: out of range, duplicate, unsorted."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.dsplit(ref_inp, indices)
    res_out = flag_gems.dsplit(inp, indices)

    assert len(res_out) == len(
        ref_out
    ), f"dsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        assert (
            res.shape == ref.shape
        ), f"dsplit section {i} shape mismatch: {tuple(res.shape)} vs {tuple(ref.shape)}"
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.dsplit
@pytest.mark.parametrize("shape", [(128,), (64, 128)])
def test_dsplit_requires_3d(shape):
    """dsplit requires at least 3 dimensions, matching torch.dsplit."""
    inp = torch.randn(shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        flag_gems.dsplit(inp, 2)
    with pytest.raises(RuntimeError):
        torch.dsplit(ref_inp, 2)
