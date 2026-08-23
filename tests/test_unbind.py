import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# unbind test shapes
UNBIND_SHAPES = [
    (2, 3),
    (3, 4),
    (4, 8),
    (2, 3, 4),
    (4, 8, 16),
    (32, 64, 128),
    (2, 4, 8, 16),
]


@pytest.mark.unbind
@pytest.mark.parametrize("shape", UNBIND_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unbind(shape, dim, dtype):
    if dim >= len(shape):
        pytest.skip(f"dim {dim} >= len(shape) {len(shape)}")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unbind(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.unbind(inp, dim)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"

    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_equal(res, ref)
        # unbind is a view op: every slice must share storage with the input.
        assert res.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()


@pytest.mark.unbind
@pytest.mark.parametrize("shape", UNBIND_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unbind_default_dim(shape, dtype):
    # Test with default dim=0
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unbind(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.unbind(inp)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"

    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_equal(res, ref)


@pytest.mark.unbind
@pytest.mark.parametrize("shape", UNBIND_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unbind_negative_dim(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unbind(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.unbind(inp, dim)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"

    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_equal(res, ref)
