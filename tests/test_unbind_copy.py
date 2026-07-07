import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# unbind_copy test shapes
UNBIND_COPY_SHAPES = [
    (2, 3),
    (3, 4),
    (4, 8),
    (2, 3, 4),
    (4, 8, 16),
    (2, 4, 8, 16),
]


@pytest.mark.unbind_copy
@pytest.mark.parametrize("shape", UNBIND_COPY_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unbind_copy(shape, dim, dtype):
    if dim >= len(shape):
        pytest.skip(f"dim {dim} >= len(shape) {len(shape)}")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unbind_copy(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.unbind_copy(inp, dim)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"

    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.unbind_copy
@pytest.mark.parametrize("shape", UNBIND_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unbind_copy_default_dim(shape, dtype):
    # Test with default dim=0
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unbind_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.unbind_copy(inp)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"

    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)
