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

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous(), "tril output should be contiguous"


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_noncontiguous(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = utils.unsqueeze_tensor(inp, 2)

    if inp.dim() >= 2:
        inp = inp.transpose(-2, -1)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous(), "tril output should always be contiguous"


@pytest.mark.tril_
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = utils.unsqueeze_tensor(inp, 2)

    ref_inp = utils.to_reference(inp.clone())
    inp_original = inp.clone().detach()

    ref_inp.tril_(diagonal)

    original_stride = inp.stride()
    original_data_ptr = inp.data_ptr()

    with flag_gems.use_gems():
        res = inp.tril_(diagonal)

    utils.gems_assert_equal(inp, ref_inp)

    assert res is inp, "tril_ should return the input tensor itself to support chaining"

    assert (
        inp.data_ptr() == original_data_ptr
    ), "tril_ should modify in-place, data pointer should not change"

    assert (
        inp.stride() == original_stride
    ), f"tril_ should preserve stride. Expected {original_stride}, got {inp.stride()}"

    M, N = inp.shape[-2], inp.shape[-1]
    should_modify = any(j > i + diagonal for i in range(M) for j in range(N))

    if should_modify:
        assert not torch.allclose(inp, inp_original), (
            "tril_ in-place modification not implemented, "
            "the original tensor remains unchanged!"
        )


@pytest.mark.tril_
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_inplace_noncontiguous(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = utils.unsqueeze_tensor(inp, 2)

    if inp.dim() >= 2:
        inp = inp.transpose(-2, -1)

    ref_inp = utils.to_reference(inp.clone())
    inp_original = inp.clone().detach()

    ref_inp.tril_(diagonal)

    original_stride = inp.stride()
    original_data_ptr = inp.data_ptr()

    with flag_gems.use_gems():
        res = inp.tril_(diagonal)

    utils.gems_assert_equal(inp, ref_inp)

    assert res is inp, "tril_ should return the input tensor itself to support chaining"

    assert inp.stride() == original_stride, (
        f"tril_ should preserve stride even for non-contiguous input. "
        f"Expected {original_stride}, got {inp.stride()}"
    )

    assert (
        inp.data_ptr() == original_data_ptr
    ), "tril_ should modify in-place, data pointer should not change"

    M, N = inp.shape[-2], inp.shape[-1]
    should_modify = any(j > i + diagonal for i in range(M) for j in range(N))

    if should_modify:
        assert not torch.allclose(
            inp, inp_original
        ), "tril_ should modify non-contiguous tensor in-place"
