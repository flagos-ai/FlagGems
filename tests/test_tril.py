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


@pytest.mark.tril
@pytest.mark.parametrize(
    "shape", [(64, 64), (128, 256), (256, 128), (1024, 1024), (33, 47)]
)
@pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_2d_various_diagonals(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.tril(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tril
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_batched(dtype):
    inp = torch.randn((4, 8, 16, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for diagonal in [-2, 0, 2]:
        ref_out = torch.tril(ref_inp, diagonal)
        with flag_gems.use_gems():
            res_out = torch.tril(inp, diagonal)
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tril
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_extreme_diagonals(dtype):
    # diagonals far outside the matrix become full-zero or full-copy fast paths
    inp = torch.randn((32, 32), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for diagonal in [-100, -33, 32, 100]:
        ref_out = torch.tril(ref_inp, diagonal)
        with flag_gems.use_gems():
            res_out = torch.tril(inp, diagonal)
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tril
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_non_square(dtype):
    for shape in [(7, 13), (13, 7), (1, 64), (64, 1)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)

        for diagonal in [-1, 0, 1]:
            ref_out = torch.tril(ref_inp, diagonal)
            with flag_gems.use_gems():
                res_out = torch.tril(inp, diagonal)
            utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tril
def test_tril_empty_tensor():
    inp = torch.empty((0, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tril(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.tril
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.int16, torch.int8])
def test_tril_integer_dtypes(dtype):
    """torch.tril supports integer tensors — cover them too."""
    for shape in [(8, 8), (32, 16), (16, 32), (4, 5, 7)]:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        for diagonal in [-1, 0, 1]:
            ref_out = torch.tril(ref_inp, diagonal)
            with flag_gems.use_gems():
                res_out = torch.tril(inp, diagonal)
            utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.tril
def test_tril_bool_dtype():
    """torch.tril supports bool tensors — verify."""
    inp = (torch.randn((16, 16), device=flag_gems.device) > 0)
    ref_inp = utils.to_reference(inp)
    for diagonal in [-1, 0, 1]:
        ref_out = torch.tril(ref_inp, diagonal)
        with flag_gems.use_gems():
            res_out = torch.tril(inp, diagonal)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.tril
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_large_shapes(dtype):
    """Stress test: tile-boundary fast paths must hold for large matrices."""
    for shape in [(2048, 2048), (4096, 1024), (1024, 4096)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        for diagonal in [-100, 0, 100]:
            ref_out = torch.tril(ref_inp, diagonal)
            with flag_gems.use_gems():
                res_out = torch.tril(inp, diagonal)
            utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (128, 64), (5, 7, 13)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out_buf = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.tril(ref_inp, 0, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.tril(inp, 0, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
