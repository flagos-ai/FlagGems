import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for diagonal_scatter tests (square, non-square, batched)
DIAGONAL_SCATTER_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (32, 64),
    (64, 32),
    (32, 64, 64),
    (16, 32, 32, 32),
]


@pytest.mark.diagonal_scatter
@pytest.mark.parametrize("shape", DIAGONAL_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    diag = torch.diagonal(inp, 0, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, 0, -2, -1)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, 0, -2, -1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_scatter
@pytest.mark.parametrize("shape", DIAGONAL_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter_offset(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test with positive offset
    diag = torch.diagonal(inp, 1, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, 1, -2, -1)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, 1, -2, -1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_scatter
@pytest.mark.parametrize("shape", DIAGONAL_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter_negative_offset(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test with negative offset
    diag = torch.diagonal(inp, -1, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, -1, -2, -1)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, -1, -2, -1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_scatter
@pytest.mark.parametrize(
    "shape, dim1, dim2",
    [
        ((8, 16, 32), 0, 1),
        ((8, 16, 32), 0, 2),
        ((8, 16, 32), 1, 0),
        ((4, 8, 16, 32), 0, 1),
        ((4, 8, 16, 32), 1, 2),
        ((4, 8, 16, 32), 0, 3),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter_non_last_dims(shape, dim1, dim2, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    diag = torch.diagonal(inp, 0, dim1, dim2)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, 0, dim1, dim2)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, 0, dim1, dim2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_scatter
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter_large_offset(dtype):
    # When offset reaches the boundary, diagonal length becomes 1
    inp = torch.randn(8, 8, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    offset = 7  # diagonal length = 1
    diag = torch.diagonal(inp, offset, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, offset, -2, -1)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, offset, -2, -1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_scatter
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter_non_square(dtype):
    # Wide matrix
    inp = torch.randn(4, 16, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    diag = torch.diagonal(inp, 0, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, 0, -2, -1)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, 0, -2, -1)

    utils.gems_assert_close(res_out, ref_out, dtype)

    # Tall matrix
    inp = torch.randn(16, 4, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    diag = torch.diagonal(inp, 0, -2, -1)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)
    ref_src = utils.to_reference(src)

    ref_out = torch.diagonal_scatter(ref_inp, ref_src, 0, -2, -1)
    with flag_gems.use_gems():
        res_out = torch.diagonal_scatter(inp, src, 0, -2, -1)

    utils.gems_assert_close(res_out, ref_out, dtype)
