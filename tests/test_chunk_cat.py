import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.chunk_cat
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_cat_1d(dtype):
    # Test 1D tensor with single input
    shapes = [(16,), (32,), (64,)]
    num_chunks_list = [2, 4]

    for shape in shapes:
        for num_chunks in num_chunks_list:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            ref_inp = utils.to_reference(inp)

            ref_out = torch.ops.aten._chunk_cat.default(
                [ref_inp], dim=0, num_chunks=num_chunks
            )
            with flag_gems.use_gems():
                res_out = torch.ops.aten._chunk_cat.default(
                    [inp], dim=0, num_chunks=num_chunks
                )

            utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.chunk_cat
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_cat_2d_dim0(dtype):
    # Test 2D tensor with dim=0
    shapes = [(8, 16), (16, 32)]
    num_chunks_list = [2, 4]

    for shape in shapes:
        for num_chunks in num_chunks_list:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            ref_inp = utils.to_reference(inp)

            ref_out = torch.ops.aten._chunk_cat.default(
                [ref_inp], dim=0, num_chunks=num_chunks
            )
            with flag_gems.use_gems():
                res_out = torch.ops.aten._chunk_cat.default(
                    [inp], dim=0, num_chunks=num_chunks
                )

            utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.chunk_cat
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_cat_2d_dim1(dtype):
    # Test 2D tensor with dim=1
    shapes = [(8, 16), (16, 32)]
    num_chunks_list = [2, 4]

    for shape in shapes:
        for num_chunks in num_chunks_list:
            inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            ref_inp = utils.to_reference(inp)

            ref_out = torch.ops.aten._chunk_cat.default(
                [ref_inp], dim=1, num_chunks=num_chunks
            )
            with flag_gems.use_gems():
                res_out = torch.ops.aten._chunk_cat.default(
                    [inp], dim=1, num_chunks=num_chunks
                )

            utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.chunk_cat
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chunk_cat_multiple_tensors(dtype):
    # Test with multiple input tensors
    shape = (8, 16)
    num_chunks = 2

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten._chunk_cat.default(
        [ref_inp1, ref_inp2], dim=0, num_chunks=num_chunks
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._chunk_cat.default(
            [inp1, inp2], dim=0, num_chunks=num_chunks
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
