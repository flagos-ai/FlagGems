import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_diagonal
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize(
    "shape, dim1, dim2",
    [
        ((256, 256), -2, -1),
        ((512, 512), 0, 1),
        ((128, 256, 256), 1, 2),
        ((64, 128, 128, 128), -1, -2),
        ((10, 20, 30, 40), 0, 3),
        ((5, 7, 11, 13), -3, -1),
        ((32, 64, 64, 64), 2, 3),
        ((2, 3, 4, 5, 6), 0, 4),
        ((2, 3, 4, 5, 6), 1, 3),
    ],
)
def test_linalg_diagonal_correctness(shape, dim1, dim2, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.diagonal(ref_A, dim1=dim1, dim2=dim2)
    with flag_gems.use_gems(include=["linalg_diagonal"]):
        result = torch.diagonal(A, dim1=dim1, dim2=dim2)
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.linalg_diagonal
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape, dim1, dim2, offset",
    [
        ((5, 5), 0, 1, 2),
        ((5, 5), 0, 1, -2),
        ((7, 7), -2, -1, 3),
        ((7, 7), -2, -1, -4),
        ((10, 10, 10), 0, 2, 1),
        ((10, 10, 10), 0, 2, -2),
        ((4, 8, 16), 0, 2, 3),
        ((4, 8, 16), 1, 2, -1),
    ],
)
def test_linalg_diagonal_offset(shape, dim1, dim2, offset, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.diagonal(ref_A, offset=offset, dim1=dim1, dim2=dim2)
    with flag_gems.use_gems(include=["linalg_diagonal"]):
        result = torch.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.linalg_diagonal
def test_linalg_diagonal_empty():
    A = torch.randn(3, 4, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.diagonal(ref_A, offset=10, dim1=0, dim2=1)
    with flag_gems.use_gems(include=["linalg_diagonal"]):
        result = torch.diagonal(A, offset=10, dim1=0, dim2=1)
    assert result.shape == ref_out.shape
    assert result.numel() == 0


@pytest.mark.linalg_diagonal
def test_linalg_diagonal_non_contiguous():
    A = torch.randn(4, 5, 6, device=flag_gems.device).transpose(0, 2)
    ref_A = utils.to_reference(A)
    ref_out = torch.diagonal(ref_A, dim1=1, dim2=2)
    with flag_gems.use_gems(include=["linalg_diagonal"]):
        result = torch.diagonal(A, dim1=1, dim2=2)
    utils.gems_assert_close(result, ref_out, torch.float32)


@pytest.mark.linalg_diagonal
def test_linalg_diagonal_2d_single_element():
    A = torch.tensor([[42.0]], device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.diagonal(ref_A)
    with flag_gems.use_gems(include=["linalg_diagonal"]):
        result = torch.diagonal(A)
    utils.gems_assert_close(result, ref_out, torch.float32)


@pytest.mark.linalg_diagonal
def test_linalg_diagonal_large():
    A = torch.randn((2048, 2048, 2048), dtype=torch.float32, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.diagonal(ref_A, dim1=1, dim2=2)
    with flag_gems.use_gems(include=["linalg_diagonal"]):
        result = torch.diagonal(A, dim1=1, dim2=2)
    utils.gems_assert_close(result, ref_out, torch.float32)
