import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# Shapes covering 1D to 4D tensors with various dimension sizes
if cfg.QUICK_MODE:
    # Quick mode: minimal shapes for fast testing
    UNSAFE_SPLIT_WITH_SIZES_SHAPES = [
        (10,),
        (16, 32),
    ]
else:
    # Full mode: 1D-4D shapes including edge cases (size-1 dim, large vocab dim)
    UNSAFE_SPLIT_WITH_SIZES_SHAPES = [
        (10,),
        (10, 4),
        (10, 4, 8),
        (10, 4, 8, 16),
        (16, 32),
        (8, 64, 128),
        (1, 8192),
        (32, 50257),
    ]


@pytest.mark.unsafe_split_with_sizes
@pytest.mark.parametrize("shape", UNSAFE_SPLIT_WITH_SIZES_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_split_with_sizes(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Define split sizes that sum to the first dimension size
    dim_size = shape[0]
    split_sizes = [dim_size // 4, dim_size // 4, dim_size - 2 * (dim_size // 4)]
    dim = 0

    ref_out = torch.unsafe_split_with_sizes(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.unsafe_split_with_sizes(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.unsafe_split_with_sizes
@pytest.mark.parametrize("shape", [(10, 4, 8), (20, 32, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_unsafe_split_with_sizes_different_dims(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Generate split sizes that sum to the dimension size
    dim_size = shape[dim]
    split_sizes = [dim_size // 4, dim_size // 4, dim_size - 2 * (dim_size // 4)]

    ref_out = torch.unsafe_split_with_sizes(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.unsafe_split_with_sizes(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.unsafe_split_with_sizes
@pytest.mark.parametrize("shape", [(10, 4), (8, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_split_with_sizes_edge_cases(shape, dtype):
    # Test with zero-size split
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    split_sizes = [2, 0, shape[0] - 2]
    dim = 0

    ref_out = torch.unsafe_split_with_sizes(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.unsafe_split_with_sizes(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.unsafe_split_with_sizes
@pytest.mark.parametrize("shape", [(0, 4), (4, 0)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_split_with_sizes_all_zero_splits(shape, dtype):
    # split_sizes containing only zeros, valid only when the split dim
    # itself has size 0.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    dim = 0 if shape[0] == 0 else 1
    split_sizes = [0, 0]

    ref_out = torch.unsafe_split_with_sizes(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.unsafe_split_with_sizes(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.unsafe_split_with_sizes
@pytest.mark.parametrize("shape", [(8,), (10, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_split_with_sizes_single_element_splits(shape, dtype):
    # Single-element splits interleaved with leading/trailing zero-size
    # splits, including the degenerate 1-element tail.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    dim_size = shape[0]
    split_sizes = [0, 1, 1, dim_size - 2, 0]
    dim = 0

    ref_out = torch.unsafe_split_with_sizes(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.unsafe_split_with_sizes(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)
