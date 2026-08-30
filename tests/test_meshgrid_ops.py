import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device


@pytest.mark.meshgrid
@pytest.mark.correctness
@pytest.mark.parametrize(
    "shapes",
    [
        [(5,), (5,)],
        [(3,), (5,)],
        [(1,), (1,)],
        [(7,), (3,)],
    ],
    ids=["same_size", "diff_size_3_5", "one_dim", "diff_size_7_3"],
)
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_basic(shapes, indexing):
    tensors = [torch.randn(shape, device=DEVICE) for shape in shapes]

    with flag_gems.use_gems():
        our_out = torch.meshgrid(*tensors, indexing=indexing)
    ref_out = torch.meshgrid(*tensors, indexing=indexing)

    for our, ref in zip(our_out, ref_out):
        utils.gems_assert_close(our, ref, our.dtype)


@pytest.mark.meshgrid
@pytest.mark.dimensional
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_multidimensional(ndim, indexing):
    tensors = [torch.randn(3 + i, device=DEVICE) for i in range(ndim)]

    with flag_gems.use_gems():
        our_out = torch.meshgrid(*tensors, indexing=indexing)
    ref_out = torch.meshgrid(*tensors, indexing=indexing)

    for our, ref in zip(our_out, ref_out):
        utils.gems_assert_close(our, ref, our.dtype)


@pytest.mark.meshgrid
@pytest.mark.dtype
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.int32, torch.int64],
    ids=["float32", "float64", "int32", "int64"],
)
def test_meshgrid_dtypes(dtype):
    x = torch.tensor([1, 2, 3], dtype=dtype, device=DEVICE)
    y = torch.tensor([4, 5, 6], dtype=dtype, device=DEVICE)

    with flag_gems.use_gems():
        our_out = torch.meshgrid(x, y, indexing="ij")
    ref_out = torch.meshgrid(x, y, indexing="ij")

    for our, ref in zip(our_out, ref_out):
        assert our.dtype == ref.dtype
        utils.gems_assert_close(our, ref, dtype)
