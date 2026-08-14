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
        [(512,), (512,)],
        [(1024,), (2048,)],
        [(256,), (256,)],
        [(4096,), (2048,)],
    ],
)
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_basic(shapes, indexing):
    tensors = [torch.randn(shape, device=DEVICE) for shape in shapes]
    ref_tensors = [utils.to_reference(inp, True) for inp in tensors]

    ref_out = torch.meshgrid(*ref_tensors, indexing=indexing)
    with flag_gems.use_gems():
        our_out = torch.meshgrid(*tensors, indexing=indexing)

    for our, ref in zip(our_out, ref_out):
        utils.gems_assert_close(our, ref, our.dtype)


@pytest.mark.meshgrid
@pytest.mark.dimensional
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_multidimensional(ndim, indexing):
    tensors = [torch.randn(64 + i * 32, device=DEVICE) for i in range(ndim)]
    ref_tensors = [utils.to_reference(inp, True) for inp in tensors]

    ref_out = torch.meshgrid(*ref_tensors, indexing=indexing)
    with flag_gems.use_gems():
        our_out = torch.meshgrid(*tensors, indexing=indexing)

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
    x = torch.arange(1, 513, dtype=dtype, device=DEVICE)
    y = torch.arange(1000, 2000, dtype=dtype, device=DEVICE)
    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)

    ref_out = torch.meshgrid(ref_x, ref_y, indexing="ij")
    with flag_gems.use_gems():
        our_out = torch.meshgrid(x, y, indexing="ij")

    for our, ref in zip(our_out, ref_out):
        ref_cast = ref.to(our.dtype)
        utils.gems_assert_close(our, ref_cast, our.dtype)
