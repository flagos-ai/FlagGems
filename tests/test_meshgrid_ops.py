import pytest
import torch

from flag_gems.ops.meshgrid import meshgrid


def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F841

        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    return "cpu"


DEVICE = get_available_device()


@pytest.mark.meshgrid
@pytest.mark.correctness
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_basic(indexing):
    x = torch.randn(5, device=DEVICE)
    y = torch.randn(5, device=DEVICE)

    our_out = meshgrid([x, y], indexing=indexing)
    ref_out = torch.meshgrid(x, y, indexing=indexing)

    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.meshgrid
@pytest.mark.correctness
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_different_sizes(indexing):
    x = torch.randn(3, device=DEVICE)
    y = torch.randn(5, device=DEVICE)

    our_out = meshgrid([x, y], indexing=indexing)
    ref_out = torch.meshgrid(x, y, indexing=indexing)

    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.meshgrid
@pytest.mark.dimensional
@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_multidimensional(ndim, indexing):
    tensors = [torch.randn(3 + i, device=DEVICE) for i in range(ndim)]

    our_out = meshgrid(tensors, indexing=indexing)
    ref_out = torch.meshgrid(*tensors, indexing=indexing)

    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.meshgrid
@pytest.mark.dtype
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_meshgrid_dtypes(dtype):
    x = torch.tensor([1, 2, 3], dtype=dtype, device=DEVICE)
    y = torch.tensor([4, 5, 6], dtype=dtype, device=DEVICE)

    our_out = meshgrid([x, y], indexing="ij")
    ref_out = torch.meshgrid(x, y, indexing="ij")

    for our, ref in zip(our_out, ref_out):
        assert our.dtype == ref.dtype
        if dtype in [torch.int32, torch.int64]:
            assert torch.equal(our, ref)
        else:
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)
