import pytest
import torch

from flag_gems.ops.meshgrid import meshgrid


def get_available_device()
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    return "cpu"


DEVICE = get_available_device()


def test_meshgrid_correctness():
    
    sizes = [1, 2, 3, 4, 5, 10, 100]

    for size in sizes:
        x = torch.randn(size, device=DEVICE)
        y = torch.randn(size, device=DEVICE)

        for indexing in ["ij", "xy"]:
            our_out = meshgrid([x, y], indexing=indexing)
            ref_out = torch.meshgrid(x, y, indexing=indexing)

            for i, (our, ref) in enumerate(zip(our_out, ref_out)):
                if size == 1:
                    assert torch.allclose(
                        our, ref, rtol=1e-4, atol=1e-4
                    ), f"Failed for size {size}, indexing {indexing}, output {i}"
                else:
                    assert torch.allclose(
                        our, ref, rtol=1e-5, atol=1e-5
                    ), f"Failed for size {size}, indexing {indexing}, output {i}"


def test_meshgrid_xy_single_element():
    
    test_cases = [
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([-1.0]), torch.tensor([3.14])),
        (torch.tensor([0.0]), torch.tensor([0.0])),
    ]

    for x, y in test_cases:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        our_out = meshgrid([x, y], indexing="xy")
        ref_out = torch.meshgrid(x, y, indexing="xy")

        for our, ref in zip(our_out, ref_out):
            assert our.shape == ref.shape
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_xy_mode():

    x = torch.tensor([1, 2, 3], device=DEVICE)
    y = torch.tensor([4, 5], device=DEVICE)

    our_out = meshgrid([x, y], indexing="xy")
    ref_out = torch.meshgrid(x, y, indexing="xy")

    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref)

    x = torch.tensor([1, 2, 3], device=DEVICE)
    y = torch.tensor([1, 2, 3], device=DEVICE)

    our_out = meshgrid([x, y], indexing="xy")
    ref_out = torch.meshgrid(x, y, indexing="xy")

    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref)


def test_meshgrid_3d():
    x = torch.randn(4, device=DEVICE)
    y = torch.randn(5, device=DEVICE)
    z = torch.randn(6, device=DEVICE)

    for indexing in ["ij", "xy"]:
        our_out = meshgrid([x, y, z], indexing=indexing)
        ref_out = torch.meshgrid(x, y, z, indexing=indexing)

        for our, ref in zip(our_out, ref_out):
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_4d():
    tensors = [torch.randn(3, device=DEVICE) for _ in range(4)]

    for indexing in ["ij", "xy"]:
        our_out = meshgrid(tensors, indexing=indexing)
        ref_out = torch.meshgrid(*tensors, indexing=indexing)

        for our, ref in zip(our_out, ref_out):
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_different_dtypes():
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        x = torch.tensor([1, 2, 3], dtype=dtype, device=DEVICE)
        y = torch.tensor([4, 5, 6], dtype=dtype, device=DEVICE)

        for indexing in ["ij", "xy"]:
            our_out = meshgrid([x, y], indexing=indexing)
            ref_out = torch.meshgrid(x, y, indexing=indexing)

            for our, ref in zip(our_out, ref_out):
                assert our.dtype == ref.dtype
                if dtype in [torch.int32, torch.int64]:
                    assert torch.equal(our, ref)
                else:
                    assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_edge_cases():
    x = torch.randn(2, device=DEVICE)
    y = torch.randn(5, device=DEVICE)
    z = torch.randn(3, device=DEVICE)

    for indexing in ["ij", "xy"]:
        our_out = meshgrid([x, y, z], indexing=indexing)
        ref_out = torch.meshgrid(x, y, z, indexing=indexing)

        for our, ref in zip(our_out, ref_out):
            assert our.shape == ref.shape
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)

    x = torch.randn(1, device=DEVICE)
    y = torch.randn(100, device=DEVICE)

    our_out = meshgrid([x, y], indexing="ij")
    ref_out = torch.meshgrid(x, y, indexing="ij")

    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref)


def test_meshgrid_error_handling():

    with pytest.raises(ValueError, match="tensors must be a non-empty list or tuple"):
        meshgrid([])

    x = torch.randn(2, device=DEVICE)
    with pytest.raises(ValueError, match="indexing must be 'ij' or 'xy'"):
        meshgrid([x], indexing="invalid")

    tensors = [torch.randn(2, device=DEVICE) for _ in range(5)]
    with pytest.raises(
        NotImplementedError, match="Currently only supports up to 4 dimensions"
    ):
        meshgrid(tensors)

    x = torch.randn(2, 3, device=DEVICE)
    with pytest.raises(ValueError, match="must be 1D"):
        meshgrid([x])

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        meshgrid([1, 2, 3])

# meshgrid tests complete

# meshgrid tests complete
