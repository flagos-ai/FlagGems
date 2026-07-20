"""
Meshgrid operator tests for FlagGems.

This test suite validates the meshgrid implementation across different
devices (CPU, CUDA, NPU), indexing modes, dimensions, and data types.
"""

import os
import sys

import pytest
import torch

# Add project root to path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

# Detect available device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try to import torch_npu for NPU support
try:
    import torch_npu  # noqa: F401

    if torch.npu.is_available():
        DEVICE = "npu:0"
except (ImportError, AttributeError):
    pass

from flag_gems.ops.meshgrid import meshgrid  # noqa: E402


class TestMeshgrid:
    """Test meshgrid operator implementation."""

    @pytest.mark.parametrize(
        "shapes,indexing",
        [
            ([(3,), (4,)], "ij"),
            ([(3,), (4,)], "xy"),
            ([(2,), (3,), (4,)], "ij"),
            ([(2,), (3,), (4,)], "xy"),
            (
                [
                    (5,),
                ],
                "ij",
            ),
            ([(1,), (1,)], "ij"),
            ([(10,), (20,), (30,)], "ij"),
            ([(2,), (3,), (4,), (5,)], "ij"),
        ],
    )
    def test_accuracy_cuda(self, shapes, indexing):
        """Test accuracy on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensors = [
            torch.arange(s[0], device="cuda", dtype=torch.float32) for s in shapes
        ]
        torch_out = torch.meshgrid(tensors, indexing=indexing)
        our_out = meshgrid(tensors, indexing=indexing)

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "shapes,indexing",
        [
            ([(3,), (4,)], "ij"),
            ([(3,), (4,)], "xy"),
            ([(2,), (3,), (4,)], "ij"),
            ([(2,), (3,), (4,)], "xy"),
        ],
    )
    def test_accuracy_cpu(self, shapes, indexing):
        """Test accuracy on CPU device."""
        tensors = [
            torch.arange(s[0], device="cpu", dtype=torch.float32) for s in shapes
        ]
        torch_out = torch.meshgrid(tensors, indexing=indexing)
        our_out = meshgrid(tensors, indexing=indexing)

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

    def test_accuracy_npu(self):
        """Test accuracy on Huawei 910B NPU device."""
        try:
            import torch_npu  # noqa: F401, F811

            if not torch.npu.is_available():
                pytest.skip("NPU not available")

            device = "npu:0"
            x = torch.tensor([1, 2, 3], device=device)
            y = torch.tensor([4, 5, 6], device=device)

            torch_out = torch.meshgrid([x, y], indexing="ij")
            our_out = meshgrid([x, y], indexing="ij")

            for t_out, o_out in zip(torch_out, our_out):
                assert t_out.shape == o_out.shape
                assert torch.allclose(t_out.cpu(), o_out.cpu(), rtol=1e-5, atol=1e-5)

        except ImportError:
            pytest.skip("torch_npu not installed")

    def test_random_correctness(self):
        """Comprehensive correctness test with random data."""
        if DEVICE == "cpu":
            pytest.skip("Skipping large correctness test on CPU")

        sizes = [1, 2, 3, 4, 5, 10, 50]

        for size in sizes:
            x = torch.randn(size, device=DEVICE)
            y = torch.randn(size, device=DEVICE)

            for indexing in ["ij", "xy"]:
                our_out = meshgrid([x, y], indexing=indexing)
                ref_out = torch.meshgrid(x, y, indexing=indexing)

                for i, (our, ref) in enumerate(zip(our_out, ref_out)):
                    rtol = 1e-4 if size == 1 else 1e-5
                    assert torch.allclose(
                        our, ref, rtol=rtol, atol=1e-5
                    ), f"Failed for size {size}, indexing {indexing}, output {i}"

    def test_xy_single_element(self):
        """Test XY mode with single element tensors."""
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

    def test_xy_mode_comprehensive(self):
        """Comprehensive XY indexing mode test."""
        if DEVICE == "cpu":
            pytest.skip("Skipping XY mode test on CPU")

        x = torch.tensor([1, 2, 3], device=DEVICE)
        y = torch.tensor([4, 5], device=DEVICE)

        torch_xy = torch.meshgrid([x, y], indexing="xy")
        our_xy = meshgrid([x, y], indexing="xy")

        for t_out, o_out in zip(torch_xy, our_xy):
            assert torch.equal(t_out, o_out)

        z = torch.tensor([6, 7, 8], device=DEVICE)
        torch_xy_3d = torch.meshgrid([x, y, z], indexing="xy")
        our_xy_3d = meshgrid([x, y, z], indexing="xy")

        for t_out, o_out in zip(torch_xy_3d, our_xy_3d):
            assert torch.equal(t_out, o_out)

        w = torch.tensor([9, 10], device=DEVICE)
        torch_xy_4d = torch.meshgrid([x, y, z, w], indexing="xy")
        our_xy_4d = meshgrid([x, y, z, w], indexing="xy")

        for t_out, o_out in zip(torch_xy_4d, our_xy_4d):
            assert torch.equal(t_out, o_out)

    def test_3d(self):
        """Test 3D meshgrid."""
        if DEVICE == "cpu":
            pytest.skip("Skipping 3D test on CPU")

        x = torch.randn(4, device=DEVICE)
        y = torch.randn(5, device=DEVICE)
        z = torch.randn(6, device=DEVICE)

        for indexing in ["ij", "xy"]:
            our_out = meshgrid([x, y, z], indexing=indexing)
            ref_out = torch.meshgrid(x, y, z, indexing=indexing)

            for our, ref in zip(our_out, ref_out):
                assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)

    def test_4d(self):
        """Test 4D meshgrid."""
        if DEVICE == "cpu":
            pytest.skip("Skipping 4D test on CPU")

        tensors = [torch.randn(3, device=DEVICE) for _ in range(4)]

        for indexing in ["ij", "xy"]:
            our_out = meshgrid(tensors, indexing=indexing)
            ref_out = torch.meshgrid(*tensors, indexing=indexing)

            for our, ref in zip(our_out, ref_out):
                assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)

    def test_different_dtypes(self):
        """Test with different data types."""
        if DEVICE == "cpu":
            pytest.skip("Skipping dtype test on CPU")

        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.float16]

        for dtype in dtypes:
            x = torch.tensor([1, 2, 3], device=DEVICE, dtype=dtype)
            y = torch.tensor([4, 5, 6], device=DEVICE, dtype=dtype)

            torch_out = torch.meshgrid([x, y], indexing="ij")
            our_out = meshgrid([x, y], indexing="ij")

            for t_out, o_out in zip(torch_out, our_out):
                assert t_out.dtype == o_out.dtype
                if dtype == torch.float16:
                    assert torch.allclose(t_out, o_out, rtol=1e-3, atol=1e-3)
                else:
                    assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

    def test_empty_input(self):
        """Test empty input raises appropriate error."""
        with pytest.raises(ValueError, match="must be a non-empty list or tuple"):
            meshgrid([])

    def test_single_tensor(self):
        """Test single tensor input (1D case)."""
        if DEVICE == "cpu":
            pytest.skip("Skipping single tensor test on CPU")

        x = torch.tensor([1, 2, 3, 4, 5], device=DEVICE)

        torch_out = torch.meshgrid([x], indexing="ij")
        our_out = meshgrid([x], indexing="ij")

        assert len(torch_out) == 1
        assert len(our_out) == 1
        assert torch.equal(torch_out[0], our_out[0])
        assert torch_out[0].shape == (5,)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Empty input
        with pytest.raises(ValueError, match="must be a non-empty list or tuple"):
            meshgrid([])

        # Non-tensor input
        with pytest.raises(TypeError):
            meshgrid([1, 2, 3])

        # Invalid indexing mode
        x = torch.tensor([1, 2, 3], device=DEVICE)
        with pytest.raises(ValueError, match="indexing must be 'ij' or 'xy'"):
            meshgrid([x], indexing="invalid")

        # 2D tensor input (should fail)
        x_2d = torch.tensor([[1, 2], [3, 4]], device=DEVICE)
        with pytest.raises(ValueError, match="must be 1D"):
            meshgrid([x_2d])

        # More than 4 dimensions
        if DEVICE != "cpu":
            tensors = [torch.randn(2, device=DEVICE) for _ in range(5)]
            with pytest.raises(
                NotImplementedError, match="only supports up to 4 dimensions"
            ):
                meshgrid(tensors)

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_large_inputs(self, size):
        """Test with larger input tensors."""
        if DEVICE == "cpu":
            pytest.skip("Skipping large input test on CPU")

        x = torch.arange(size, device=DEVICE, dtype=torch.float32)
        y = torch.arange(size, device=DEVICE, dtype=torch.float32)

        torch_out = torch.meshgrid([x, y], indexing="ij")
        our_out = meshgrid([x, y], indexing="ij")

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

    def test_contiguity(self):
        """Test that output tensors are contiguous."""
        if DEVICE == "cpu":
            pytest.skip("Skipping contiguity test on CPU")

        x = torch.tensor([1, 2, 3], device=DEVICE)
        y = torch.tensor([4, 5], device=DEVICE)

        our_out = meshgrid([x, y], indexing="ij")

        for t in our_out:
            assert t.numel() == t.shape[0] * t.shape[1]
            t_contiguous = t.contiguous()
            assert t_contiguous.shape == t.shape
            assert torch.equal(t_contiguous, t)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
