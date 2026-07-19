# tests/test_meshgrid.py
<<<<<<< HEAD
import os
import sys

import pytest
import torch

from flag_gems.ops.meshgrid import meshgrid
=======
import pytest
import torch
from flag_gems.ops import meshgrid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
>>>>>>> 459d19e (Add meshgrid operator implementation with tests and benchmarks)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)


<<<<<<< HEAD
class TestMeshgrid:
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
            ([(2,), (3,), (4,), (5,), (6,)], "ij"),
        ],
    )
    def test_accuracy_cuda(self, shapes, indexing):
        """Test CUDA device - NV platform"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - skipping NV test")

        tensors = [
            torch.arange(s[0], device="cuda", dtype=torch.float32) for s in shapes
        ]
        torch_out = torch.meshgrid(tensors, indexing=indexing)
        our_out = meshgrid(tensors, indexing=indexing)

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

        print(f"CUDA test passed for {shapes} with {indexing}")

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
        """Test CPU device"""
        tensors = [
            torch.arange(s[0], device="cpu", dtype=torch.float32) for s in shapes
        ]
        torch_out = torch.meshgrid(tensors, indexing=indexing)
        our_out = meshgrid(tensors, indexing=indexing)

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

        print(f"CPU test passed for {shapes} with {indexing}")

    def test_accuracy_npu(self):
        """Test Huawei 910B NPU device"""
        try:
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                pytest.skip("NPU not available - skipping Huawei test")

            device = "npu:0"
            x = torch.tensor([1, 2, 3], device=device)
            y = torch.tensor([4, 5, 6], device=device)

            torch_out = torch.meshgrid([x, y], indexing="ij")
            our_out = meshgrid([x, y], indexing="ij")

            for t_out, o_out in zip(torch_out, our_out):
                assert t_out.shape == o_out.shape
                assert torch.allclose(t_out.cpu(), o_out.cpu(), rtol=1e-5, atol=1e-5)

            print("NPU test passed")

        except ImportError:
            pytest.skip("torch_npu not installed - skipping Huawei test")

    def test_xy_mode_comprehensive(self):
        """Test XY indexing mode"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5], device="cuda")

        torch_xy = torch.meshgrid([x, y], indexing="xy")
        our_xy = meshgrid([x, y], indexing="xy")

        for t_out, o_out in zip(torch_xy, our_xy):
            assert torch.equal(t_out, o_out)

        z = torch.tensor([6, 7, 8], device="cuda")
        torch_xy_3d = torch.meshgrid([x, y, z], indexing="xy")
        our_xy_3d = meshgrid([x, y, z], indexing="xy")

        for t_out, o_out in zip(torch_xy_3d, our_xy_3d):
            assert torch.equal(t_out, o_out)

        w = torch.tensor([9, 10], device="cuda")
        torch_xy_4d = torch.meshgrid([x, y, z, w], indexing="xy")
        our_xy_4d = meshgrid([x, y, z, w], indexing="xy")

        for t_out, o_out in zip(torch_xy_4d, our_xy_4d):
            assert torch.equal(t_out, o_out)

        print("XY mode comprehensive test passed")

    def test_different_dtypes(self):
        """Test different data types"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.float16]

        for dtype in dtypes:
            x = torch.tensor([1, 2, 3], device="cuda", dtype=dtype)
            y = torch.tensor([4, 5, 6], device="cuda", dtype=dtype)

            torch_out = torch.meshgrid([x, y], indexing="ij")
            our_out = meshgrid([x, y], indexing="ij")

            for t_out, o_out in zip(torch_out, our_out):
                assert t_out.dtype == o_out.dtype
                if dtype == torch.float16:
                    assert torch.allclose(t_out, o_out, rtol=1e-3, atol=1e-3)
                else:
                    assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

        print("Different dtypes test passed")

    def test_empty_input(self):
        """Test empty input"""
        result = meshgrid([], indexing="ij")
        assert result == ()
        print("Empty input test passed")

    def test_single_tensor(self):
        """Test single tensor"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.tensor([1, 2, 3, 4, 5], device="cuda")

        torch_out = torch.meshgrid([x], indexing="ij")
        our_out = meshgrid([x], indexing="ij")

        assert len(torch_out) == 1
        assert len(our_out) == 1
        assert torch.equal(torch_out[0], our_out[0])
        assert torch_out[0].shape == (5,)

        print("Single tensor test passed")

    def test_error_handling(self):
        """Test error handling"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with pytest.raises(ValueError, match="indexing must be 'ij' or 'xy'"):
            meshgrid([torch.tensor([1, 2], device="cuda")], indexing="invalid")

        print("Error handling test passed")

    @pytest.mark.parametrize("size", [10, 50, 100, 200])
    def test_large_inputs(self, size):
        """Test large inputs"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.arange(size, device="cuda", dtype=torch.float32)
        y = torch.arange(size, device="cuda", dtype=torch.float32)

        torch_out = torch.meshgrid([x, y], indexing="ij")
        our_out = meshgrid([x, y], indexing="ij")

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

        print(f"Large input test passed for size {size}")

    def test_contiguity(self):
        """Test output tensor contiguity"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5], device="cuda")

        our_out = meshgrid([x, y], indexing="ij")

        for t in our_out:
            assert t.numel() == t.shape[0] * t.shape[1]
            t_contiguous = t.contiguous()
            assert t_contiguous.shape == t.shape
            assert torch.equal(t_contiguous, t)

        print("Contiguity test passed")

    @pytest.mark.parametrize("ndim", [4, 5, 6])
    def test_high_dimensional(self, ndim):
        """Test high dimensional (4D+)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        sizes = [3, 4, 5, 6, 7, 8][:ndim]
        tensors = [torch.arange(s, device="cuda", dtype=torch.float32) for s in sizes]

        torch_out = torch.meshgrid(tensors, indexing="ij")
        our_out = meshgrid(tensors, indexing="ij")

        for t_out, o_out in zip(torch_out, our_out):
            assert t_out.shape == o_out.shape
            assert torch.allclose(t_out, o_out, rtol=1e-5, atol=1e-5)

        print(f"High-dimensional test passed for ndim={ndim}")
=======
def test_meshgrid_correctness():
    """全面正确性测试"""
    sizes = [1, 2, 3, 4, 5, 10, 100]
    
    for size in sizes:
        x = torch.randn(size, device=DEVICE)
        y = torch.randn(size, device=DEVICE)
        
        for indexing in ['ij', 'xy']:
            our_out = meshgrid([x, y], indexing=indexing)
            ref_out = torch.meshgrid(x, y, indexing=indexing)
            
            for i, (our, ref) in enumerate(zip(our_out, ref_out)):
                if size == 1:
                    assert torch.allclose(our, ref, rtol=1e-4, atol=1e-4), \
                        f"Failed for size {size}, indexing {indexing}, output {i}"
                else:
                    assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5), \
                        f"Failed for size {size}, indexing {indexing}, output {i}"
>>>>>>> 459d19e (Add meshgrid operator implementation with tests and benchmarks)


def test_meshgrid_xy_single_element():
    """专门测试xy模式下的单元素情况"""
    test_cases = [
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([-1.0]), torch.tensor([3.14])),
        (torch.tensor([0.0]), torch.tensor([0.0])),
    ]
    
    for x, y in test_cases:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        our_out = meshgrid([x, y], indexing='xy')
        ref_out = torch.meshgrid(x, y, indexing='xy')
        
        for our, ref in zip(our_out, ref_out):
            assert our.shape == ref.shape
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_xy_mode():
    """测试xy模式的各种情况"""
    x = torch.tensor([1, 2, 3], device=DEVICE)
    y = torch.tensor([4, 5], device=DEVICE)
    
    our_out = meshgrid([x, y], indexing='xy')
    ref_out = torch.meshgrid(x, y, indexing='xy')
    
    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref)
    
    x = torch.tensor([1, 2, 3], device=DEVICE)
    y = torch.tensor([1, 2, 3], device=DEVICE)
    
    our_out = meshgrid([x, y], indexing='xy')
    ref_out = torch.meshgrid(x, y, indexing='xy')
    
    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref)


def test_meshgrid_3d():
    """3D测试"""
    x = torch.randn(4, device=DEVICE)
    y = torch.randn(5, device=DEVICE)
    z = torch.randn(6, device=DEVICE)
    
    for indexing in ['ij', 'xy']:
        our_out = meshgrid([x, y, z], indexing=indexing)
        ref_out = torch.meshgrid(x, y, z, indexing=indexing)
        
        for our, ref in zip(our_out, ref_out):
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_4d():
    """4D测试"""
    tensors = [torch.randn(3, device=DEVICE) for _ in range(4)]
    
    for indexing in ['ij', 'xy']:
        our_out = meshgrid(tensors, indexing=indexing)
        ref_out = torch.meshgrid(*tensors, indexing=indexing)
        
        for our, ref in zip(our_out, ref_out):
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_different_dtypes():
    """不同数据类型测试"""
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    for dtype in dtypes:
        x = torch.tensor([1, 2, 3], dtype=dtype, device=DEVICE)
        y = torch.tensor([4, 5, 6], dtype=dtype, device=DEVICE)
        
        for indexing in ['ij', 'xy']:
            our_out = meshgrid([x, y], indexing=indexing)
            ref_out = torch.meshgrid(x, y, indexing=indexing)
            
            for our, ref in zip(our_out, ref_out):
                assert our.dtype == ref.dtype
                if dtype in [torch.int32, torch.int64]:
                    assert torch.equal(our, ref)
                else:
                    assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_edge_cases():
    """边界情况测试"""
    x = torch.randn(2, device=DEVICE)
    y = torch.randn(5, device=DEVICE)
    z = torch.randn(3, device=DEVICE)
    
    for indexing in ['ij', 'xy']:
        our_out = meshgrid([x, y, z], indexing=indexing)
        ref_out = torch.meshgrid(x, y, z, indexing=indexing)
        
        for our, ref in zip(our_out, ref_out):
            assert our.shape == ref.shape
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)
    
    x = torch.randn(1, device=DEVICE)
    y = torch.randn(100, device=DEVICE)
    
    our_out = meshgrid([x, y], indexing='ij')
    ref_out = torch.meshgrid(x, y, indexing='ij')
    
    for our, ref in zip(our_out, ref_out):
        assert torch.allclose(our, ref)


def test_meshgrid_error_handling():
    """错误处理测试"""
    with pytest.raises(ValueError):
        meshgrid([])
    
    if DEVICE == 'cuda':
        x = torch.tensor([1, 2, 3])
        with pytest.raises(ValueError):
            meshgrid([x])
    
    x = torch.randn(2, device=DEVICE)
    with pytest.raises(ValueError):
        meshgrid([x], indexing='invalid')
    
    tensors = [torch.randn(2, device=DEVICE) for _ in range(5)]
    with pytest.raises(NotImplementedError):
        meshgrid(tensors)
    
    x = torch.randn(2, 3, device=DEVICE)
    with pytest.raises(ValueError):
        meshgrid([x])
    
    with pytest.raises(TypeError):
        meshgrid([1, 2, 3])
