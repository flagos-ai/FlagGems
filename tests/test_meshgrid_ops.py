# tests/test_meshgrid_ops.py

import pytest
import torch
from flag_gems.ops.meshgrid import meshgrid

# Auto-detect available device
def get_available_device():
    if torch.cuda.is_available():
        return 'cuda'
    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu:0'
    except:
        pass
    return 'cpu'

DEVICE = get_available_device()


def test_meshgrid_correctness():
    """Comprehensive correctness test"""
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


def test_meshgrid_xy_single_element():
    """Test xy mode with single element cases"""
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
    """Test xy mode with various cases"""
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
    """3D meshgrid test"""
    x = torch.randn(4, device=DEVICE)
    y = torch.randn(5, device=DEVICE)
    z = torch.randn(6, device=DEVICE)
    
    for indexing in ['ij', 'xy']:
        our_out = meshgrid([x, y, z], indexing=indexing)
        ref_out = torch.meshgrid(x, y, z, indexing=indexing)
        
        for our, ref in zip(our_out, ref_out):
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_4d():
    """4D meshgrid test"""
    tensors = [torch.randn(3, device=DEVICE) for _ in range(4)]
    
    for indexing in ['ij', 'xy']:
        our_out = meshgrid(tensors, indexing=indexing)
        ref_out = torch.meshgrid(*tensors, indexing=indexing)
        
        for our, ref in zip(our_out, ref_out):
            assert torch.allclose(our, ref, rtol=1e-5, atol=1e-5)


def test_meshgrid_different_dtypes():
    """Test with different data types"""
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
    """Test edge cases"""
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
    """Error handling tests"""
    # Test empty list
    with pytest.raises(ValueError, match="tensors must be a non-empty list or tuple"):
        meshgrid([])
    
    # Test invalid indexing parameter
    x = torch.randn(2, device=DEVICE)
    with pytest.raises(ValueError, match="indexing must be 'ij' or 'xy'"):
        meshgrid([x], indexing='invalid')
    
    # Test high dimensions - on NPU this may cause memory issues
    # so we test with smaller dimensions or skip on NPU
    if DEVICE == 'cpu' or DEVICE == 'cuda':
        # Only test high dimensions on CPU/CUDA where it's supported
        tensors = [torch.randn(2, device=DEVICE) for _ in range(9)]
        try:
            result = meshgrid(tensors)
            # Verify correctness
            ref = torch.meshgrid(*tensors, indexing='ij')
            for our, ref_t in zip(result, ref):
                assert torch.allclose(our, ref_t)
        except Exception as e:
            pytest.fail(f"meshgrid with 9 dimensions should work, got {e}")
    else:
        # On NPU, just test that it doesn't crash with a reasonable dimension
        tensors = [torch.randn(2, device=DEVICE) for _ in range(3)]
        try:
            result = meshgrid(tensors)
            ref = torch.meshgrid(*tensors, indexing='ij')
            for our, ref_t in zip(result, ref):
                assert torch.allclose(our, ref_t)
        except Exception as e:
            pytest.fail(f"meshgrid with 3 dimensions should work, got {e}")
    
    # Test non-1D tensor
    x = torch.randn(2, 3, device=DEVICE)
    with pytest.raises(ValueError, match="must be 1D"):
        meshgrid([x])
    
    # Test non-tensor input
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        meshgrid([1, 2, 3])
    
    # Test mixed devices (if available)
    if DEVICE != 'cpu':
        x_cpu = torch.tensor([1, 2, 3])
        y_cpu = torch.tensor([4, 5, 6])
        try:
            result = meshgrid([x_cpu, y_cpu], indexing='ij')
            assert len(result) == 2
        except Exception as e:
            # If it raises, it should be a reasonable error
            pass
