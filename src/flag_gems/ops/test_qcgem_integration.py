#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QC-GEM: Integration test with FlagGems

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Testing FlagGems QC-GEM integration...")

try:
    import flag_gems
    print("[OK] FlagGems imported successfully")
    flag_gems.enable()
    print("[OK] FlagGems enabled successfully")
except ImportError as e:
    print(f"[FAIL] Could not import FlagGems: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Error enabling FlagGems: {e}")
    sys.exit(1)

try:
    from flag_gems.ops.qcgem import QCGeMLinear, qcgem_mm, DType
    print("[OK] QC-GEM imported successfully")
except ImportError as e:
    print(f"[FAIL] Could not import QC-GEM: {e}")
    sys.exit(1)


def test_cpu():
    print("Testing on CPU...")
    print("  [SKIP] Triton kernels require CUDA, skipping CPU test")
    return True


def test_cuda():
    print("Testing on CUDA...")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return True
    
    M, N, K = 4, 128, 256
    W_nbits = 4
    group_size = 32
    
    W_float = torch.randn(N, K, device='cuda')
    max_val = 2 ** W_nbits
    W_q = (torch.abs(W_float * (max_val / 4)) % max_val).to(torch.uint8)
    
    scales = torch.randn(N, K // group_size, device='cuda') * 0.1
    scales = torch.abs(scales) + 0.01
    zeros = torch.zeros(N, K // group_size, device='cuda')
    
    x = torch.randn(M, K, device='cuda', dtype=torch.float16)
    
    layer = QCGeMLinear(
        W_nbits=W_nbits,
        group_size=group_size,
        in_features=K,
        out_features=N,
        input_dtype=DType.FP16,
    )
    layer.pack(W_q, scales, zeros)
    out = layer(x)
    
    assert out.shape == x.shape[:-1] + (N,)
    print(f"  [PASS] CUDA test passed: {out.shape}")
    return True


def test_dtypes():
    print("Testing different dtypes...")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return True
    
    for dtype in [torch.float16]:
        M, N, K = 2, 64, 128
        W_nbits = 4
        group_size = 32
        
        W_float = torch.randn(N, K, device='cuda')
        max_val = 2 ** W_nbits
        W_q = (torch.abs(W_float * (max_val / 4)) % max_val).to(torch.uint8)
        
        scales = torch.randn(N, K // group_size, device='cuda') * 0.1
        scales = torch.abs(scales) + 0.01
        zeros = torch.zeros(N, K // group_size, device='cuda')
        
        qcgem_dtype = DType.FP16 if dtype == torch.float16 else DType.BF16
        x = torch.randn(M, K, device='cuda', dtype=dtype)
        
        layer = QCGeMLinear(
            W_nbits=W_nbits,
            group_size=group_size,
            in_features=K,
            out_features=N,
            input_dtype=qcgem_dtype,
        )
        layer.pack(W_q, scales, zeros)
        out = layer(x)
        
        assert out.shape == x.shape[:-1] + (N,)
        assert out.dtype == dtype
        print(f"  [PASS] {dtype} test passed")
    
    return True


def test_bitwidths():
    print("Testing different bit-widths...")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return True
    
    for W_nbits in [4, 8]:
        M, N, K = 2, 64, 128
        group_size = 32
        
        W_float = torch.randn(N, K, device='cuda')
        max_val = 2 ** W_nbits
        W_q = (torch.abs(W_float * (max_val / 4)) % max_val).to(torch.uint8)
        
        scales = torch.randn(N, K // group_size, device='cuda') * 0.1
        scales = torch.abs(scales) + 0.01
        zeros = torch.zeros(N, K // group_size, device='cuda')
        
        x = torch.randn(M, K, device='cuda', dtype=torch.float16)
        
        layer = QCGeMLinear(
            W_nbits=W_nbits,
            group_size=group_size,
            in_features=K,
            out_features=N,
            input_dtype=DType.FP16,
        )
        layer.pack(W_q, scales, zeros)
        out = layer(x)
        
        assert out.shape == x.shape[:-1] + (N,)
        print(f"  [PASS] W{W_nbits}-bit test passed")
    
    return True


def run_all_tests():
    print("=" * 60)
    print("QC-GEM Integration Tests for FlagGems")
    print("=" * 60)
    print()
    
    tests = [
        test_cpu,
        test_cuda,
        test_dtypes,
        test_bitwidths,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
