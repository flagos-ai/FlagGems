# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for _cslt_sparse_mm operator.

NOTE: This operator is NOT implemented in FlagGems because the cuSPARSELt
compressed format is vendor-specific and cannot be reimplemented portably
in Triton. These tests verify that the native PyTorch implementation works
correctly, but they do NOT test a FlagGems implementation.

The operator is deliberately excluded from FlagGems registration to avoid
infinite recursion issues.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.cslt_sparse_mm
@pytest.mark.parametrize("shape", [(32, 64, 16), (64, 128, 32), (128, 256, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_cslt_sparse_mm(shape, dtype):
    """Test _cslt_sparse_mm basic functionality (native PyTorch only)."""
    M, K, N = shape

    # Create a sparse matrix with some sparsity pattern
    A = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    mask = torch.rand_like(A) > 0.5
    A_sparse = A * mask

    # Compress using cuSPARSELt
    compressed_A = torch._cslt_compress(A_sparse)

    # Create dense B matrix
    B = torch.randn(K, N, dtype=dtype, device=flag_gems.device)

    # Reference output using PyTorch native (on CPU for comparison)
    ref_compressed_A = utils.to_reference(compressed_A)
    ref_B = utils.to_reference(B)
    ref_out = torch._cslt_sparse_mm(ref_compressed_A, ref_B)

    # Native CUDA implementation (FlagGems does not override this operator)
    res_out = torch._cslt_sparse_mm(compressed_A, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cslt_sparse_mm
@pytest.mark.parametrize("shape", [(32, 64, 16), (64, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_cslt_sparse_mm_with_bias(shape, dtype):
    """Test _cslt_sparse_mm with bias parameter (native PyTorch only)."""
    M, K, N = shape

    # Create sparse matrix and compress
    A = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    mask = torch.rand_like(A) > 0.5
    A_sparse = A * mask
    compressed_A = torch._cslt_compress(A_sparse)

    # Create dense B matrix and bias
    B = torch.randn(K, N, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(N, dtype=dtype, device=flag_gems.device)

    # Reference output
    ref_compressed_A = utils.to_reference(compressed_A)
    ref_B = utils.to_reference(B)
    ref_bias = utils.to_reference(bias)
    ref_out = torch._cslt_sparse_mm(ref_compressed_A, ref_B, bias=ref_bias)

    # Native CUDA implementation
    res_out = torch._cslt_sparse_mm(compressed_A, B, bias=bias)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cslt_sparse_mm
@pytest.mark.parametrize("shape", [(32, 64, 16)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_accuracy_cslt_sparse_mm_with_alpha(shape, dtype):
    """Test _cslt_sparse_mm with alpha parameter (native PyTorch only)."""
    M, K, N = shape

    # Create sparse matrix and compress
    A = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    mask = torch.rand_like(A) > 0.5
    A_sparse = A * mask
    compressed_A = torch._cslt_compress(A_sparse)

    # Create dense B matrix and alpha
    B = torch.randn(K, N, dtype=dtype, device=flag_gems.device)
    alpha = torch.tensor(2.0, dtype=dtype, device=flag_gems.device)

    # Reference output
    ref_compressed_A = utils.to_reference(compressed_A)
    ref_B = utils.to_reference(B)
    ref_alpha = utils.to_reference(alpha)
    ref_out = torch._cslt_sparse_mm(ref_compressed_A, ref_B, alpha=ref_alpha)

    # Native CUDA implementation
    res_out = torch._cslt_sparse_mm(compressed_A, B, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)
