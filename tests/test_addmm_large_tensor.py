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
Test addmm operation with large tensors that would trigger coreDim > 65535.

This test case is designed to verify the fix for the coreDim overflow issue
on Ascend NPU. The issue was that the 2D grid calculation for addmm could
result in a coreDim value exceeding the NPU hardware limit of 65535.

Background:
- Ascend NPU has a hardware limitation: coreDim (grid dimension) cannot exceed 65535
- For addmm_kernel with BLOCK_M=128, BLOCK_N=128, SPLIT_K=1:
  - grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), SPLIT_K)
  - When M=4096, N=4096, grid[0] = 32 * 32 = 1024 (OK)
  - When M=32768, N=32768, grid[0] = 256 * 256 = 65536 (exceeds limit!)
- The fix flattens the 2D grid into 1D and caps at 65535, with kernel internal
  grid-stride loop handling the remaining tiles

Test cases cover:
1. Matrices just below the threshold
2. Matrices just above the threshold
3. Square matrices (M == N == K) for _addmm_small_* path
4. Non-square large matrices
5. Different alpha/beta values
"""

import importlib

import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K,description",
    [
        # Small baseline
        (512, 512, 512, "Baseline"),
        # Just below threshold: sqrt(65535) ≈ 256, so 256*256 = 65536 (exceeds)
        # Use 255*128 = 32640 (safe), 250*128 = 32000 (safe)
        (200 * 128, 200 * 128, 512, "Below threshold (200*200=40k tiles) ~1.5GB"),
        # Just above threshold: 256*256 = 65536
        (256 * 128, 256 * 128, 512, "Above threshold (256*256=65k tiles) ~4GB"),
        # Larger - use smaller K to control memory
        (300 * 128, 300 * 128, 512, "Large square (300*300=90k tiles) ~6.5GB"),
        # Non-square large matrices - keep total memory low
        (512 * 128, 200 * 128, 512, "Non-square (512*200=102k tiles) ~6.5GB"),
        (200 * 128, 512 * 128, 512, "Non-square (200*512=102k tiles) ~6.5GB"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_large_matrix(M, N, K, description, dtype):
    """Test addmm with large matrices that trigger coreDim > 65535.

    Memory estimate (float32):
    - M=N=256*128=32768, K=512: ~4.2 GB
    - M=N=300*128=38400, K=512: ~6.5 GB
    - M=512*128=65536, N=200*128=25600, K=512: ~6.5 GB
    """
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2)
    res_out = flag_gems.addmm(bias, mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256 * 128, 256 * 128, 512),  # ~4 GB
        (300 * 128, 200 * 128, 512),  # ~4.5 GB, non-square
    ],
)
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_large_matrix_alpha_beta(M, N, K, alpha, beta, dtype):
    """Test addmm with large matrices and different alpha/beta values.

    Memory estimate (float32):
    - M=N=32768, K=512: ~4.2 GB
    - M=38400, N=25600, K=512: ~4.5 GB
    """
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "N",
    [
        128,  # Small - uses _addmm_small_* path
        256,  # Medium
        512,  # Large
        1024,  # Very large
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_small_square_large(N, dtype):
    """Test _addmm_small_* path with large square matrices.

    This tests the special fast path for M==N==K with specific sizes.
    When N > 512, _addmm_small_add and _addmm_small_finish can have
    coreDim issues with M*N / 8192 exceeding 65535.
    """
    if N not in [128, 256, 512, 1024] or N < 128:
        pytest.skip("_addmm_small path only supports specific sizes")

    M = K = N
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    # Test with alpha=beta=1
    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=1.0, beta=1.0)
    res_out = flag_gems.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    # Test with different alpha/beta
    ref_out2 = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=2.0, beta=0.5)
    res_out2 = flag_gems.addmm(bias, mat1, mat2, alpha=2.0, beta=0.5)
    utils.gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=K)


@pytest.mark.addmm_out
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (256 * 128, 256 * 128, 512),  # ~4 GB
        (300 * 128, 200 * 128, 512),  # ~4.5 GB
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_out_large_matrix(M, N, K, dtype):
    """Test addmm_out with large matrices."""
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = utils.to_reference(out, True)

    torch.addmm(ref_bias, ref_mat1, ref_mat2, out=ref_out)
    flag_gems.addmm_out(bias, mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend addmm fallback tuning candidates",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("b_column_major", [False, True])
@pytest.mark.parametrize("vector_bias", [False, True])
def test_addmm_fallback_autotuning(monkeypatch, dtype, b_column_major, vector_bias):
    # A cached best config or timing can hide an invalid autotuning candidate.
    # Exercise fresh tuning, including the NT layout that triggered an aicore
    # exception with the shared MM tuning space. The normal tuner still handles
    # candidates that the compiler rejects before launching them.
    M, N, K = 15, 160, 1024
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn(
        (N, K) if b_column_major else (K, N), dtype=dtype, device=flag_gems.device
    )
    if b_column_major:
        b = b.t()
    bias = torch.randn(
        (N,) if vector_bias else (M, N), dtype=dtype, device=flag_gems.device
    )
    expected = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(a, True),
        utils.to_reference(b, True),
        alpha=0.5,
        beta=2.0,
    )
    op_globals = flag_gems.addmm.__globals__
    tuner = op_globals["addmm_kernel"]
    while not isinstance(tuner, triton.runtime.Autotuner):
        tuner = tuner.fn
    # Bypass LibEntry's dispatch cache and isolate both tuning caches without
    # clearing or writing the user's persistent tuning results.
    libentry_module = importlib.import_module("flag_gems.utils.libentry")
    monkeypatch.setitem(op_globals, "addmm_kernel", tuner)
    monkeypatch.setattr(tuner, "cache", {})
    monkeypatch.setattr(libentry_module.libcache, "get_benchmark", lambda *args: {})
    actual = flag_gems.addmm(bias, a, b, alpha=0.5, beta=2.0)
    utils.gems_assert_close(actual, expected, dtype, reduce_dim=K)
    assert tuner.benchmark_success_count > 0
    assert tuner.benchmark_cache_hit_count == 0
