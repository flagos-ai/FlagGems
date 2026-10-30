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
Test mul operation with large tensors that would trigger coreDim > 65535.

This test case is designed to verify the fix for the coreDim overflow issue
on Ascend NPU. The issue was that when n_elements is very large, the grid
calculation could result in a coreDim value exceeding the NPU hardware limit
of 65535 (UINT16_MAX).

Background:
- Ascend NPU has a hardware limitation: coreDim (grid dimension) cannot exceed 65535
- For mul_scalar_kernel with default BLOCK_SIZE=1024:
  - coreDim = ceil(n_elements / BLOCK_SIZE)
  - When n_elements > 67,108,864 (65536 * 1024), coreDim would exceed 65535
- The fix ensures grid size is capped at 65535, and the kernel's internal loop
  handles the remaining blocks

Test cases cover:
1. Tensors just below the threshold (should work without adjustment)
2. Tensors just above the threshold (triggers the grid size cap)
3. Tensors matching the sizes from actual training errors (141050 * 1024, etc.)
4. Very large tensors (stress test)
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "n_elements,description",
    [
        # Small tensor - baseline test
        (10240, "Small tensor (baseline)"),
        # Just below threshold: 65535 * 1024 - 1 = 67,107,839
        (67107839, "Just below coreDim threshold"),
        # Just above threshold: 65536 * 1024 = 67,108,864
        (67108864, "Just above coreDim threshold"),
        # From actual training error logs
        (141050 * 1024, "From device 0-3 error log (144,435,200)"),
        (169351 * 1024, "From device 8-11 error log (173,415,424)"),
        (310400 * 1024, "From device 4-7 error log (317,849,600)"),
        # Very large tensor - stress test
        # Note: This creates a ~4GB tensor (1G * 4 bytes), ensure device has enough memory
        pytest.param(
            1024 * 1024 * 1024,
            "Very large tensor (1G elements, 4GB)",
        ),
    ],
)
@pytest.mark.parametrize("scalar", [2.5, -1.5, 0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mul_large_tensor_scalar(n_elements, description, scalar, dtype):
    """Test mul operation with large tensors that trigger coreDim > 65535."""
    if n_elements == 1024 * 1024 * 1024:
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            pytest.skip("Requires an Ascend NPU with at least 8GB device memory")
        if npu.get_device_properties(0).total_memory < 8 * 1024**3:
            pytest.skip("Requires at least 8GB device memory")

    # Create large tensor
    inp = torch.randn(n_elements, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # Compute reference result
    ref_out = torch.mul(ref_inp, scalar)

    # Compute FlagGems result
    # This should not raise "coreDim can't be greater than UINT16_MAX" error
    res_out = flag_gems.mul(inp, scalar)

    # Verify correctness
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "n_elements",
    [
        67108864,  # 65536 * 1024
        141050 * 1024,
        169351 * 1024,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mul_large_tensor_scalar_dtypes(n_elements, dtype):
    """Test mul operation with large tensors across different dtypes."""
    if dtype == torch.bfloat16 and not flag_gems.runtime.device.support_bf16:
        pytest.skip("bfloat16 not supported on this device")

    inp = torch.randn(n_elements, dtype=dtype, device=flag_gems.device)
    scalar = 2.5
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mul(ref_inp, scalar)
    res_out = flag_gems.mul(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul_
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "n_elements",
    [
        67108864,  # 65536 * 1024
        173415424,  # 169351 * 1024
    ],
)
@pytest.mark.parametrize("scalar", [2.0, -3.5])
def test_mul_large_tensor_scalar_inplace(n_elements, scalar):
    """Test inplace mul operation with large tensors."""
    inp = torch.randn(n_elements, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.mul_(scalar)
    res_out = flag_gems.mul_(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, torch.float32)
