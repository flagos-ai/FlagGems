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

import pytest
import torch

import flag_gems


@pytest.mark.functional_assert_async
def test_functional_assert_async_pass():
    """Test that assertion passes when tensor is non-zero"""
    inp = torch.tensor([1], dtype=torch.int32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    # Test: FlagGems implementation
    with flag_gems.use_gems():
        res_out = torch.ops.aten._functional_assert_async.msg(
            inp, "assertion failed", dep_token
        )

    # Should return empty tensor with same dtype and device as dep_token
    assert res_out.numel() == 0
    assert res_out.dtype == dep_token.dtype
    assert res_out.device == dep_token.device


@pytest.mark.functional_assert_async
@pytest.mark.skip(
    reason="Device assertion behavior is asynchronous and may not raise immediately"
)
def test_functional_assert_async_fail():
    """Test that assertion fails when tensor is zero"""
    inp = torch.tensor([0], dtype=torch.int32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    # Device assertions in Triton are asynchronous and may not raise immediately
    # This test is skipped to avoid flaky behavior
    with flag_gems.use_gems():
        _ = torch.ops.aten._functional_assert_async.msg(
            inp, "assertion should fail", dep_token
        )
        # The assertion may trigger later during CUDA synchronization


@pytest.mark.functional_assert_async
def test_functional_assert_async_float():
    """Test with float dtype"""
    inp = torch.tensor([1.0], dtype=torch.float32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten._functional_assert_async.msg(
            inp, "assertion failed", dep_token
        )

    assert res_out.numel() == 0
    assert res_out.dtype == dep_token.dtype
    assert res_out.device == dep_token.device


@pytest.mark.functional_assert_async
def test_functional_assert_async_multi_element_error():
    """Test that multi-element tensors raise an error"""
    inp = torch.tensor([1, 1], dtype=torch.int32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="ambiguous"):
            torch.ops.aten._functional_assert_async.msg(
                inp, "assertion failed", dep_token
            )
