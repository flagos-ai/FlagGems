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

from . import accuracy_utils as utils


@pytest.mark.functional_assert_async
def test_functional_assert_async_pass():
    """Test that assertion passes when tensor is non-zero"""
    inp = torch.tensor([1], dtype=torch.int32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_dep = utils.to_reference(dep_token)

    # Reference: PyTorch's CPU implementation
    ref_out = torch.ops.aten._functional_assert_async.msg(
        ref_inp, "assertion failed", ref_dep
    )

    # Test: FlagGems implementation
    with flag_gems.use_gems():
        res_out = torch.ops.aten._functional_assert_async.msg(
            inp, "assertion failed", dep_token
        )

    # Both should return empty tensors
    assert ref_out.numel() == 0
    assert res_out.numel() == 0
    assert res_out.dtype == dep_token.dtype
    assert res_out.device == dep_token.device


@pytest.mark.functional_assert_async
def test_functional_assert_async_fail():
    """Test that assertion fails when tensor is zero"""
    inp = torch.tensor([0], dtype=torch.int32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    # This should trigger device assertion and potentially hang or error
    # Skipping actual execution to avoid test hanging
    with flag_gems.use_gems():
        with pytest.raises(Exception):
            # Device assertion failures may manifest as runtime errors
            torch.ops.aten._functional_assert_async.msg(
                inp, "assertion should fail", dep_token
            )


@pytest.mark.functional_assert_async
def test_functional_assert_async_float():
    """Test with float dtype"""
    inp = torch.tensor([1.0], dtype=torch.float32, device=flag_gems.device)
    dep_token = torch.empty(0, dtype=torch.float32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_dep = utils.to_reference(dep_token)

    ref_out = torch.ops.aten._functional_assert_async.msg(
        ref_inp, "assertion failed", ref_dep
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._functional_assert_async.msg(
            inp, "assertion failed", dep_token
        )

    assert ref_out.numel() == 0
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
