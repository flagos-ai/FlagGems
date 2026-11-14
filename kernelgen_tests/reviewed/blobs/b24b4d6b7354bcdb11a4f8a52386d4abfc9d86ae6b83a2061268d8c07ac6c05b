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

_ATEN_OP = torch.ops.aten._foreach_max


def _gems_op(*args, **kwargs):
    return flag_gems.foreach_max(*args, **kwargs)


@pytest.mark.foreach_max
def test_foreach_max():
    values = [2**62, 3, -5]
    tensors = [
        torch.tensor(values, dtype=torch.int64, device=flag_gems.device),
        torch.tensor(
            list(reversed(values)), dtype=torch.int64, device=flag_gems.device
        ),
    ]
    refs = [utils.to_reference(tensor) for tensor in tensors]
    expected = _ATEN_OP(refs)
    actual = _gems_op(tensors)

    assert len(actual) == len(expected)
    for result, want in zip(actual, expected):
        assert result.shape == torch.Size([])
        assert result.dtype == want.dtype
        utils.gems_assert_equal(result, want)


@pytest.mark.foreach_max
def test_foreach_max_noncontiguous():
    tensors = [
        torch.arange(
            24, dtype=torch.float32, device=flag_gems.device
        ).reshape(4, 6).transpose(0, 1)
    ]
    refs = [utils.to_reference(tensor, True) for tensor in tensors]
    expected = _ATEN_OP(refs)
    actual = _gems_op(tensors)

    assert len(actual) == len(tensors)
    assert actual[0].dtype == tensors[0].dtype
    assert actual[0].shape == torch.Size([])
    assert actual[0].device == tensors[0].device
    utils.gems_assert_close(actual[0], expected[0], tensors[0].dtype)
