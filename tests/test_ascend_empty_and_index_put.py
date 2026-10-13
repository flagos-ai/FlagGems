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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend kernel regression cases"
)


@pytest.mark.diff
@pytest.mark.parametrize(
    "shape,n", [((0, 7), 1), ((3, 0), 1), ((0, 7), 0), ((3, 2), 4)]
)
def test_diff_empty_dimensions(shape, n):
    x = torch.randn(shape, device=flag_gems.device)
    expected = torch.diff(x, n=n)
    actual = flag_gems.diff(x, n=n)
    assert torch.equal(actual, expected)


@pytest.mark.pad
@pytest.mark.parametrize("shape", [(0, 5), (2, 0)])
def test_constant_pad_empty_input(shape):
    x = torch.empty(shape, device=flag_gems.device)
    expected = torch.nn.functional.pad(x, (2, 3, 1, 1), value=2.5)
    actual = flag_gems.pad(x, (2, 3, 1, 1), value=2.5)
    assert torch.equal(actual, expected)


@pytest.mark.index_put
@pytest.mark.parametrize("shape,positions", [((8, 4, 16), [1, 3, -1]), ((2, 0), [0])])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_put_first_dimension(shape, positions, inplace):
    x = torch.zeros(shape, dtype=torch.bfloat16, device=flag_gems.device)
    indices = torch.tensor(positions, dtype=torch.int64, device=x.device)
    values = torch.ones((len(positions), *shape[1:]), dtype=x.dtype, device=x.device)
    expected = x.clone()
    expected.index_put_((indices,), values)
    actual = x.clone()
    if inplace:
        result = flag_gems.index_put_(actual, (indices,), values)
        assert result is actual
    else:
        result = flag_gems.index_put(actual, (indices,), values)
        assert result is not actual
        assert torch.equal(actual, x)
    assert torch.equal(result, expected)


@pytest.mark.index_put
def test_index_put_uses_tensor_device():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    x = torch.zeros((4, 8), device="npu:0")
    indices = torch.tensor([1, 3], device=x.device)
    values = torch.ones((2, 8), device=x.device)
    expected = x.clone()
    expected.index_put_((indices,), values)
    with torch.npu.device(1):
        actual = x.clone()
        flag_gems.index_put_(actual, (indices,), values)
    assert torch.equal(actual, expected)
