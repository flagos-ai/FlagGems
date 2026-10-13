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
