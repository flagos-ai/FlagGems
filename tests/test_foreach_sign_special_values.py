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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "inplace",
    [
        pytest.param(False, marks=pytest.mark.foreach_sign),
        pytest.param(True, marks=pytest.mark.foreach_sign_),
    ],
)
def test_sign_special_values(dtype, inplace):
    values = torch.tensor(
        [float("-inf"), -3.0, -0.0, 0.0, 2.0, float("inf"), float("nan")], dtype=dtype
    )
    expected = torch.sign(values)
    source = values.to(flag_gems.device)
    if inplace:
        flag_gems._foreach_sign_([source])
        result = source
    else:
        result = flag_gems._foreach_sign([source])[0]
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
