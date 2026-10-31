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


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("max", marks=pytest.mark.foreach_max),
        pytest.param("norm", marks=pytest.mark.foreach_norm_scalar),
    ],
)
def test_large_foreach_reduction_on_device(name):
    values = [
        torch.linspace(-1, 1, 131072, dtype=torch.float32, device=flag_gems.device)
        for _ in range(2)
    ]
    reference = [v.cpu() for v in values]
    if name == "max":
        actual = flag_gems._foreach_max(values)
        expected = torch._foreach_max(reference)
    else:
        actual = flag_gems._foreach_norm(values, 2)
        expected = torch._foreach_norm(reference, 2)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-5)
