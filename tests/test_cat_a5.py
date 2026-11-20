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

pytestmark = [
    pytest.mark.cat,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.aten.cat.default


def _inputs(shape, dtype, device):
    if len(shape) == 4:
        batch, rows, left, right = shape
        return (
            [
                torch.randn((batch, rows, width), dtype=dtype, device=device)
                for width in (left, right)
            ],
        ), {"dim": -1}

    return ([torch.randn(n, device=device, dtype=dtype) for n in shape],), {}


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1024, 1024, 2048),
        (1048576, 1048576, 2097152),
        (1, 64, 512, 64),
        (2, 3, 17, 9),
    ],
)
def test_cat_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.cat(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
