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


@pytest.mark.adaptive_max_pool2d
# Includes unbatched input and a pooling window larger than the old 32x32 limit.
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((2, 3, 7, 9), (3, 4)),
        ((1, 2, 8, 8), (1, 1)),
        ((3, 5, 6), (2, 4)),
        ((1, 2, 70, 65), (1, 1)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_max_pool2d(shape, output_size, dtype):
    inp = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    expected, expected_indices = torch.ops.aten.adaptive_max_pool2d(
        utils.to_reference(inp, True), output_size
    )

    with flag_gems.use_gems():
        actual, actual_indices = torch.ops.aten.adaptive_max_pool2d(inp, output_size)

    utils.gems_assert_close(actual, expected, dtype)
    utils.gems_assert_equal(actual_indices, expected_indices)


@pytest.mark.adaptive_max_pool2d
def test_adaptive_max_pool2d_nan_and_tie_indices():
    inp = torch.tensor(
        [[[[float("nan"), float("nan"), 3.0, 3.0]]]],
        device=flag_gems.device,
        dtype=torch.float32,
    )
    expected, expected_indices = torch.ops.aten.adaptive_max_pool2d(
        utils.to_reference(inp), (1, 1)
    )
    with flag_gems.use_gems():
        actual, actual_indices = torch.ops.aten.adaptive_max_pool2d(inp, (1, 1))
    utils.gems_assert_close(actual, expected, torch.float32, equal_nan=True)
    utils.gems_assert_equal(actual_indices, expected_indices)
