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


@pytest.mark.channel_shuffle
@pytest.mark.parametrize(
    "shape_groups", [((1, 4, 2, 2), 2), ((2, 8, 4, 4), 4), ((4, 16, 8, 8), 4)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_channel_shuffle(shape_groups, dtype):
    shape, groups = shape_groups
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_out = torch.ops.aten.channel_shuffle(ref_input, groups)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.channel_shuffle(input_tensor, groups)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
