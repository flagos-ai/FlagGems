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

HYPOT_SHAPES = [
    ((2, 3), (2, 3)),
    ((2, 3), (1, 3)),
    ((2, 3), (2, 1)),
    ((2, 3), (1, 1)),
    ((128, 256), (128, 256)),
    ((128, 256), (1, 256)),
    ((128, 256), (128, 1)),
    ((512, 512), (512, 512)),
    ((512, 512), (1, 512)),
    ((512, 512), (512, 1)),
]


@pytest.mark.hypot_
@pytest.mark.parametrize("self_shape,other_shape", HYPOT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("contig", [True, False])
def test_hypot_(self_shape, other_shape, dtype, contig):
    if contig:
        base_self = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    else:
        src = torch.randn(
            (self_shape[1], self_shape[0]), dtype=dtype, device=flag_gems.device
        )
        base_self = src.permute(1, 0)
    base_other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)

    ref_self = utils.to_reference(base_self, True)
    ref_other = utils.to_reference(base_other, True)
    ref_out = torch.ops.aten.hypot_(ref_self, ref_other)

    act_self = base_self.clone()
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hypot_(act_self, base_other)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(act_self, ref_out, dtype)
