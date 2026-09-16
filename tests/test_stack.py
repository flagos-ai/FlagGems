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

if flag_gems.vendor_name == "cambricon":
    CAMBRICON_STACK_SHAPES = [
        [
            (8, 8, 128),
            (8, 8, 128),
            (8, 8, 128),
        ],
        [
            (32, 64, 128, 8),
            (32, 64, 128, 8),
            (32, 64, 128, 8),
            (32, 64, 128, 8),
        ],
    ]

    STACK_SHAPES_TEST = utils.STACK_SHAPES + CAMBRICON_STACK_SHAPES
else:
    STACK_SHAPES_TEST = utils.STACK_SHAPES


@pytest.mark.stack
@pytest.mark.parametrize("shape", STACK_SHAPES_TEST)
@pytest.mark.parametrize("dim", utils.STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_stack(shape, dim, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch.stack(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.stack(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)
