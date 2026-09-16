# Copyright 2026, The FlagOS Contributors.
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
from . import conftest as cfg

if cfg.QUICK_MODE:
    # Single tiny shape keeps QUICK_MODE fast; mirrors sibling lift_fresh_copy.
    LIFT_FRESH_SHAPES = [(2, 3)]
else:
    # Small-to-medium 2D shapes cover tail masking and multiple blocks; mirrors
    # the sibling lift_fresh_copy test shapes.
    LIFT_FRESH_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.lift_fresh
@pytest.mark.parametrize("shape", LIFT_FRESH_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_fresh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift_fresh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.lift_fresh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
