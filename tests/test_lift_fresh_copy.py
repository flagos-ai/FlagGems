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
from . import conftest as cfg

if cfg.QUICK_MODE:
    LIFT_FRESH_COPY_SHAPES = [(2, 3)]
else:
    LIFT_FRESH_COPY_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.lift_fresh_copy
@pytest.mark.parametrize("shape", LIFT_FRESH_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_lift_fresh_copy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift_fresh_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.lift_fresh_copy(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
