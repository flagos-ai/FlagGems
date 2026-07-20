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
    UPSAMPLE_NEAREST1D_SHAPES = [(4, 8, 64)]
else:
    UPSAMPLE_NEAREST1D_SHAPES = [(2, 3, 16), (4, 8, 64), (8, 16, 256)]


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", UPSAMPLE_NEAREST1D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("factor", [2, 3])
def test_accuracy__upsample_nearest_exact1d(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_size = [shape[-1] * factor]

    ref_out = torch.ops.aten._upsample_nearest_exact1d(ref_x, out_size, None)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._upsample_nearest_exact1d(x, out_size, None)

    utils.gems_assert_close(res_out, ref_out, dtype)
