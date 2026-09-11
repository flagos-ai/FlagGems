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
    REPEAT_SIZES = [(2, 3, 4, 5)]
else:
    REPEAT_SIZES = [(2, 3, 4, 5), (5, 0, 4)]


@pytest.mark.repeat
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("sizes", REPEAT_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_repeat(shape, sizes, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    sizes = utils.unsqueeze_tuple(sizes, inp.ndim)

    ref_out = ref_inp.repeat(*sizes)
    with flag_gems.use_gems():
        res_out = inp.repeat(*sizes)

    utils.gems_assert_close(res_out, ref_out, dtype)
