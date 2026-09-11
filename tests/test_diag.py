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


@pytest.mark.diag
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D + utils.UT_SHAPES_2D)
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_diag(shape, diagonal, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randint(0, 0x7FFF, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diag(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.diag(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)
