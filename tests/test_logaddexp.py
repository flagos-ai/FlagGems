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


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.logaddexp.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.logaddexp.out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
