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
    ZERO_SHAPES = [(2, 3)]
else:
    ZERO_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.zero
@pytest.mark.parametrize("shape", ZERO_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_zero(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    act_x = x.clone()

    ref_out = torch.ops.aten.zero(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.zero(act_x)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.zero_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
def test_zero_(shape, dtype):
    out = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    ref_out = utils.to_reference(out)
    ref_out.zero_()

    with flag_gems.use_gems():
        out.zero_()

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.zero_out
@pytest.mark.parametrize("shape", ZERO_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_zero_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    act_x = x.clone()

    ref_out = torch.ops.aten.zero.out(ref_x, out=ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.zero.out(act_x, out=act_x)

    utils.gems_assert_close(act_out, ref_out, dtype)
