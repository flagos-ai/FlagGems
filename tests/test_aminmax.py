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
    DIMS_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
    KEEPDIM_DIMS_SHAPE = [(True, DIMS_LIST[0], utils.REDUCTION_SHAPES[0])]
else:
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    KEEPDIM_DIMS_SHAPE = list(
        zip([True, False] * 2, DIMS_LIST, utils.REDUCTION_SHAPES + [(7, 4, 11, 1)])
    )


@pytest.mark.aminmax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_aminmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # torch.aminmax only supports single dim, use torch.amin/amax for multi-dim
    if isinstance(dim, list):
        ref_min = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
        ref_max = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    else:
        ref_min, ref_max = torch.aminmax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        if isinstance(dim, list):
            res_min = torch.amin(inp, dim=dim, keepdim=keepdim)
            res_max = torch.amax(inp, dim=dim, keepdim=keepdim)
        else:
            res_min, res_max = torch.aminmax(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.aminmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_aminmax_no_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch.aminmax(ref_inp)
    with flag_gems.use_gems():
        res_min, res_max = torch.aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)
