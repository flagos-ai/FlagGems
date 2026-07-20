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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]

random.seed(time.time() // 100)


@pytest.mark.select_scatter
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_select_scatter(shape, dim, dtype):
    index = random.randint(0, shape[dim] - 1)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    del src_shape[dim]
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flag_gems.use_gems():
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.select_scatter
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_select_scatter_with_self_overlapping_input():
    dim = 0
    index = 1
    inp = torch.randn((1, 4), device=flag_gems.device).broadcast_to((3, 4))
    src = torch.randn((4,), device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flag_gems.use_gems():
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    utils.gems_assert_equal(res_out, ref_out)
