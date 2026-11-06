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

import math
import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
else:
    DIM_LIST = [0, 1]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

random.seed(time.time() // 100)


@pytest.mark.index_select
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_select(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_size = inp.size(dim)

    index = torch.randint(
        0, index_size, [math.floor(index_size * 0.8)], device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    res_out = flag_gems.index_select(inp, dim, index)

    utils.gems_assert_equal(res_out, ref_out)


def _expanded_index(n, device):
    return torch.zeros(1, dtype=torch.int64, device=device).expand(n)


def _strided_index(n, device):
    return torch.randint(0, 2, (2 * n,), device=device)[::2]


def _column_index(n, device):
    return torch.randint(0, 2, (n, 3), device=device)[:, 1]


@pytest.mark.index_select
@pytest.mark.parametrize("make_index", [_expanded_index, _strided_index, _column_index])
def test_index_select_non_contiguous_index(make_index):
    inp = torch.randn(2, 2, 257, 5, device=flag_gems.device)
    dim = -4
    index = make_index(124, flag_gems.device)
    assert not index.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    res_out = flag_gems.index_select(inp, dim, index)

    utils.gems_assert_equal(res_out, ref_out)
