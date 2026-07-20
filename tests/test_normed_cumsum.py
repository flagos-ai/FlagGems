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
    NORMED_CUMSUM_CASES = [
        ((17,), -1),
        ((4, 33), -1),
        ((7, 19), 0),
        ((2, 5, 31), -1),
    ]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NORMED_CUMSUM_CASES = [
        ((17,), -1),
        ((2637,), -1),
        ((4, 33), -1),
        ((32, 1025), -1),
        ((7, 19), 0),
        ((513, 16), 0),
        ((2, 5, 31), -1),
        ((4, 8, 257), -1),
    ]

random.seed(time.time() // 100)


def torch_normed_cumsum(inp, dim=-1):
    ref_inp = utils.to_reference(inp, True)
    return torch.cumsum(ref_inp, dim=dim) / ref_inp.sum(dim=dim, keepdim=True)


@pytest.mark.normed_cumsum
@pytest.mark.parametrize(("shape", "dim"), NORMED_CUMSUM_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_normed_cumsum(shape, dim, dtype):
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1

    ref_out = torch_normed_cumsum(inp, dim=dim)
    res_out = flag_gems.normed_cumsum(inp, dim=dim)

    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=shape[dim],
        atol=1e-3,
    )


@pytest.mark.normed_cumsum
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_normed_cumsum_output_is_normalized(dtype):
    torch.manual_seed(0)
    inp = torch.rand((4, 257), dtype=dtype, device=flag_gems.device) + 0.1

    res_out = flag_gems.normed_cumsum(inp)

    expected_last = utils.to_reference(torch.ones_like(res_out[..., -1]))
    utils.gems_assert_close(res_out[..., -1], expected_last, dtype, atol=1e-3)
