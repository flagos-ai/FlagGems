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
from .conftest import QUICK_MODE

# Quick mode uses minimal shape; full mode covers various ranks and sizes
RENORM_SHAPES = (
    [(2, 8)]
    if QUICK_MODE
    else [
        (10, 20),
        (20, 10),
        (5, 32, 20),
        (4, 8, 16),
        (2, 4, 8, 16),
    ]
)


@pytest.mark.renorm
@pytest.mark.parametrize("shape", RENORM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_renorm(shape, dtype, p, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    maxnorm = 1.0

    ref_inp = utils.to_reference(inp)

    ref_out = torch.renorm(ref_inp, p, dim, maxnorm)
    with flag_gems.use_gems():
        res_out = torch.renorm(inp, p, dim, maxnorm)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.renorm_
@pytest.mark.parametrize("shape", RENORM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_renorm_(shape, dtype, p, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    maxnorm = 1.0

    ref_out = ref_inp.renorm_(p, dim, maxnorm)
    with flag_gems.use_gems():
        res_out = inp.renorm_(p, dim, maxnorm)

    utils.gems_assert_close(res_out, ref_out, dtype)
