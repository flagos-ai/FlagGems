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

RENORM_SHAPES = (
    [(2, 32), (4, 128)]
    if QUICK_MODE
    else [(1, 2), (4, 128), (16, 256), (4, 1024), (1, 4096)]
)
RENORM_DIM_LIST = [1] if QUICK_MODE else [0, 1, -1]
RENORM_P_LIST = [2.0] if QUICK_MODE else [1.0, 2.0, 3.5]
RENORM_MAXNORM_LIST = [1.0] if QUICK_MODE else [0.5, 1.0, 2.0]


@pytest.mark.renorm_
@pytest.mark.parametrize("shape", RENORM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", RENORM_DIM_LIST)
@pytest.mark.parametrize("p", RENORM_P_LIST)
@pytest.mark.parametrize("maxnorm", RENORM_MAXNORM_LIST)
def test_renorm_(shape, dtype, dim, p, maxnorm):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.renorm_(ref_inp, p, dim, maxnorm)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.renorm_(inp, p, dim, maxnorm)

    # renorm_ is in-place, so both tensors should be modified
    # Compare the modified tensors
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
