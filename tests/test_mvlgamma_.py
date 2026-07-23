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

P_LIST = [1, 2, 3, 5, 8, 12]
LOW_PRECISION_DTYPES = [
    dtype for dtype in (torch.float16, torch.bfloat16) if dtype in utils.FLOAT_DTYPES
]
FULL_PRECISION_DTYPES = [
    dtype for dtype in utils.FLOAT_DTYPES if dtype not in LOW_PRECISION_DTYPES
]


def _numel(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


LOW_PRECISION_SHAPES = [
    shape for shape in utils.POINTWISE_SHAPES if _numel(shape) <= 1024
]
MVLGAMMA_CASES = [
    (shape, dtype)
    for shape in utils.POINTWISE_SHAPES
    for dtype in FULL_PRECISION_DTYPES
] + [(shape, dtype) for shape in LOW_PRECISION_SHAPES for dtype in LOW_PRECISION_DTYPES]


@pytest.mark.mvlgamma_
@pytest.mark.parametrize("shape,dtype", MVLGAMMA_CASES)
@pytest.mark.parametrize("p", P_LIST)
def test_mvlgamma_(shape, dtype, p):
    torch.manual_seed(42)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    inp = inp + (p - 1) / 2 + 1.0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.mvlgamma_(p)
    with flag_gems.use_gems():
        res_out = inp.mvlgamma_(p)

    # Use relaxed tolerance for float16 due to lgamma precision limitations
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
