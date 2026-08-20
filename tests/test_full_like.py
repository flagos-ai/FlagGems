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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.full_like
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize(
    "fill_value", [3.1415926, 2, False, float("inf"), float("nan")]
)
def test_full_like(shape, dtype, fill_value):
    if isinstance(fill_value, float) and (
        math.isinf(fill_value) or math.isnan(fill_value)
    ):
        if dtype not in utils.ALL_FLOAT_DTYPES:
            # Skipping inf/nan test for non-float dtypes
            return

    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    # without dtype
    ref_out = torch.full_like(ref_inp, fill_value)
    with flag_gems.use_gems():
        res_out = torch.full_like(inp, fill_value)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)

    # with dtype
    ref_out = torch.full_like(ref_inp, fill_value, dtype=dtype)
    with flag_gems.use_gems():
        res_out = torch.full_like(inp, fill_value, dtype=dtype)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
