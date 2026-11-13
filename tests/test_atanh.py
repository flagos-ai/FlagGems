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


@pytest.mark.atanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_atanh(shape, dtype):
    # Generate input in (-1, 1) range for valid atanh values. Integer inputs
    # only admit 0 inside that range, so the {-1, 0, 1} window is used instead
    # to also cover the +-inf and NaN edges that ATen produces outside it;
    # atanh promotes integer inputs to float32, matching ATen.
    if dtype in utils.SMALL_INT_DTYPES:
        inp = utils.randint_in_range(shape, dtype, low=-1, high=2).to(flag_gems.device)
    else:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -0.99, 0.99
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atanh(ref_inp)
    res_out = flag_gems.atanh(inp)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atanh"), equal_nan=True
    )


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.BOOL_TYPES)
def test_atanh__integer_rejects(shape, dtype):
    # ATen raises for in-place atanh on integer/bool inputs; FlagGems mirrors
    # that instead of silently truncating.
    inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.atanh_(inp)
