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


@pytest.mark.asin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.asin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asin(res_inp)
    ref_out = ref_out.to(res_out.dtype)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.asin_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asin_(inp)

    ref_out = ref_out.to(res_out.dtype)
    ref_inp = ref_inp.to(inp.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)
