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

HARDTANH_MIN_MAX = [(-1.0, 1.0), (0.0, 1.0), (-0.5, 0.5), (-2.0, 0.5)]


@pytest.mark.hardtanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3.0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.hardtanh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh_(inp)

    assert res_out is inp
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardtanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("min_max", HARDTANH_MIN_MAX)
def test_hardtanh__explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3.0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.hardtanh_(ref_inp, min_val, max_val)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh_(inp, min_val, max_val)

    assert res_out is inp
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardtanh_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh__special_values(dtype):
    # Cover boundary values: +/-inf, +/-0, nan. hardtanh_ must propagate NaN
    # like PyTorch and clamp the infinities to [min_val, max_val].
    values = [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.0,
        -1.0,
        1.0,
        2.0,
        -2.0,
    ]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.hardtanh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh_(inp)

    assert res_out is inp
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
