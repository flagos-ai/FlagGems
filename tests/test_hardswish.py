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


@pytest.mark.hardswish_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardswish_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.hardswish_(ref_inp)
    res_out = flag_gems.hardswish_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardswish
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardswish(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3.0
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.hardswish(ref_inp)
    res_out = flag_gems.hardswish(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardswish_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardswish_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3.0
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.hardswish.out(ref_inp, out=ref_out)

    out = torch.empty_like(inp)
    res_out = flag_gems.hardswish_out(inp, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.hardswish
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardswish_special_values(dtype):
    # Cover boundary values: +/-inf, +/-0, nan and the piecewise knots at +/-3.
    # Compare against PyTorch so whatever aten produces for these (e.g. NaN from
    # -inf * 0 in the mid region) is matched exactly.
    values = [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.0,
        -3.0,
        3.0,
        -6.0,
        6.0,
    ]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.hardswish(ref_inp)
    res_out = flag_gems.hardswish(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
