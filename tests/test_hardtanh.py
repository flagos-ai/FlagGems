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

HARDTANH_MIN_MAX = [(-1.0, 1.0), (-0.5, 0.5), (0.0, 6.0), (-2.0, 0.5)]


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.hardtanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("min_max", HARDTANH_MIN_MAX)
def test_hardtanh_explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.hardtanh(ref_inp, min_val, max_val)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh(res_inp, min_val, max_val)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardtanh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.hardtanh.out(ref_inp, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh.out(inp, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.hardtanh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("min_max", HARDTANH_MIN_MAX)
def test_hardtanh_out_explicit(shape, dtype, min_max):
    min_val, max_val = min_max
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.hardtanh.out(ref_inp, min_val, max_val, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh.out(inp, min_val, max_val, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_special_values(dtype):
    # Cover boundary values: +/-inf, +/-0, nan. hardtanh must propagate NaN
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
    res_inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.hardtanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.hardtanh_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_out_special_values(dtype):
    # Same boundary values as test_hardtanh_special_values, but exercising the
    # hardtanh.out variant: NaN must propagate and the infinities clamp to
    # [min_val, max_val].
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
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.hardtanh.out(ref_inp, out=ref_out)

    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh.out(inp, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
