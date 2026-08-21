# Copyright 2026, The FlagOS Contributors.
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

# fp64 is not supported on every platform (e.g. ascend, iluvatar).
_GAMMAINCC_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    _GAMMAINCC_DTYPES.append(torch.float64)

# Q(a, x) is only defined for a > 0 and x >= 0; keep both inputs positive so
# the reference kernel stays on its converging path.
_LOW = 0.1
_SPAN = 10


def _make_inputs(shape, dtype):
    a = torch.rand(shape, dtype=dtype, device=flag_gems.device) * _SPAN + _LOW
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * _SPAN + _LOW
    return a, x


@pytest.mark.special_gammaincc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _GAMMAINCC_DTYPES)
def test_special_gammaincc(shape, dtype, caplog):
    a, x = _make_inputs(shape, dtype)
    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)

    ref_out = torch.ops.aten.special_gammaincc(ref_a, ref_x)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_gammaincc"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_gammaincc(a, x)

    assert "GEMS SPECIAL_GAMMAINCC" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)
    # The functional form must leave its inputs untouched.
    utils.gems_assert_close(a, ref_a, dtype)
    utils.gems_assert_close(x, ref_x, dtype)


@pytest.mark.special_gammaincc_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _GAMMAINCC_DTYPES)
def test_special_gammaincc_out(shape, dtype, caplog):
    a, x = _make_inputs(shape, dtype)
    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)

    ref_buf = torch.empty(shape, dtype=ref_a.dtype, device=ref_a.device)
    ref_out = torch.ops.aten.special_gammaincc.out(ref_a, ref_x, out=ref_buf)

    res_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_gammaincc"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_gammaincc.out(a, x, out=res_buf)

    assert "GEMS SPECIAL_GAMMAINCC_OUT" in caplog.text
    # The out= form must return the very buffer it was handed.
    assert res_out.data_ptr() == res_buf.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_gammaincc
@pytest.mark.parametrize("dtype", _GAMMAINCC_DTYPES)
def test_special_gammaincc_boundary(dtype):
    """Q(a, 0) = 1 for every a > 0, and Q is monotonically decreasing in x."""
    a = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0], dtype=dtype, device=flag_gems.device)
    zeros = torch.zeros_like(a)
    ref_a = utils.to_reference(a, True)
    ref_zeros = utils.to_reference(zeros, True)

    ref_out = torch.ops.aten.special_gammaincc(ref_a, ref_zeros)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_gammaincc(a, zeros)
    utils.gems_assert_close(res_out, ref_out, dtype)

    large = torch.full_like(a, 40.0)
    ref_large = utils.to_reference(large, True)
    ref_tail = torch.ops.aten.special_gammaincc(ref_a, ref_large)
    with flag_gems.use_gems():
        res_tail = torch.ops.aten.special_gammaincc(a, large)
    utils.gems_assert_close(res_tail, ref_tail, dtype)


@pytest.mark.special_gammaincc
@pytest.mark.parametrize("shape", [(64,)])
def test_special_gammaincc_rejects_half(shape):
    """The underlying kernel has no Half/BFloat16 path; the error must surface."""
    for dtype in (torch.float16, torch.bfloat16):
        a = torch.rand(shape, dtype=dtype, device=flag_gems.device) + _LOW
        x = torch.rand(shape, dtype=dtype, device=flag_gems.device) + _LOW
        with flag_gems.use_gems():
            with pytest.raises(RuntimeError):
                torch.ops.aten.special_gammaincc(a, x)
