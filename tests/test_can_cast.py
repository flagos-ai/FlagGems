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

from . import test_utils as tu

# Check cast compatibility for dtype pairs; there is no tensor payload.
_CAN_CAST_DTYPES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.complex64,
] + tu.selected_cases([torch.int16, torch.float64, torch.complex32, torch.complex128])

_INVALID_SCALARTYPE_CASES = [
    pytest.param("float32", id="str"),
    pytest.param(None, id="none"),
    pytest.param(3.14, id="float"),
    pytest.param([1, 2], id="list"),
]


def _assert_result(res_out, ref_out):
    # Accept a Python bool or a zero-dimensional bool tensor.
    if isinstance(res_out, torch.Tensor):
        assert res_out.ndim == 0
        assert res_out.dtype == torch.bool
        res_out = res_out.item()
    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.can_cast
@pytest.mark.parametrize("from_dtype", _CAN_CAST_DTYPES)
@pytest.mark.parametrize("to_dtype", _CAN_CAST_DTYPES)
def test_can_cast(from_dtype, to_dtype):
    ref_out = torch.ops.aten.can_cast(from_dtype, to_dtype)
    res_out = flag_gems.can_cast(from_dtype, to_dtype)

    _assert_result(res_out, ref_out)


@pytest.mark.can_cast
@pytest.mark.parametrize("bad_arg", _INVALID_SCALARTYPE_CASES)
def test_can_cast_rejects_non_scalartype_from(bad_arg):
    with pytest.raises(RuntimeError):
        torch.ops.aten.can_cast(bad_arg, torch.float32)
    # A plain-Python candidate naturally raises TypeError/ValueError (or an
    # AttributeError) for the same inputs, which is equally acceptable.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.can_cast(bad_arg, torch.float32)


@pytest.mark.can_cast
@pytest.mark.parametrize("bad_arg", _INVALID_SCALARTYPE_CASES)
def test_can_cast_rejects_non_scalartype_to(bad_arg):
    with pytest.raises(RuntimeError):
        torch.ops.aten.can_cast(torch.float32, bad_arg)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.can_cast(torch.float32, bad_arg)
