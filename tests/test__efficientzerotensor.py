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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
for _name in ("_efficientzerotensor", "_efficientzerotensor_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# Create fresh zeros, or overwrite and return the supplied out buffer.
_EFFICIENTZEROTENSOR_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.bool]
    + tu.selected_cases([torch.int16])
    + ([torch.float64] if utils.fp64_is_supported else [])
)

_OUT_RANGE_PARAMS = [
    (dtype, value_range)
    for dtype in _EFFICIENTZEROTENSOR_DTYPES
    for value_range in tu.selected_ranges()
]

_ZERO_SIZE_SHAPES = [(0,), (0, 3), (2, 0, 4), (0, 0)]


def _reference_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


@pytest.mark._efficientzerotensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_zero_fill(shape, dtype):
    ref_out = torch.ops.aten._efficientzerotensor(
        shape, dtype=dtype, device=_reference_device()
    )

    res_out = flag_gems._efficientzerotensor(
        shape, dtype=dtype, device=flag_gems.device
    )

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    # flag_gems.device may carry no index (e.g. 'cuda') while a fresh tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    # The factory returns a fresh, non-view, all-zero tensor.
    assert not res_out._is_view()
    # Values are exactly zero for every dtype, so exact equality is valid.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._efficientzerotensor
@pytest.mark.parametrize("shape", _ZERO_SIZE_SHAPES)
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_zero_size(shape, dtype):
    ref_out = torch.ops.aten._efficientzerotensor(
        shape, dtype=dtype, device=_reference_device()
    )

    res_out = flag_gems._efficientzerotensor(
        shape, dtype=dtype, device=flag_gems.device
    )

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark._efficientzerotensor_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _OUT_RANGE_PARAMS)
def test__efficientzerotensor_out_range(shape, dtype, value_range):
    ref_device = _reference_device()
    garbage = tu.make_input(dtype, shape, value_range)
    ref_buf = garbage.clone().to(ref_device)
    act_buf = garbage.clone()

    ref_out = torch.ops.aten._efficientzerotensor.out(shape, out=ref_buf)

    res_out = flag_gems._efficientzerotensor(shape, out=act_buf)
    assert res_out is act_buf

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device.type == torch.device(flag_gems.device).type
    utils.gems_assert_equal(act_buf, ref_buf)


@pytest.mark._efficientzerotensor_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _EFFICIENTZEROTENSOR_DTYPES)
def test__efficientzerotensor_out_overwrites(shape, dtype):
    ref_device = _reference_device()
    ref_buf = torch.full(shape, 1, dtype=dtype, device=ref_device)
    act_buf = torch.full(shape, 1, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten._efficientzerotensor.out(shape, out=ref_buf)

    res_out = flag_gems._efficientzerotensor(shape, out=act_buf)
    assert res_out is act_buf

    assert res_out.shape == ref_out.shape == torch.Size(shape)
    assert res_out.dtype == ref_out.dtype == dtype
    utils.gems_assert_equal(act_buf, ref_buf)


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_negative_size():
    with pytest.raises(RuntimeError):
        torch.ops.aten._efficientzerotensor(
            (-1,), dtype=torch.float32, device=flag_gems.device
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._efficientzerotensor(
            (-1,), dtype=torch.float32, device=flag_gems.device
        )


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_non_integer_size():
    with pytest.raises(RuntimeError):
        torch.ops.aten._efficientzerotensor(
            (2.5,), dtype=torch.float32, device=flag_gems.device
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._efficientzerotensor(
            (2.5,), dtype=torch.float32, device=flag_gems.device
        )


@pytest.mark._efficientzerotensor
def test__efficientzerotensor_rejects_non_strided_layout():
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._efficientzerotensor(
            (2, 3),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._efficientzerotensor(
            (2, 3),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
