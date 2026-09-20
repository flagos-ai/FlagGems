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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import conftest as cfg
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
for _name in (
    "_make_per_tensor_quantized_tensor",
    "_make_per_tensor_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# Copy integer storage and attach per-tensor scale/zero-point metadata.
_MAKE_PERTENSOR_INPUT_DTYPES = [torch.int8, torch.uint8, torch.int32]

_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

_WRONG_QUANT_DTYPE = {
    torch.uint8: torch.qint8,
    torch.int8: torch.quint8,
    torch.int32: torch.quint8,
}

_MAKE_PERTENSOR_SCALES = [0.01, 0.5, 1.0]

_MAKE_PERTENSOR_ZERO_POINTS = [-1, 2]

_NON_FINITE_SCALES = [float("nan"), float("inf"), float("-inf")]

_REJECTED_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int16,
    torch.int64,
    torch.bool,
]

_GRID_SHAPES = tu.selected_shapes() + [(0,)]

_SMALL_SHAPES = [(7,), (4, 8), (2, 3, 5)]

_BOUNDARY_PATTERNS = ("min_max_0_1", "constant_min", "constant_max")


def _make_input(shape, dtype, device=None):
    # randint excludes high, so max + 1 includes the dtype maximum.
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min,
        info.max + 1,
        shape,
        dtype=dtype,
        device=flag_gems.device if device is None else device,
    )


def _boundary_input(dtype, pattern):
    info = torch.iinfo(dtype)
    if pattern == "min_max_0_1":
        values = [info.min, info.max, 0, 1]
        if dtype != torch.uint8:
            values.append(-1)
        tensor = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    elif pattern == "constant_min":
        tensor = torch.full((8,), info.min, dtype=dtype, device=flag_gems.device)
    else:
        tensor = torch.full((8,), info.max, dtype=dtype, device=flag_gems.device)
    # All patterns are 1-D; widen to 2-D so the op sees a non-trivial shape.
    return tensor.repeat(4, 1)


def _ref_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _assert_quant_metadata(res_out, ref_out):
    assert res_out.is_quantized
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.qscheme() == ref_out.qscheme()
    res_scale, ref_scale = res_out.q_scale(), ref_out.q_scale()
    if math.isnan(ref_scale):
        assert math.isnan(res_scale)
    else:
        assert res_scale == ref_scale
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.stride() == ref_out.stride()
    tu.assert_result_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__make_per_tensor_quantized_tensor_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out)
    # The input is only read; it must be untouched.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", _SMALL_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor_qparams(shape, dtype, scale, zero_point):
    inp = _make_input(shape, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(
        ref_inp, scale, zero_point
    )
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, scale, zero_point)

    _assert_quant_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("pattern", _BOUNDARY_PATTERNS)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_boundary_values(dtype, pattern):
    inp = _boundary_input(dtype, pattern)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("scale", _NON_FINITE_SCALES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_non_finite_scale(dtype, scale):
    inp = _make_input((4, 8), dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, scale, 0)
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, scale, 0)

    _assert_quant_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_non_contiguous(dtype):
    base = _make_input((16, 8), dtype)
    ref_base = tu.to_reference(base)
    inp = base[:, ::2]
    ref_inp = ref_base[:, ::2]

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.5, -3)
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 0.5, -3)

    _assert_quant_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__make_per_tensor_quantized_tensor_out_value_ranges(shape, dtype, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    # The out buffers start with different qparams so the overwrite performed by
    # the op is observable. The out dtype must already be the derived quantized
    # dtype (the out overload cannot change the out tensor's dtype).
    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=_QUANT_DTYPE[dtype], device=_ref_device(), scale=1.0, zero_point=0
    )
    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor.out(
        ref_inp, 0.5, -3, out=ref_out_buf
    )

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        shape,
        dtype=_QUANT_DTYPE[dtype],
        device=flag_gems.device,
        scale=1.0,
        zero_point=0,
    )
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 0.5, -3, out=act_out_buf)
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
@pytest.mark.parametrize("scale", _MAKE_PERTENSOR_SCALES)
@pytest.mark.parametrize("zero_point", _MAKE_PERTENSOR_ZERO_POINTS)
def test__make_per_tensor_quantized_tensor_out_qparams(dtype, scale, zero_point):
    # Overwrite the initial scale=1 and zero_point=0 qparams.
    inp = _make_input((4, 8), dtype)
    ref_inp = tu.to_reference(inp)

    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        (4, 8), dtype=_QUANT_DTYPE[dtype], device=_ref_device(), scale=1.0, zero_point=0
    )
    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor.out(
        ref_inp, scale, zero_point, out=ref_out_buf
    )

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        (4, 8),
        dtype=_QUANT_DTYPE[dtype],
        device=flag_gems.device,
        scale=1.0,
        zero_point=0,
    )
    res_out = flag_gems._make_per_tensor_quantized_tensor(
        inp, scale, zero_point, out=act_out_buf
    )
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark._make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test__make_per_tensor_quantized_tensor_rejects_non_storage_dtype(dtype):
    inp = torch.tensor([1, 2, 3], dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    with pytest.raises(RuntimeError):
        torch.ops.aten._make_per_tensor_quantized_tensor(ref_inp, 0.1, 0)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_tensor_quantized_tensor(inp, 0.1, 0)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_out_rejects_non_quantized_buffer(dtype):
    inp = _make_input((2, 3), dtype)
    ref_inp = tu.to_reference(inp)

    ref_buf = torch.empty((2, 3), dtype=torch.float32, device=_ref_device())
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(
            ref_inp, 0.1, 0, out=ref_buf
        )

    act_buf = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_tensor_quantized_tensor(inp, 0.1, 0, out=act_buf)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.parametrize("dtype", _MAKE_PERTENSOR_INPUT_DTYPES)
def test__make_per_tensor_quantized_tensor_out_rejects_wrong_quantized_dtype(dtype):
    inp = _make_input((2, 3), dtype)
    ref_inp = tu.to_reference(inp)

    ref_buf = torch.ops.aten._empty_affine_quantized(
        (2, 3),
        dtype=_WRONG_QUANT_DTYPE[dtype],
        device=_ref_device(),
        scale=1.0,
        zero_point=0,
    )
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(
            ref_inp, 0.1, 0, out=ref_buf
        )

    act_buf = torch.ops.aten._empty_affine_quantized(
        (2, 3),
        dtype=_WRONG_QUANT_DTYPE[dtype],
        device=flag_gems.device,
        scale=1.0,
        zero_point=0,
    )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_tensor_quantized_tensor(inp, 0.1, 0, out=act_buf)


@pytest.mark._make_per_tensor_quantized_tensor_out
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason="CPU reference resizes the out buffer; only the CUDA reference rejects "
    "a .out size that does not match the buffer (resize_ is unimplemented on "
    "QuantizedCUDA)",
)
def test__make_per_tensor_quantized_tensor_out_rejects_shape_mismatch():
    inp = torch.randint(0, 100, (4, 4), dtype=torch.uint8, device=flag_gems.device)
    buf = torch.ops.aten._empty_affine_quantized(
        (2, 2), dtype=torch.quint8, device=flag_gems.device, scale=1.0, zero_point=0
    )
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._make_per_tensor_quantized_tensor.out(inp, 0.5, 0, out=buf)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._make_per_tensor_quantized_tensor(inp, 0.5, 0, out=buf)
