# Copyright (c) 2025, FlagGems Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import test_utils as tu

# q_scale reads the per-tensor affine scale stored in the quantizer and returns
# it as a Python float (kernels exist for QuantizedCPU and QuantizedCUDA). Only
# quantized containers carry a quantizer, so the spec ranges and shapes are
# applied to the float32 quantize source and the nine spec dtypes appear as
# negative rows. Broadcast and backward do not apply: the result is a scalar and
# quantized dtypes cannot require grad.

_Q_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

_BASE_SCALE = 0.1
_BASE_ZERO_POINT = 0
_SWEEP_SHAPE = (20, 320, 15)
_VIEW_SHAPE = (6, 8)
_EMPTY_SHAPES = [(0,), (0, 3)]

# The scale is stored verbatim, so zero, negative and extreme values stay valid.
_SCALE_ROWS = [
    (q_dtype, scale) for q_dtype in _Q_DTYPES for scale in (0.5, -0.5, 0.0, 1e-8, 1e8)
]
_SPECIAL_SCALE_ROWS = [
    (q_dtype, scale)
    for q_dtype in _Q_DTYPES
    for scale in (float("nan"), float("inf"), float("-inf"))
]

_ZERO_POINT_ROWS = [
    (torch.quint8, 0),
    (torch.quint8, 255),
    (torch.qint8, -128),
    (torch.qint8, 127),
    (torch.qint32, 0),
    (torch.qint32, 2147483647),
]

_SPECIAL_DATA_ROWS = [
    (q_dtype, kind) for q_dtype in _Q_DTYPES for kind in ("nan", "inf", "mixed")
]

_VIEW_KINDS = ["slice", "transpose", "offset", "row"]

_NON_QUANTIZED_DTYPES = tu.REQUIRED_DTYPES + [torch.float64, torch.bool]
_PER_CHANNEL_DTYPES = [torch.quint8, torch.qint8, torch.qint32]
_NON_TENSOR_ARGS = [3.14, None, [1.0]]


def _meta(inp):
    # Exactly comparable stored metadata; the float scale is checked separately so
    # a stored NaN compares as a matching NaN instead of failing tuple equality.
    return (
        inp.dtype,
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.qscheme(),
        inp.q_zero_point(),
    )


def _snapshot(inp):
    # to_reference compacts a quantized tensor's strides and resets its storage
    # offset, so the payload is copied for the read-only comparison.
    return _meta(inp), inp.q_scale(), tu.to_reference(inp.int_repr())


def _assert_scale_equal(got, want):
    # Python float compared through float64 buffers: exact stored precision with
    # matching nan/inf, never narrowed to the float32 quantize source.
    assert type(got) is float
    tu.assert_result_equal(
        torch.tensor(got, dtype=torch.float64),
        torch.tensor(want, dtype=torch.float64),
    )


def _assert_read_only(inp, snapshot):
    meta, scale, int_repr = snapshot
    assert _meta(inp) == meta
    _assert_scale_equal(inp.q_scale(), scale)
    tu.assert_result_equal(inp.int_repr(), int_repr)


def _special_src(kind, shape):
    src = torch.ones(shape, dtype=torch.float32, device=flag_gems.device)
    flat = src.reshape(-1)
    if kind == "nan":
        flat[0] = float("nan")
    elif kind == "inf":
        flat[0] = float("inf")
        flat[-1] = float("-inf")
    else:
        flat[0] = float("nan")
        flat[1] = float("inf")
    return src


def _quantized_view(kind, q_dtype):
    src = torch.rand(_VIEW_SHAPE, dtype=torch.float32, device=flag_gems.device)
    base = torch.quantize_per_tensor(src, _BASE_SCALE, _BASE_ZERO_POINT, q_dtype)
    if kind == "slice":
        return base[:, 2:6]
    if kind == "transpose":
        return base.t()
    if kind == "offset":
        # index a larger packing so the view has a nonzero storage offset
        src3 = torch.rand(
            (4,) + _VIEW_SHAPE, dtype=torch.float32, device=flag_gems.device
        )
        return torch.quantize_per_tensor(src3, _BASE_SCALE, _BASE_ZERO_POINT, q_dtype)[
            2
        ]
    return base[3]


@pytest.mark.q_scale
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("q_dtype", _Q_DTYPES)
def test_q_scale_per_tensor_affine(shape, value_range, q_dtype):
    inp = torch.quantize_per_tensor(
        tu.make_input(torch.float32, shape, value_range),
        _BASE_SCALE,
        _BASE_ZERO_POINT,
        q_dtype,
    )
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("case", _SCALE_ROWS)
def test_q_scale_stored_scale(case):
    # Every finite scale (zero, negative, tiny, large) is checked in both levels.
    q_dtype, scale = case
    src = torch.rand(_SWEEP_SHAPE, dtype=torch.float32, device=flag_gems.device)
    inp = torch.quantize_per_tensor(src, scale, _BASE_ZERO_POINT, q_dtype)
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("case", _ZERO_POINT_ROWS)
def test_q_scale_zero_point_boundary(case):
    q_dtype, zero_point = case
    src = torch.rand(_SWEEP_SHAPE, dtype=torch.float32, device=flag_gems.device)
    inp = torch.quantize_per_tensor(src, _BASE_SCALE, zero_point, q_dtype)
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("case", tu.selected_cases(_SPECIAL_SCALE_ROWS, quick=[]))
def test_q_scale_special_scale(case):
    q_dtype, scale = case
    src = torch.rand(_SWEEP_SHAPE, dtype=torch.float32, device=flag_gems.device)
    inp = torch.quantize_per_tensor(src, scale, _BASE_ZERO_POINT, q_dtype)
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("case", tu.selected_cases(_SPECIAL_DATA_ROWS, quick=[]))
def test_q_scale_special_data(case):
    q_dtype, kind = case
    inp = torch.quantize_per_tensor(
        _special_src(kind, _SWEEP_SHAPE),
        _BASE_SCALE,
        _BASE_ZERO_POINT,
        q_dtype,
    )
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("kind", _VIEW_KINDS)
@pytest.mark.parametrize("q_dtype", _Q_DTYPES)
def test_q_scale_quantized_view(kind, q_dtype):
    # Non-contiguous views with a nonzero storage offset; metadata is checked
    # against the candidate input's own snapshot, so to_reference compacting the
    # reference strides cannot weaken the check.
    inp = _quantized_view(kind, q_dtype)
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("q_dtype", _Q_DTYPES)
def test_q_scale_empty(shape, q_dtype):
    inp = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        _BASE_SCALE,
        _BASE_ZERO_POINT,
        q_dtype,
    )
    ref_inp = tu.to_reference(inp)
    snapshot = _snapshot(inp)

    ref_out = torch.ops.aten.q_scale(ref_inp)
    res_out = flag_gems.q_scale(inp)

    _assert_scale_equal(res_out, ref_out)
    _assert_read_only(inp, snapshot)


@pytest.mark.q_scale
@pytest.mark.parametrize("dtype", _NON_QUANTIZED_DTYPES)
def test_q_scale_rejects_non_quantized(dtype):
    # No quantizer means no stored scale for any plain (non-quantized) dtype.
    inp = torch.zeros((2, 3), dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.q_scale(inp)


@pytest.mark.q_scale
@pytest.mark.parametrize("q_dtype", _PER_CHANNEL_DTYPES)
def test_q_scale_rejects_per_channel(q_dtype):
    # q_scale requires kPerTensorAffine; a per-channel quantizer has no single
    # stored scale. quantize_per_channel needs its scales and zero points on the
    # input's device, so they follow flag_gems.device as well.
    src = torch.rand((4, 6), dtype=torch.float32, device=flag_gems.device)
    inp = torch.quantize_per_channel(
        src,
        torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float64, device=flag_gems.device
        ),
        torch.zeros(4, dtype=torch.int64, device=flag_gems.device),
        0,
        q_dtype,
    )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.q_scale(inp)


@pytest.mark.q_scale
@pytest.mark.parametrize("arg", _NON_TENSOR_ARGS)
def test_q_scale_rejects_non_tensor(arg):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.q_scale(arg)
