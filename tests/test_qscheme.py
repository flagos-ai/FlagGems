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

# aten::qscheme(Tensor self) -> QScheme reports the quantizer metadata a
# quantized tensor already carries and returns a plain Python int
# (0 per_tensor_affine, 1 per_channel_affine, 4 per_channel_affine_float_qparams).
# It reads no payload, and the native operator has no kernel for non-quantized
# dtypes, so the spec's nine required dtype families are exercised as negative
# rows while the positive dtype axis is the three quantized element types. No
# broadcast or backward grid applies: the query has a single tensor operand and
# quantized tensors carry no autograd support.
_PER_TENSOR = 0
_PER_CHANNEL = 1
_PER_CHANNEL_FLOAT_QPARAMS = 4

_QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Per-channel schemes save one (scale, zero_point) pair per channel along axis 0
# and need a channel axis, so the 0-dim spec shape appears in the per-tensor row
# only. Scalar and empty boundaries stay in both execution levels because
# building them is cheap and the query itself is metadata-only.
_PT_BOUNDARY_SHAPES = [(), (0,), (0, 3)]
_PC_BOUNDARY_SHAPES = [(0,), (0, 3)]

_PT_SHAPES = tu.selected_cases(
    list(tu.REQUIRED_SHAPES) + _PT_BOUNDARY_SHAPES,
    quick=list(tu.QUICK_SHAPES) + _PT_BOUNDARY_SHAPES,
)
_PC_SHAPES = tu.selected_cases(
    [shape for shape in tu.REQUIRED_SHAPES if shape] + _PC_BOUNDARY_SHAPES,
    quick=[shape for shape in tu.QUICK_SHAPES if shape] + _PC_BOUNDARY_SHAPES,
)

# View rows carry (builder, expected scheme): the reported scheme comes from the
# saved quantizer, and native quantization normalizes the result of a view --
# indexing the channel axis leaves a single channel and reports
# per_tensor_affine, while wider channel slices stay per_channel_affine. Those
# values were probed on the native operator. Float-qparams tensors cannot be
# viewed at all (indexing one raises "Setting strides is possible only on
# uniformly or per channel quantized tensors"), so views cover the other two
# schemes.
_PT_VIEW_GRID = {
    "row": (lambda q: q[0], _PER_TENSOR),
    "slice": (lambda q: q[0:2], _PER_TENSOR),
    "transpose": (lambda q: q.t(), _PER_TENSOR),
    "reshape": (lambda q: q.reshape(6, 4), _PER_TENSOR),
    "detach": (lambda q: q.detach(), _PER_TENSOR),
}
_PT_VIEWS = list(_PT_VIEW_GRID)

_PC_VIEW_GRID = {
    "row": (lambda q: q[1], _PER_TENSOR),
    "slice": (lambda q: q[1:3], _PER_CHANNEL),
    "narrow": (lambda q: q.narrow(0, 1, 2), _PER_CHANNEL),
    "reshape": (lambda q: q.reshape(2, 12), _PER_CHANNEL),
    "detach": (lambda q: q.detach(), _PER_CHANNEL),
    "contiguous": (lambda q: q.contiguous(), _PER_CHANNEL),
}
_PC_VIEWS = list(_PC_VIEW_GRID)

# The nan/inf matrix rides on the float payload the quantizer consumes, because
# the operator itself accepts only quantized dtypes.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (scenario, qdtype)
        for _dtype, scenario in tu.special_value_cases([torch.float32])
        for qdtype in _QUANT_DTYPES
    ],
    quick=[],
)

# Non-quantized dtypes are the invalid-input axis.
_NEGATIVE_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool]


def _quantize_per_tensor(payload, qdtype):
    return torch.quantize_per_tensor(payload, 0.1, 10, qdtype)


def _quantize_per_channel(payload, qdtype):
    channels = payload.shape[0]
    scales = torch.full((channels,), 0.25, dtype=torch.float64, device=payload.device)
    zero_points = torch.full((channels,), 10, dtype=torch.int64, device=payload.device)
    return torch.quantize_per_channel(payload, scales, zero_points, 0, qdtype)


def _empty_per_channel_float_qparams(shape, qdtype, device):
    # Only the saved scheme is queried, so the payload stays uninitialized and is
    # never read or compared.
    channels = shape[0]
    scales = torch.full((channels,), 0.25, dtype=torch.float64, device=device)
    zero_points = torch.full((channels,), 0.5, dtype=torch.float64, device=device)
    return torch._empty_per_channel_affine_quantized(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=0,
        dtype=qdtype,
        device=device,
    )


def _build_quantized(scheme, shape, qdtype):
    """Build a quantized tensor whose saved quantization scheme is ``scheme``."""
    if scheme == _PER_TENSOR:
        return _quantize_per_tensor(
            tu.make_input(torch.float32, shape, ["-1", "1"]), qdtype
        )
    if scheme == _PER_CHANNEL:
        return _quantize_per_channel(
            tu.make_input(torch.float32, shape, ["-1", "1"]), qdtype
        )
    return _empty_per_channel_float_qparams(shape, qdtype, flag_gems.device)


def _assert_scheme(res, ref, expected):
    """qscheme returns a plain Python int, so compare it directly as one."""
    assert type(res) is int
    assert res == ref
    assert res == expected


def _metadata_snapshot(inp):
    """Layout, storage and saved-quantizer state a metadata query must not change."""
    scheme = torch.ops.aten.qscheme(inp)
    snapshot = {
        "scheme": scheme,
        "dtype": inp.dtype,
        "shape": tuple(inp.shape),
        "stride": tuple(inp.stride()),
        "device": inp.device,
        "storage_ptr": inp.untyped_storage().data_ptr(),
        "storage_bytes": inp.untyped_storage().nbytes(),
    }
    if scheme in (_PER_CHANNEL, _PER_CHANNEL_FLOAT_QPARAMS):
        snapshot["axis"] = inp.q_per_channel_axis()
        snapshot["scales_ptr"] = inp.q_per_channel_scales().data_ptr()
        snapshot["scales"] = inp.q_per_channel_scales().tolist()
        snapshot["zero_points"] = inp.q_per_channel_zero_points().tolist()
    else:
        snapshot["scale"] = inp.q_scale()
        snapshot["zero_point"] = inp.q_zero_point()
    return snapshot


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize("shape", _PT_SHAPES)
def test_qscheme_per_tensor_affine(shape, qdtype):
    inp = _build_quantized(_PER_TENSOR, shape, qdtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.qscheme(ref_inp)
    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, ref_out, _PER_TENSOR)


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize("shape", _PC_SHAPES)
def test_qscheme_per_channel_affine(shape, qdtype):
    inp = _build_quantized(_PER_CHANNEL, shape, qdtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.qscheme(ref_inp)
    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, ref_out, _PER_CHANNEL)


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize("shape", _PC_SHAPES)
def test_qscheme_per_channel_float_qparams(shape, qdtype):
    inp = _build_quantized(_PER_CHANNEL_FLOAT_QPARAMS, shape, qdtype)
    # clone() refuses float-qparams tensors ("clone for quantized Tensor only
    # works for PerTensorAffine and PerChannelAffine qscheme"), so the oracle is
    # an independently constructed tensor built from the same recipe.
    ref_inp = _build_quantized(_PER_CHANNEL_FLOAT_QPARAMS, shape, qdtype)

    ref_out = torch.ops.aten.qscheme(ref_inp)
    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, ref_out, _PER_CHANNEL_FLOAT_QPARAMS)


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_qscheme_value_range_invariance(shape, value_range, qdtype):
    # The payload only carries the saved quantizer, so every range must report
    # the same scheme as the native operator.
    inp = _quantize_per_tensor(tu.make_input(torch.float32, shape, value_range), qdtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.qscheme(ref_inp)
    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, ref_out, _PER_TENSOR)


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize("view", _PT_VIEWS)
def test_qscheme_per_tensor_view(view, qdtype):
    build, expected = _PT_VIEW_GRID[view]
    inp = _build_quantized(_PER_TENSOR, (4, 6), qdtype)
    view_inp = build(inp)
    # The view keeps the base storage with its own offset/stride layout, and
    # detach() is the oracle handle because a quantized view is not cloneable.
    storage = view_inp.untyped_storage()
    ref_out = torch.ops.aten.qscheme(view_inp.detach())

    res_out = flag_gems.qscheme(view_inp)

    _assert_scheme(res_out, ref_out, expected)
    # Querying the scheme must not reallocate or copy the view's storage.
    assert view_inp.untyped_storage().data_ptr() == storage.data_ptr()


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize("view", _PC_VIEWS)
def test_qscheme_per_channel_view(view, qdtype):
    build, expected = _PC_VIEW_GRID[view]
    inp = _build_quantized(_PER_CHANNEL, (4, 6), qdtype)
    view_inp = build(inp)
    storage = view_inp.untyped_storage()
    ref_out = torch.ops.aten.qscheme(view_inp.detach())

    res_out = flag_gems.qscheme(view_inp)

    _assert_scheme(res_out, ref_out, expected)
    assert view_inp.untyped_storage().data_ptr() == storage.data_ptr()


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
@pytest.mark.parametrize(
    "scheme", [_PER_TENSOR, _PER_CHANNEL, _PER_CHANNEL_FLOAT_QPARAMS]
)
def test_qscheme_is_readonly(scheme, qdtype):
    inp = _build_quantized(scheme, (4, 6), qdtype)
    before = _metadata_snapshot(inp)

    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, before["scheme"], scheme)
    # The query must leave the saved quantizer, the qparams, the storage and the
    # device of its input untouched.
    assert _metadata_snapshot(inp) == before


@pytest.mark.qscheme
@pytest.mark.parametrize("qdtype", _QUANT_DTYPES)
def test_qscheme_cpu_quantized_tensor(qdtype):
    # The CPU form is a real native form of this metadata query, so the candidate
    # receives the CPU tensor itself and must keep it there.
    inp = _build_quantized(_PER_CHANNEL, (4, 6), qdtype).cpu()

    ref_out = torch.ops.aten.qscheme(inp)
    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, ref_out, _PER_CHANNEL)
    assert inp.device.type == "cpu"


@pytest.mark.qscheme
@pytest.mark.parametrize("scenario,qdtype", _SPECIAL_ROWS)
def test_qscheme_special_value_payload(scenario, qdtype):
    inp = _quantize_per_tensor(tu.make_special_input(torch.float32, scenario), qdtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.qscheme(ref_inp)
    res_out = flag_gems.qscheme(inp)

    _assert_scheme(res_out, ref_out, _PER_TENSOR)


@pytest.mark.qscheme
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_qscheme_rejects_unquantized_dtype(dtype):
    inp = torch.zeros(4, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.qscheme(inp)


@pytest.mark.qscheme
@pytest.mark.parametrize("dtype,scenario", tu.special_value_cases(tu.REQUIRED_DTYPES))
def test_qscheme_rejects_unquantized_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    with pytest.raises(RuntimeError):
        flag_gems.qscheme(inp)


@pytest.mark.qscheme
def test_qscheme_rejects_sparse_tensor():
    inp = torch.zeros(4, device=flag_gems.device).to_sparse()
    with pytest.raises(RuntimeError):
        flag_gems.qscheme(inp)


@pytest.mark.qscheme
def test_qscheme_rejects_dequantized_tensor():
    inp = _quantize_per_tensor(
        tu.make_input(torch.float32, (4, 6), ["-1", "1"]), torch.quint8
    ).dequantize()
    with pytest.raises(RuntimeError):
        flag_gems.qscheme(inp)


@pytest.mark.qscheme
def test_qscheme_rejects_none():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.qscheme(None)
