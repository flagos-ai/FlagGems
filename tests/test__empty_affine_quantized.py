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

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import conftest as cfg
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
for _name in ("_empty_affine_quantized", "_empty_affine_quantized_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# Allocate uninitialized quantized storage; compare layout and per-tensor qparams.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

QUANT_SCALES = [-1.0, 0.0, 0.25, 1.0]

QUANT_ZERO_POINTS = [-2, 0, 3]

NON_FINITE_SCALES = [float("nan"), float("inf"), float("-inf")]

# The factory stores int64 zero points beyond the quantized storage range.
WIDE_ZERO_POINTS = [1 << 40]

CHANNELS_LAST_SHAPES = [(1, 3, 8, 8), (2, 3, 16, 16), (16, 3, 32, 32)]

CHANNELS_LAST_3D_SHAPES = [(2, 3, 8, 8, 8), (4, 7, 5, 5, 5)]

EMPTY_SHAPES = [(0,), (0, 3)]


def _ref_device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _assert_quant_metadata(res_out, ref_out):
    assert res_out.is_quantized
    assert res_out.qscheme() == ref_out.qscheme()
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    # The factory stores the scale (double) and zero_point (int64) verbatim.
    res_scale, ref_scale = res_out.q_scale(), ref_out.q_scale()
    if math.isnan(ref_scale):
        assert math.isnan(res_scale)
    else:
        assert res_scale == ref_scale
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # flag_gems.device may carry no index (e.g. 'cuda') while a created tensor
    # reports 'cuda:0', so compare the device type only.
    assert res_out.device.type == torch.device(flag_gems.device).type


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", QUANT_SCALES)
@pytest.mark.parametrize("zero_point", QUANT_ZERO_POINTS)
def test__empty_affine_quantized(shape, dtype, scale, zero_point):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=scale, zero_point=zero_point
    )

    res_out = flag_gems._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=scale, zero_point=zero_point
    )

    _assert_quant_metadata(res_out, ref_out)
    # The factory must return a fresh tensor, not a view of an internal buffer
    # (only the .out overload may return an aliased result).
    assert not res_out._is_view()
    # Default memory_format is the contiguous layout.
    assert res_out.is_contiguous()


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_value_ranges(shape, value_range, dtype):
    scale = tu.make_input(torch.float64, (1,), value_range).item()
    zero_point = tu.make_input(torch.int64, (1,), value_range).item()

    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=scale, zero_point=zero_point
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=scale, zero_point=zero_point
    )

    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", NON_FINITE_SCALES)
def test__empty_affine_quantized_non_finite_scale(shape, dtype, scale):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=scale, zero_point=0
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=scale, zero_point=0
    )

    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("zero_point", WIDE_ZERO_POINTS)
def test__empty_affine_quantized_wide_zero_point(shape, dtype, zero_point):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=1.0, zero_point=zero_point
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=1.0, zero_point=zero_point
    )

    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_empty(shape, dtype):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device()
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device
    )

    assert res_out.numel() == ref_out.numel() == 0
    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", CHANNELS_LAST_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_channels_last(shape, dtype):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), memory_format=torch.channels_last
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, memory_format=torch.channels_last
    )

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.is_contiguous(memory_format=torch.channels_last)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("shape", CHANNELS_LAST_3D_SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test__empty_affine_quantized_channels_last_3d(shape, dtype):
    ref_out = torch.ops.aten._empty_affine_quantized(
        shape,
        dtype=dtype,
        device=_ref_device(),
        memory_format=torch.channels_last_3d,
    )
    res_out = flag_gems._empty_affine_quantized(
        shape,
        dtype=dtype,
        device=flag_gems.device,
        memory_format=torch.channels_last_3d,
    )

    _assert_quant_metadata(res_out, ref_out)
    assert res_out.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark._empty_affine_quantized_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", QUANT_SCALES)
@pytest.mark.parametrize("zero_point", QUANT_ZERO_POINTS)
def test__empty_affine_quantized_out(shape, dtype, scale, zero_point):
    # Prefill different qparams so out must overwrite existing metadata.
    ref_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=_ref_device(), scale=2.5, zero_point=-5
    )
    ref_out = torch.ops.aten._empty_affine_quantized.out(
        shape, scale=scale, zero_point=zero_point, out=ref_out_buf
    )

    act_out_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=flag_gems.device, scale=2.5, zero_point=-5
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, scale=scale, zero_point=zero_point, out=act_out_buf
    )
    assert res_out is act_out_buf

    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized_out
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_cases([(16, 8)], quick=[(2, 19, 7)]))
def test__empty_affine_quantized_out_value_ranges(value_range, shape):
    scale = tu.make_input(torch.float64, (1,), value_range).item()
    zero_point = tu.make_input(torch.int64, (1,), value_range).item()

    ref_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=torch.quint8, device=_ref_device(), scale=2.5, zero_point=-5
    )
    ref_out = torch.ops.aten._empty_affine_quantized.out(
        shape, scale=scale, zero_point=zero_point, out=ref_buf
    )

    act_buf = torch.ops.aten._empty_affine_quantized(
        shape, dtype=torch.quint8, device=flag_gems.device, scale=2.5, zero_point=-5
    )
    res_out = flag_gems._empty_affine_quantized(
        shape, scale=scale, zero_point=zero_point, out=act_buf
    )
    assert res_out is act_buf

    _assert_quant_metadata(res_out, ref_out)


@pytest.mark._empty_affine_quantized_out
def test__empty_affine_quantized_out_non_contiguous_view():
    # Update the view qparams while leaving the base tensor qparams untouched.
    dtype = torch.quint8
    ref_base = torch.ops.aten._empty_affine_quantized(
        (16, 8), dtype=dtype, device=_ref_device(), scale=1.0, zero_point=0
    )
    ref_sliced = ref_base[:, ::2]
    ref_out = torch.ops.aten._empty_affine_quantized.out(
        (16, 4), scale=0.5, zero_point=3, out=ref_sliced
    )

    act_base = torch.ops.aten._empty_affine_quantized(
        (16, 8), dtype=dtype, device=flag_gems.device, scale=1.0, zero_point=0
    )
    act_sliced = act_base[:, ::2]
    res_out = flag_gems._empty_affine_quantized(
        (16, 4), scale=0.5, zero_point=3, out=act_sliced
    )
    assert res_out is act_sliced

    _assert_quant_metadata(res_out, ref_out)
    # The view's qparams are reset while the base keeps its original qparams.
    assert act_base.q_scale() == ref_base.q_scale()
    assert act_base.q_zero_point() == ref_base.q_zero_point()


@pytest.mark._empty_affine_quantized
def test__empty_affine_quantized_rejects_negative_size():
    with pytest.raises(RuntimeError):
        torch.ops.aten._empty_affine_quantized((-1,), dtype=torch.quint8)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._empty_affine_quantized((-1,), dtype=torch.quint8)


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8, torch.bool])
def test__empty_affine_quantized_rejects_non_quantized_dtype(dtype):
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._empty_affine_quantized((2, 3), dtype=dtype)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._empty_affine_quantized((2, 3), dtype=dtype)


@pytest.mark._empty_affine_quantized
def test__empty_affine_quantized_rejects_sparse_layout():
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._empty_affine_quantized(
            (2, 3), dtype=torch.quint8, layout=torch.sparse_coo
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._empty_affine_quantized(
            (2, 3), dtype=torch.quint8, layout=torch.sparse_coo
        )


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.channels_last_3d],
)
def test__empty_affine_quantized_rejects_invalid_memory_format(memory_format):
    # A rank-4 factory supports neither preserve_format nor channels_last_3d.
    if memory_format == torch.channels_last_3d:
        with pytest.raises((RuntimeError, TypeError)):
            torch.ops.aten._empty_affine_quantized(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            flag_gems._empty_affine_quantized(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )
    else:
        with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
            torch.ops.aten._empty_affine_quantized(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )
        with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
            flag_gems._empty_affine_quantized(
                (1, 3, 8, 8), dtype=torch.quint8, memory_format=memory_format
            )


@pytest.mark._empty_affine_quantized
@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": "not-a-float"},
        {"zero_point": 1.5},
        {"zero_point": 1 << 70},
    ],
)
def test__empty_affine_quantized_rejects_invalid_scalar_qparams(kwargs):
    with pytest.raises((RuntimeError, TypeError, ValueError, OverflowError)):
        torch.ops.aten._empty_affine_quantized((2, 3), dtype=torch.quint8, **kwargs)
    with pytest.raises((RuntimeError, TypeError, ValueError, OverflowError)):
        flag_gems._empty_affine_quantized((2, 3), dtype=torch.quint8, **kwargs)


@pytest.mark._empty_affine_quantized_out
def test__empty_affine_quantized_out_rejects_non_quantized_buffer():
    ref_buf = torch.empty((2, 3), dtype=torch.float32, device=_ref_device())
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        torch.ops.aten._empty_affine_quantized.out((2, 3), out=ref_buf)

    act_buf = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._empty_affine_quantized((2, 3), out=act_buf)


@pytest.mark._empty_affine_quantized_out
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason="CPU reference resizes the out buffer; only the CUDA reference rejects "
    "a .out size that does not match the buffer (resize_ is unimplemented on "
    "QuantizedCUDA)",
)
def test__empty_affine_quantized_out_rejects_shape_mismatch():
    buf = torch.ops.aten._empty_affine_quantized(
        (2, 3), dtype=torch.quint8, device=flag_gems.device
    )
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.aten._empty_affine_quantized.out((4, 6), out=buf)
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems._empty_affine_quantized((4, 6), out=buf)
