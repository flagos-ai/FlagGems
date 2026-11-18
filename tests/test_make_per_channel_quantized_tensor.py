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

pytestmark = [
    pytest.mark.make_per_channel_quantized_tensor,
]

_QINT = [torch.uint8, torch.int8, torch.int32]

# (shape, axis) pairs: 2D both axes, a 3D middle axis, a 4D inner axis and a
# negative axis (stored verbatim by the reference).
_SHAPES_AXES = [
    ((4, 8), 0),
    ((4, 8), 1),
    ((2, 3, 4), 1),
    ((2, 3, 4, 5), 2),
    ((16, 32), -1),
]


def _ref(inp, scales, zero_points, axis):
    return torch.ops.aten._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )


def _rand_int(shape, dtype):
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.int32:
        return torch.randint(-1000, 1000, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-128, 128, shape, dtype=dtype, device=flag_gems.device)


def _assert_same_quantized(res, ref):
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert res.stride() == ref.stride()
    assert res.qscheme() == ref.qscheme()
    assert res.q_per_channel_axis() == ref.q_per_channel_axis()
    utils.gems_assert_equal(res.q_per_channel_scales(), ref.q_per_channel_scales())
    utils.gems_assert_equal(
        res.q_per_channel_zero_points(), ref.q_per_channel_zero_points()
    )
    utils.gems_assert_equal(res.int_repr(), ref.int_repr())


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", _SHAPES_AXES)
@pytest.mark.parametrize("dtype", _QINT)
def test_make_per_channel_quantized_tensor(shape, axis, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rand_int(shape, dtype)
    n = shape[axis]
    scales = torch.rand(n, device=flag_gems.device) + 0.01
    zero_points = torch.randint(0, 10, (n,), device=flag_gems.device, dtype=torch.int64)

    ref_out = _ref(inp, scales, zero_points, axis)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    _assert_same_quantized(res_out, ref_out)


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", _SHAPES_AXES)
@pytest.mark.parametrize("dtype", _QINT)
def test_make_per_channel_quantized_tensor_float_zero_points(shape, axis, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rand_int(shape, dtype)
    n = shape[axis]
    scales = torch.rand(n, device=flag_gems.device) + 0.01
    zero_points = torch.rand(n, device=flag_gems.device)

    ref_out = _ref(inp, scales, zero_points, axis)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    assert res_out.qscheme() == torch.per_channel_affine_float_qparams
    _assert_same_quantized(res_out, ref_out)


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("dtype", _QINT)
def test_make_per_channel_quantized_tensor_noncontiguous(dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rand_int((16, 32), dtype)[:, ::2]
    n = inp.shape[1]
    scales = torch.rand(n, device=flag_gems.device) + 0.01
    zero_points = torch.randint(0, 10, (n,), device=flag_gems.device, dtype=torch.int64)

    ref_out = _ref(inp, scales, zero_points, 1)
    res_out = flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)

    _assert_same_quantized(res_out, ref_out)


@pytest.mark.make_per_channel_quantized_tensor
def test_make_per_channel_quantized_tensor_empty():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.empty((0, 4), dtype=torch.int8, device=flag_gems.device)
    scales = torch.rand(4, device=flag_gems.device) + 0.5
    zero_points = torch.randint(0, 10, (4,), device=flag_gems.device, dtype=torch.int64)

    res_out = flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)

    assert tuple(res_out.shape) == (0, 4)
    assert res_out.dtype == torch.qint8
    utils.gems_assert_equal(res_out.q_per_channel_scales(), scales.double())
    utils.gems_assert_equal(res_out.q_per_channel_zero_points(), zero_points)


def _float_reference(inp, scales, zero_points, axis, qdtype):
    """AO-style per-channel affine quantization, computed with plain torch ops.

    ``_make_per_channel_quantized_tensor`` does not accept float inputs, so this
    is the reference for the on-the-fly quantization branch. The scales used by
    the tests are powers of two, so ``x * (1 / scale)`` and ``x / scale`` agree
    exactly and the only rounding step is the half-to-even one.
    """
    ndim = inp.dim()
    ax = axis % ndim
    view = [1] * ndim
    view[ax] = inp.shape[ax]
    inv = (1.0 / scales).reshape(view).to(torch.float32)
    zp = zero_points.reshape(view).to(torch.int32)
    qmin, qmax = (-128, 127) if qdtype == torch.qint8 else (0, 255)
    q = torch.round(inp.to(torch.float32) * inv).to(torch.int32) + zp
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int8 if qdtype == torch.qint8 else torch.uint8)


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape,axis", [((4, 8), 0), ((2, 3, 4), 1), ((16, 32), -1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_make_per_channel_quantized_tensor_float_input(shape, axis, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 8.0
    n = shape[axis]
    # Powers of two make 1 / scale exact in float32.
    scales = torch.pow(2.0, torch.randint(-4, 0, (n,), device=flag_gems.device).float())
    zero_points = torch.randint(0, 10, (n,), device=flag_gems.device, dtype=torch.int64)

    res_out = flag_gems._make_per_channel_quantized_tensor(
        inp, scales, zero_points, axis
    )

    qdtype = torch.qint8 if dtype == torch.float32 else torch.quint8
    qmin, qmax = (-128, 127) if qdtype == torch.qint8 else (0, 255)
    expected = _float_reference(inp, scales, zero_points, axis, qdtype)
    assert res_out.dtype == qdtype
    assert res_out.qscheme() == torch.per_channel_affine
    # The reference stores the axis verbatim, including negative values.
    assert res_out.q_per_channel_axis() == axis
    utils.gems_assert_equal(res_out.int_repr(), expected)
    utils.gems_assert_equal(res_out.q_per_channel_scales(), scales.double())
    utils.gems_assert_equal(res_out.q_per_channel_zero_points(), zero_points)


@pytest.mark.make_per_channel_quantized_tensor
def test_make_per_channel_quantized_tensor_unsupported_dtype():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.randint(-1000, 1000, (4, 8), dtype=torch.int64, device=flag_gems.device)
    scales = torch.rand(8, device=flag_gems.device) + 0.5
    zero_points = torch.zeros(8, device=flag_gems.device, dtype=torch.int64)

    with pytest.raises(TypeError):
        flag_gems._make_per_channel_quantized_tensor(inp, scales, zero_points, 1)
