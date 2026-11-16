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
"""Accuracy tests for ``aten::_empty_affine_quantized``.

This operator is a factory: it allocates a fresh, *uninitialized* buffer and
attaches a per-tensor affine quantizer.  The values in the buffer are
unspecified by contract (the ATen reference returns whatever the allocator
hands back), so nothing here compares ``int_repr()`` between the two
implementations.  What is checked instead is the metadata that the op actually
promises:

* dtype, shape, device and layout of the result;
* ``qscheme()`` (``per_tensor_affine``) and the ``q_scale()`` /
  ``q_zero_point()`` read back through the tensor methods;
* strides, including ``memory_format=torch.channels_last``;
* the ``.out`` overload, which resizes ``out``, applies the ``scale`` /
  ``zero_point`` arguments and returns ``out``.

Error behaviour is delegated to the native factory, so the rejection cases
(missing dtype, string ``memory_format``) are pinned against the reference.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = [pytest.mark.empty_affine_quantized]

_QUANTIZED_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Shapes cover the ordinary 2-D/4-D cases plus a zero-extent output.
_SHAPES = [(2, 4), (3, 8), (2, 3, 4, 5), (0, 4)]


def _ref(size, **kwargs):
    return torch.ops.aten._empty_affine_quantized.default(size, **kwargs)


def _gems(size, **kwargs):
    return flag_gems._empty_affine_quantized(size, **kwargs)


def _assert_metadata(res, ref, *, size, dtype, scale, zero_point):
    assert tuple(res.shape) == tuple(size)
    assert res.dtype == dtype
    assert res.device == ref.device
    assert res.layout == ref.layout
    assert res.is_quantized

    assert res.qscheme() == ref.qscheme() == torch.per_tensor_affine
    assert res.q_scale() == pytest.approx(scale)
    assert res.q_zero_point() == zero_point
    assert res.stride() == ref.stride()

    # The int representation exists and has the requested shape; its contents
    # are uninitialized and intentionally not compared.
    int_repr = res.int_repr()
    assert tuple(int_repr.shape) == tuple(size)
    assert int_repr.device == res.device


@pytest.mark.empty_affine_quantized
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _QUANTIZED_DTYPES)
def test_empty_affine_quantized(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    scale, zero_point = 0.5, 3

    ref_out = _ref(
        list(shape),
        dtype=dtype,
        device=flag_gems.device,
        scale=scale,
        zero_point=zero_point,
    )
    res_out = _gems(
        list(shape),
        dtype=dtype,
        device=flag_gems.device,
        scale=scale,
        zero_point=zero_point,
    )

    _assert_metadata(
        res_out, ref_out, size=shape, dtype=dtype, scale=scale, zero_point=zero_point
    )


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized_memory_format():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape = (2, 3, 4, 5)

    ref_out = _ref(
        list(shape),
        dtype=torch.quint8,
        device=flag_gems.device,
        memory_format=torch.channels_last,
    )
    res_out = _gems(
        list(shape),
        dtype=torch.quint8,
        device=flag_gems.device,
        memory_format=torch.channels_last,
    )

    assert res_out.stride() == ref_out.stride()
    assert res_out.is_contiguous(memory_format=torch.channels_last)
    assert res_out.qscheme() == ref_out.qscheme() == torch.per_tensor_affine


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized_device_default_cpu():
    shape = (2, 4)
    ref_out = _ref(list(shape), dtype=torch.quint8)
    res_out = _gems(list(shape), dtype=torch.quint8)

    assert res_out.device.type == "cpu"
    assert ref_out.device.type == "cpu"
    assert tuple(res_out.shape) == shape
    assert res_out.qscheme() == ref_out.qscheme() == torch.per_tensor_affine
    assert res_out.q_scale() == ref_out.q_scale() == 1.0
    assert res_out.q_zero_point() == ref_out.q_zero_point() == 0


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized_out_matches_reference():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    shape = (2, 4)
    scale, zero_point = 2.5, 7

    ref_out = torch.quantize_per_tensor(
        torch.zeros(shape, device=flag_gems.device), 1.0, 0, torch.quint8
    )
    res_out = torch.quantize_per_tensor(
        torch.zeros(shape, device=flag_gems.device), 1.0, 0, torch.quint8
    )

    ref = torch.ops.aten._empty_affine_quantized.out(
        shape,
        scale=scale,
        zero_point=zero_point,
        memory_format=torch.contiguous_format,
        out=ref_out,
    )
    res = flag_gems._empty_affine_quantized_out(
        shape,
        scale=scale,
        zero_point=zero_point,
        memory_format=torch.contiguous_format,
        out=res_out,
    )

    assert ref is ref_out
    assert res is res_out
    assert tuple(res.shape) == tuple(ref.shape) == shape
    assert res.dtype == ref.dtype == torch.quint8
    assert res.device == ref.device
    assert res.qscheme() == ref.qscheme() == torch.per_tensor_affine
    # The native .out path replaces the destination quantizer with the
    # scale/zero_point arguments.
    assert res.q_scale() == pytest.approx(scale) == pytest.approx(ref.q_scale())
    assert res.q_zero_point() == zero_point == ref.q_zero_point()
    assert res.stride() == ref.stride()


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized_requires_quantized_dtype():
    shape = (2, 4)

    # The exact message depends on which dispatch key ATen resolves, so only
    # the rejection itself is pinned.  NotImplementedError subclasses
    # RuntimeError, which covers the device-key and QuantizedCPU paths.
    with pytest.raises(RuntimeError):
        _ref(list(shape))
    with pytest.raises(RuntimeError):
        _gems(list(shape))


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized_rejects_string_memory_format():
    shape = (2, 4)

    # ``memory_format`` must be a ``torch.memory_format``; the ATen packet
    # rejects a string during argument casting, and FlagGems reports the same
    # rejection through redispatch.
    with pytest.raises(RuntimeError):
        _ref(list(shape), dtype=torch.quint8, memory_format="contiguous_format")
    with pytest.raises(RuntimeError):
        _gems(list(shape), dtype=torch.quint8, memory_format="contiguous_format")


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized_registered_in_full_config():
    from flag_gems import _FULL_CONFIG

    keys = {entry[0] for entry in _FULL_CONFIG}
    assert "_empty_affine_quantized" in keys
    assert "_empty_affine_quantized.out" in keys
