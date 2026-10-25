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

import flag_gems

from .test_upsample_nearest2d_backward import SCALE_CASES, _ceil_cpu_reference

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead", reason="PPU native explicit-scale contract"
)
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.uint8]
CONTRACT_CASES = SCALE_CASES + [
    # Round both the scale and its coordinate product to FP32 before ceil.
    ((1, 2, 11, 3), (6, 8), 0.1, 0.7),
    ((1, 2, 3, 11), (8, 6), 0.7, 0.1),
    # The final input interval does not consume the remaining output tail.
    ((1, 2, 3, 5), (8, 12), 0.7, 0.7),
    ((1, 2, 3, 5), (6, 10), None, None),
    ((1, 2, 7, 9), (3, 4), 0.4, 0.3),
    ((1, 2, 7, 9), (3, 4), 2.2, 1.7),
    ((1, 2, 1, 3), (5, 7), 0.7, 2.2),
    ((1, 2, 3, 1), (7, 5), 2.2, 0.7),
]


def _guarded_tensor(shape, dtype, layout):
    n, c, h, w = shape
    strides = (c * h * w, h * w, w, 1)
    if layout == "channels_last":
        strides = (c * h * w, 1, c * w, c)
    elif layout == "strided":
        strides = tuple(2 * s for s in strides)
    extent = 1 + sum((d - 1) * s for d, s in zip(shape, strides))
    storage = torch.full((extent + 2,), 251, dtype=dtype, device=flag_gems.device)
    value = storage.as_strided(shape, strides, 1)
    touched = torch.zeros(extent + 2, dtype=torch.bool)
    touched.as_strided(shape, strides, 1).fill_(True)
    return value, storage, ~touched


def _check_contract(case, dtype, layout, variant):
    shape, size, sh, sw = case
    source_shape = (*shape[:2], *size)
    values = (torch.arange(math.prod(source_shape)) % 17).reshape(source_shape)
    if dtype != torch.uint8:
        values -= 8
    values = values.to(dtype)
    source, source_storage, _ = _guarded_tensor(source_shape, dtype, layout)
    source.copy_(values.to(flag_gems.device))
    source_before = source_storage.cpu().clone()
    # This oracle uses only CPU arithmetic over the supplied gradient. It does
    # not invoke Gems or CPU ATen's incompatible legacy backward mapping.
    expected = _ceil_cpu_reference(values, size, shape, sh, sw).to(dtype)
    native_out, native_storage, native_guard = _guarded_tensor(shape, dtype, layout)
    output, storage, guard = _guarded_tensor(shape, dtype, layout)
    if dtype == torch.uint8:
        prefill = expected.bitwise_not().to(flag_gems.device)
        native_out.copy_(prefill)
        output.copy_(prefill)
    else:
        native_out.fill_(float("nan"))
        output.fill_(float("nan"))
    native_kwargs = {} if variant == "default" else {"grad_input": native_out}
    native = getattr(torch.ops.aten.upsample_nearest2d_backward, variant)(
        source, size, shape, sh, sw, **native_kwargs
    )
    torch.testing.assert_close(native.cpu(), expected, rtol=0, atol=0)
    if variant == "grad_input":
        assert native is native_out
        assert torch.all(native_storage.cpu()[native_guard] == 251)
        stride, pointer = output.stride(), output.data_ptr()
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            source, size, shape, sh, sw, grad_input=output
        )
        assert actual is output
        assert actual.stride() == stride
        assert actual.data_ptr() == pointer
        assert actual.storage_offset() == 1
        assert torch.all(storage.cpu()[guard] == 251)
    else:
        actual = flag_gems.ops.upsample_nearest2d_backward(source, size, shape, sh, sw)
        if layout == "channels_last":
            assert actual.is_contiguous(memory_format=torch.channels_last)
    assert actual.shape == shape
    assert actual.dtype == dtype
    assert actual.device == source.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source_storage.cpu(), source_before, rtol=0, atol=0)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("case", CONTRACT_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "strided"])
def test_upsample_nearest2d_backward_ppu_scale_contract(case, dtype, layout):
    _check_contract(case, dtype, layout, "default")


@pytest.mark.upsample_nearest2d_backward_grad_input
@pytest.mark.parametrize("case", CONTRACT_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "strided"])
def test_upsample_nearest2d_backward_grad_input_ppu_scale_contract(case, dtype, layout):
    _check_contract(case, dtype, layout, "grad_input")


@pytest.mark.upsample_nearest2d_backward_grad_input
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("old_shape", [(0,), (2,), (2, 3, 4)])
def test_upsample_nearest2d_backward_grad_input_ppu_scale_resize(dtype, old_shape):
    shape, size, sh, sw = CONTRACT_CASES[0]
    values = torch.arange(math.prod(size)).reshape(1, 1, *size).to(dtype)
    source = values.to(flag_gems.device)
    output = torch.full(old_shape, 123, dtype=dtype, device=flag_gems.device)
    native_out = torch.full_like(output, 123)
    expected = _ceil_cpu_reference(values, size, shape, sh, sw).to(dtype)
    native = torch.ops.aten.upsample_nearest2d_backward.grad_input(
        source, size, shape, sh, sw, grad_input=native_out
    )
    actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
        source, size, shape, sh, sw, grad_input=output
    )
    assert actual is output
    assert native is native_out
    assert actual.stride() == native.stride()
    torch.testing.assert_close(native.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.upsample_nearest2d_backward_grad_input
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_ppu_scale_accumulation(dtype, variant):
    values = torch.ones((1, 1, 64, 65), dtype=dtype)
    if dtype == torch.float64:
        values += (
            torch.arange(values.numel(), dtype=torch.float64).reshape(values.shape)
            * 1e-9
        )
    shape, size = (1, 1, 1, 1), (64, 65)
    expected = _ceil_cpu_reference(values, size, shape, 63.7, 64.2).to(dtype)
    source = values.to(flag_gems.device)
    kwargs = (
        {}
        if variant == "default"
        else {
            "grad_input": torch.full(shape, 123, dtype=dtype, device=flag_gems.device)
        }
    )
    native = getattr(torch.ops.aten.upsample_nearest2d_backward, variant)(
        source, size, shape, 63.7, 64.2, **kwargs
    ).cpu()
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            source, size, shape, 63.7, 64.2
        )
    else:
        kwargs["grad_input"].fill_(123)
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            source, size, shape, 63.7, 64.2, **kwargs
        )
        assert actual is kwargs["grad_input"]
    # FP64 reduction order may differ; this bound still rejects FP32 demotion.
    atol = 1e-10 if dtype == torch.float64 else 0
    torch.testing.assert_close(native, expected, rtol=0, atol=atol)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=atol)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.upsample_nearest2d_backward_grad_input
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_ppu_explicit_int64_address(variant):
    stride = 2**31 + 17
    source = torch.empty_strided(
        (1, 1, 2, 4),
        (2 * stride, 2 * stride, stride, 1),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    values = torch.arange(8, dtype=torch.float32).reshape(source.shape)
    source.copy_(values.to(flag_gems.device))
    shape, size = (1, 1, 2, 2), (2, 4)
    expected = _ceil_cpu_reference(values, size, shape, 0.7, 0.7)
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            source, size, shape, 0.7, 0.7
        )
    else:
        output = torch.empty_strided(
            shape,
            (2 * stride, 2 * stride, stride, 1),
            dtype=torch.float32,
            device=flag_gems.device,
        )
        output.fill_(float("nan"))
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            source, size, shape, 0.7, 0.7, grad_input=output
        )
        assert actual is output
        assert actual.stride() == (2 * stride, 2 * stride, stride, 1)
    native = torch.ops.aten.upsample_nearest2d_backward.default(
        source, size, shape, 0.7, 0.7
    )
    torch.testing.assert_close(native.cpu().double(), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0, atol=0)
