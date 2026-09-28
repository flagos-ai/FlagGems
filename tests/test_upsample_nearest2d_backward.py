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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_ASCEND_SIZE_ONLY = False
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _ASCEND_SIZE_ONLY = get_cann_version() == "8.5.0"

DTYPES = [
    torch.float32,
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not utils.bf16_is_supported, reason="Backend does not support BF16"
        ),
    ),
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            (not utils.fp64_is_supported and flag_gems.vendor_name != "mthreads")
            or (flag_gems.vendor_name == "mthreads" and not utils.TO_CPU)
            or flag_gems.vendor_name == "iluvatar",
            reason="Backend or native backward reference does not support FP64",
        ),
    ),
]
CASES = [
    ((2, 3, 7, 9), (7, 9), None, None),
    ((2, 3, 7, 9), (14, 18), None, None),
    ((3, 5, 7, 11), (19, 23), None, None),
    ((1, 8, 13, 19), (5, 7), None, None),
    ((2, 3, 31, 47), (2, 3), None, None),
    ((1, 2, 5, 7), (1, 1), None, None),
    ((1, 3, 1, 9), (13, 5), None, None),
    ((2, 1, 7, 1), (3, 11), None, None),
    ((1, 1, 1, 1), (17, 19), None, None),
    ((0, 3, 5, 7), (9, 11), None, None),
    ((2, 0, 5, 7), (9, 11), None, None),
    ((2, 3, 7, 9), (14, 18), 2.0, 2.0),
    ((1, 2, 7, 11), (17, 19), 2.5, 1.75),
]
SCALE_CASES = [
    ((1, 1, 2, 3), (4, 6), 0.7, 2.2),
    ((1, 3, 3, 3), (5, 7), 0.7, 0.7),
    ((1, 4, 3, 3), (3, 7), 0.7, 0.7),
    ((2, 3, 3, 3), (5, 3), 0.7, 2.2),
    ((1, 2, 3, 5), (7, 11), None, 0.4),
    ((1, 2, 3, 5), (7, 11), 0.4, None),
    ((1, 1, 3, 5), (3, 5), 0.7, 2.2),
]


def _tensor(shape, dtype, layout="contiguous"):
    value = (
        torch.randint(0, 256, shape, dtype=dtype, device=flag_gems.device)
        if dtype == torch.uint8
        else torch.randn(shape, dtype=dtype, device=flag_gems.device)
    )
    if layout == "channels_last":
        return value.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    if layout == "strided":
        storage = torch.empty(
            (*shape[:-1], shape[-1] * 2), dtype=dtype, device=flag_gems.device
        )
        result = storage[..., ::2]
        result.copy_(value)
        return result
    if layout == "transpose":
        return value.transpose(-1, -2).contiguous().transpose(-1, -2)
    return value


def _ceil_cpu_reference(grad_output, output_size, input_size, sh, sw):
    # Independent CPU oracle for the native GPU backward convention. CPU ATen
    # instead transposes its legacy forward map when scale and size disagree.
    if tuple(input_size[-2:]) == tuple(output_size):
        return grad_output.cpu().to(torch.float64)
    oh, ow = output_size
    ih, iw = input_size[-2:]
    h_scale = np.float32(oh / ih if sh is None or _ASCEND_SIZE_ONLY else sh)
    w_scale = np.float32(ow / iw if sw is None or _ASCEND_SIZE_ONLY else sw)
    result = torch.zeros(input_size, dtype=torch.float64)
    source = grad_output.cpu().to(torch.float64)
    for y in range(ih):
        y0 = min(math.ceil(float(np.float32(y) * h_scale)), oh)
        y1 = min(math.ceil(float(np.float32(y + 1) * h_scale)), oh)
        for x in range(iw):
            x0 = min(math.ceil(float(np.float32(x) * w_scale)), ow)
            x1 = min(math.ceil(float(np.float32(x + 1) * w_scale)), ow)
            result[:, :, y, x] = source[:, :, y0:y1, x0:x1].sum((-2, -1))
    return result


def _reference(grad_output, output_size, input_size, sh, sw):
    if flag_gems.vendor_name == "mthreads" and tuple(input_size[-2:]) == tuple(
        output_size
    ):
        # muDNN applies explicit scales even for equal spatial sizes. The
        # operator contract requires identity, as do CPU, CUDA and Ascend.
        return utils.to_reference(grad_output, upcast=utils.TO_CPU)
    if utils.TO_CPU and flag_gems.vendor_name in (
        "nvidia",
        "ascend",
        "mthreads",
        "hygon",
        "iluvatar",
    ):
        return _ceil_cpu_reference(grad_output, output_size, input_size, sh, sw)
    return torch.ops.aten.upsample_nearest2d_backward.default(
        utils.to_reference(grad_output, upcast=utils.TO_CPU),
        output_size,
        input_size,
        sh,
        sw,
    )


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES + SCALE_CASES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "strided"])
def test_upsample_nearest2d_backward(case, dtype, layout):
    shape, size, sh, sw = case
    grad_output = _tensor((*shape[:2], *size), dtype, layout)
    expected = _reference(grad_output, size, shape, sh, sw)
    actual = flag_gems.ops.upsample_nearest2d_backward(grad_output, size, shape, sh, sw)
    assert actual.shape == shape
    assert actual.dtype == dtype
    assert actual.device == grad_output.device
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES + SCALE_CASES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transpose"])
@pytest.mark.upsample_nearest2d_backward_grad_input
def test_upsample_nearest2d_backward_grad_input(case, dtype, layout):
    shape, size, sh, sw = case
    grad_output = _tensor((*shape[:2], *size), dtype, "strided")
    grad_input = _tensor(shape, dtype, layout)
    original_stride = grad_input.stride()
    expected = _reference(grad_output, size, shape, sh, sw)
    actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
        grad_output, size, shape, sh, sw, grad_input=grad_input
    )
    assert actual is grad_input
    assert actual.stride() == original_stride
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES + SCALE_CASES)
@pytest.mark.upsample_nearest2d_backward_grad_input
def test_upsample_nearest2d_backward_grad_input_contiguous(case, dtype):
    shape, size, sh, sw = case
    grad_output = _tensor((*shape[:2], *size), dtype)
    grad_input = _tensor(shape, dtype)
    original_stride = grad_input.stride()
    expected = _reference(grad_output, size, shape, sh, sw)
    actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
        grad_output, size, shape, sh, sw, grad_input=grad_input
    )
    assert actual is grad_input
    assert actual.stride() == original_stride
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("case", SCALE_CASES)
def test_upsample_nearest2d_backward_cpu_legacy_reference(case):
    shape, size, sh, sw = case
    grad_output = torch.arange(math.prod((*shape[:2], *size)), dtype=torch.float64)
    grad_output = grad_output.reshape(*shape[:2], *size)
    expected = torch.zeros(shape, dtype=torch.float64)
    indices = []
    for input_length, output_length, scale in zip(shape[-2:], size, (sh, sw)):
        if input_length == output_length:
            indices.append(list(range(output_length)))
        elif output_length == 2 * input_length:
            indices.append([i >> 1 for i in range(output_length)])
        else:
            ratio = np.float32(
                input_length / output_length if scale is None else 1 / scale
            )
            indices.append(
                [
                    min(int(np.float32(i) * ratio), input_length - 1)
                    for i in range(output_length)
                ]
            )
    for y, iy in enumerate(indices[0]):
        for x, ix in enumerate(indices[1]):
            expected[:, :, iy, ix] += grad_output[:, :, y, x]
    actual = torch.ops.aten.upsample_nearest2d_backward.default(
        grad_output, size, shape, sh, sw
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("variant", ["default", "grad_input"])
@pytest.mark.parametrize("case", SCALE_CASES)
def test_upsample_nearest2d_backward_native_device_scales(case, variant):
    shape, size, sh, sw = case
    grad_output = _tensor((*shape[:2], *size), torch.float32)
    kwargs = (
        {}
        if variant == "default"
        else {"grad_input": torch.empty_like(_tensor(shape, torch.float32))}
    )
    native = getattr(torch.ops.aten.upsample_nearest2d_backward, variant)
    expected = native(grad_output, size, shape, sh, sw, **kwargs).clone()
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            grad_output, size, shape, sh, sw
        )
    else:
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            grad_output, size, shape, sh, sw, **kwargs
        )
    if flag_gems.vendor_name == "mthreads" and tuple(shape[-2:]) == tuple(size):
        # The native muDNN identity/explicit-scale result is not an identity
        # oracle. Keep this required case and verify the identity contract.
        expected = grad_output
    torch.testing.assert_close(actual, expected)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 3, 4)])
def test_upsample_nearest2d_backward_grad_input_resize(shape):
    grad_output = _tensor((2, 3, 10, 14), torch.float32)
    output = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    native_output = torch.empty_like(output)
    native = torch.ops.aten.upsample_nearest2d_backward.grad_input(
        grad_output, (10, 14), (2, 3, 5, 7), grad_input=native_output
    )
    actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
        grad_output, (10, 14), (2, 3, 5, 7), grad_input=output
    )
    assert actual is output
    assert native is native_output
    assert actual.stride() == native.stride()
    torch.testing.assert_close(actual, native)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("mismatch", ["dtype", "device"])
def test_upsample_nearest2d_backward_grad_input_mismatch(mismatch):
    grad_output = _tensor((1, 1, 4, 6), torch.float32)
    output = torch.empty(
        (1, 1, 2, 3),
        dtype=torch.float16 if mismatch == "dtype" else torch.float32,
        device="cpu" if mismatch == "device" else flag_gems.device,
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.upsample_nearest2d_backward.grad_input(
            grad_output, (4, 6), (1, 1, 2, 3), grad_input=output
        )
    with pytest.raises(RuntimeError):
        flag_gems.ops.upsample_nearest2d_backward_grad_input(
            grad_output, (4, 6), (1, 1, 2, 3), grad_input=output
        )


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("variant", ["default", "grad_input"])
@pytest.mark.parametrize(
    "go_shape,size,shape",
    [
        ((1, 4, 6), (4, 6), (1, 1, 2, 3)),
        ((1, 1, 4, 6), (4,), (1, 1, 2, 3)),
        ((1, 1, 4, 6), (4, 6), (1, 2, 3)),
        ((1, 1, 4, 6), (4, 7), (1, 1, 2, 3)),
        ((1, 1, 4, 6), (4, 6), (2, 1, 2, 3)),
        ((1, 1, 4, 6), (4, 6), (1, 2, 2, 3)),
        ((1, 1, 4, 6), (4, 6), (1, 1, 0, 3)),
        ((1, 1, 0, 6), (0, 6), (1, 1, 2, 3)),
    ],
)
def test_upsample_nearest2d_backward_invalid(go_shape, size, shape, variant):
    grad_output = _tensor(go_shape, torch.float32)
    kwargs = (
        {} if variant == "default" else {"grad_input": torch.empty_like(grad_output)}
    )
    with pytest.raises(RuntimeError):
        getattr(torch.ops.aten.upsample_nearest2d_backward, variant)(
            grad_output, size, shape, **kwargs
        )
    with pytest.raises(RuntimeError):
        if variant == "default":
            flag_gems.ops.upsample_nearest2d_backward(grad_output, size, shape)
        else:
            flag_gems.ops.upsample_nearest2d_backward_grad_input(
                grad_output, size, shape, **kwargs
            )


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_requires_positive_scales(scale, variant):
    # This is deliberately stricter than ATen's historical nonpositive defaults.
    grad_output = _tensor((1, 1, 4, 6), torch.float32)
    with pytest.raises(RuntimeError, match="scales"):
        if variant == "default":
            flag_gems.ops.upsample_nearest2d_backward(
                grad_output, (4, 6), (1, 1, 2, 3), scale, None
            )
        else:
            flag_gems.ops.upsample_nearest2d_backward_grad_input(
                grad_output,
                (4, 6),
                (1, 1, 2, 3),
                None,
                scale,
                grad_input=torch.empty((1, 1, 2, 3), device=flag_gems.device),
            )


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_deterministic(variant):
    grad_output = _tensor((2, 3, 37, 41), torch.float32, "strided")
    kwargs = (
        {}
        if variant == "default"
        else {"grad_input": _tensor((2, 3, 13, 17), torch.float32, "transpose")}
    )
    expected = torch.ops.aten.upsample_nearest2d_backward.default(
        grad_output, (37, 41), (2, 3, 13, 17)
    )
    name = "upsample_nearest2d_backward" + (
        "_grad_input" if variant == "grad_input" else ""
    )
    function = getattr(flag_gems.ops, name)
    first = function(grad_output, (37, 41), (2, 3, 13, 17), **kwargs).clone()
    second = function(grad_output, (37, 41), (2, 3, 13, 17), **kwargs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, expected)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("variant", ["default", "grad_input"])
@pytest.mark.parametrize("dtype", DTYPES[1:3])
def test_upsample_nearest2d_backward_accumulates_fp32(variant, dtype):
    grad_output = torch.ones((1, 1, 64, 64), dtype=dtype, device=flag_gems.device)
    expected = torch.full((1, 1, 1, 1), 4096, dtype=dtype)
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            grad_output, (64, 64), (1, 1, 1, 1)
        )
    else:
        output = torch.empty((1, 1, 1, 1), dtype=dtype, device=flag_gems.device)
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            grad_output, (64, 64), (1, 1, 1, 1), grad_input=output
        )
        assert actual is output
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.skipif(
    (not utils.fp64_is_supported and flag_gems.vendor_name != "mthreads")
    or flag_gems.vendor_name == "iluvatar",
    reason="Backend does not support FP64",
)
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_preserves_fp64(variant):
    values = (1 + torch.arange(12, dtype=torch.float64) * 1e-9).reshape(1, 1, 3, 4)
    grad_output = values.to(flag_gems.device)
    expected = values.sum().reshape(1, 1, 1, 1)
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            grad_output, (3, 4), (1, 1, 1, 1)
        )
    else:
        output = torch.empty((1, 1, 1, 1), dtype=torch.float64, device=flag_gems.device)
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            grad_output, (3, 4), (1, 1, 1, 1), grad_input=output
        )
        assert actual is output
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-13)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Native CANN backward does not support uint8",
)
@pytest.mark.parametrize("variant", ["default", "grad_input"])
@pytest.mark.parametrize("case", CASES + SCALE_CASES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "strided"])
def test_upsample_nearest2d_backward_uint8(variant, case, layout):
    shape, size, sh, sw = case
    grad_output = _tensor((*shape[:2], *size), torch.uint8, layout)
    output = _tensor(shape, torch.uint8, layout)
    original_stride = output.stride()
    if utils.TO_CPU:
        if flag_gems.vendor_name in (
            "nvidia",
            "ascend",
            "mthreads",
            "hygon",
            "iluvatar",
        ):
            expected = _ceil_cpu_reference(grad_output, size, shape, sh, sw)
        else:
            # CPU has no uint8 backward overload. Its FP64 legacy mapping
            # provides an exact integer-sum oracle before uint8 conversion.
            expected = torch.ops.aten.upsample_nearest2d_backward.default(
                grad_output.cpu().to(torch.float64), size, shape, sh, sw
            )
        expected = expected.to(torch.uint8)
    else:
        native = getattr(torch.ops.aten.upsample_nearest2d_backward, variant)
        native_kwargs = (
            {}
            if variant == "default"
            else {
                "grad_input": torch.empty_strided(
                    shape, original_stride, device=flag_gems.device, dtype=torch.uint8
                )
            }
        )
        expected = native(grad_output, size, shape, sh, sw, **native_kwargs).clone()
        if flag_gems.vendor_name == "mthreads" and tuple(shape[-2:]) == tuple(size):
            expected = grad_output
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            grad_output, size, shape, sh, sw
        )
    else:
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            grad_output, size, shape, sh, sw, grad_input=output
        )
        assert actual is output
        assert actual.stride() == original_stride
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.skipif(
    utils.QUICK_MODE, reason="Large address test requires an 8 GiB sparse allocation"
)
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_int64_address(variant):
    # Only eight values are touched, but the second row is more than 2^31
    # elements away. This exercises real 64-bit device pointer arithmetic.
    stride = 2**31 + 17
    grad_output = torch.empty_strided(
        (1, 1, 2, 4),
        (2 * stride, 2 * stride, stride, 1),
        device=flag_gems.device,
        dtype=torch.float32,
    )
    values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
    grad_output.copy_(values.to(flag_gems.device))
    expected = torch.ops.aten.upsample_nearest2d_backward.default(
        values, (2, 4), (1, 1, 2, 2)
    )
    if variant == "default":
        actual = flag_gems.ops.upsample_nearest2d_backward(
            grad_output, (2, 4), (1, 1, 2, 2)
        )
    else:
        output = torch.empty_strided(
            (1, 1, 2, 2),
            (2 * stride, 2 * stride, stride, 1),
            device=flag_gems.device,
            dtype=torch.float32,
        )
        actual = flag_gems.ops.upsample_nearest2d_backward_grad_input(
            grad_output, (2, 4), (1, 1, 2, 2), grad_input=output
        )
        assert actual is output
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar", reason="Iluvatar FP64 limitation"
)
@pytest.mark.parametrize("variant", ["default", "grad_input"])
def test_upsample_nearest2d_backward_iluvatar_rejects_fp64(variant):
    source = torch.empty((1, 1, 4, 6), device=flag_gems.device, dtype=torch.float64)
    kwargs = (
        {}
        if variant == "default"
        else {
            "grad_input": torch.empty(
                (1, 1, 2, 3), device=flag_gems.device, dtype=torch.float64
            )
        }
    )
    name = "upsample_nearest2d_backward" + (
        "_grad_input" if variant == "grad_input" else ""
    )
    with pytest.raises(RuntimeError, match="Iluvatar does not support FP64"):
        getattr(flag_gems.ops, name)(source, (4, 6), (1, 1, 2, 3), **kwargs)
