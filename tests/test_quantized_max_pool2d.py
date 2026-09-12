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

from . import accuracy_utils as utils

# quantized_max_pool2d operates on per-tensor quantized tensors (torch.quint8 /
# torch.qint8). PyTorch's native quantized pooling kernel is CPU-only in this
# build, so the inputs are kept on CPU; the FlagGems kernel moves the integer
# representation onto the accelerator internally and returns a CPU quantized
# tensor. We parametrize over the quantized dtypes rather than utils.FLOAT_DTYPES
# and keep the input dtype fixed at the quantized scheme.
QUANT_DTYPES = [torch.quint8, torch.qint8]


# (shape, kernel_size, stride, padding, dilation, ceil_mode)
QUANT_MAX_POOL2D_CONFIGS = [
    # Classic 2x2 pooling with default stride
    ((2, 4, 16, 16), (2, 2), [], 0, 1, False),
    # 3x3 kernel, stride 2, padding 1 (ResNet style)
    ((4, 8, 32, 32), 3, 2, 1, 1, False),
    # Non-square kernel/stride/padding
    ((2, 3, 28, 28), (3, 5), (1, 2), (1, 0), 1, False),
    # Dilation
    ((1, 4, 20, 20), 2, 1, 0, 2, False),
    # ceil_mode
    ((2, 4, 15, 15), 3, 2, 1, 1, True),
    # Larger spatial dims
    ((1, 16, 56, 56), 3, 2, 1, 1, False),
    # Asymmetric padding
    ((2, 2, 16, 20), 2, 2, (1, 0), 1, False),
]


def _make_quant_tensor(shape, dtype, scale, zero_point):
    if dtype == torch.quint8:
        data = torch.rand(shape) * 255.0
    else:
        data = torch.randint(-128, 128, shape).float()
    return torch.quantize_per_tensor(data, scale, zero_point, dtype)


@pytest.mark.quantized_max_pool2d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANT_MAX_POOL2D_CONFIGS,
)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", [0.1, 1.5])
@pytest.mark.parametrize("zero_point", [0, -128])
def test_quantized_max_pool2d(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype, scale, zero_point
):
    # Quantized inputs stay on CPU: the native CUDA quantized pooling kernel is not
    # supported for QUInt8/QInt8 in this PyTorch build, so dispatching a CUDA tensor
    # would fall through to the broken native path. The FlagGems implementation
    # moves the integer storage onto the accelerator internally.
    # Pick a zero_point valid for the dtype (qint8 in [-128, 127], quint8 in [0, 255]).
    if dtype == torch.quint8:
        zero_point = abs(zero_point) + 3
    res_inp = _make_quant_tensor(shape, dtype, scale, zero_point)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantized_max_pool2d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    # GEMS direct call: the kernel pools the integer representation on the
    # accelerator and returns a CPU quantized tensor with the input's scale/zp.
    res_out = flag_gems.quantized_max_pool2d(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # The pooling preserves the quantization parameters exactly; compare the
    # dequantized (float32) outputs as well as the raw integer representation.
    assert res_out.dtype == dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    utils.gems_assert_close(res_out.dequantize(), ref_out.dequantize(), torch.float32)


@pytest.mark.quantized_max_pool2d
def test_quantized_max_pool2d_zero_batch():
    # A zero-element batch produces an empty output tensor; ensure the kernel and
    # the quantized-tensor bookkeeping handle the zero-element path.
    res_inp = _make_quant_tensor((0, 3, 8, 8), torch.quint8, 0.5, 3)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantized_max_pool2d(ref_inp, [2, 2])
    res_out = flag_gems.quantized_max_pool2d(res_inp, [2, 2])
    assert res_out.shape == ref_out.shape
    assert res_out.numel() == 0
    assert res_out.dtype == torch.quint8


@pytest.mark.quantized_max_pool2d
def test_quantized_max_pool2d_extreme_values():
    # All-equal and saturated quantized values exercise the max-reduction edges.
    for dtype in QUANT_DTYPES:
        scale = 0.1
        zero_point = 128 if dtype == torch.quint8 else 0
        # All zeros (dequant = -zero_point*scale)
        data = torch.zeros((2, 2, 8, 8))
        res_inp = torch.quantize_per_tensor(data, scale, zero_point, dtype)
        ref_inp = utils.to_reference(res_inp)

        ref_out = torch.quantized_max_pool2d(ref_inp, [2, 2])
        res_out = flag_gems.quantized_max_pool2d(res_inp, [2, 2])
        utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())

        # Saturation: all max values
        if dtype == torch.quint8:
            data = torch.full((2, 2, 8, 8), 255.0)
        else:
            data = torch.full((2, 2, 8, 8), 127.0)
        res_inp = torch.quantize_per_tensor(data, scale, zero_point, dtype)
        ref_inp = utils.to_reference(res_inp)
        ref_out = torch.quantized_max_pool2d(ref_inp, [2, 2])
        res_out = flag_gems.quantized_max_pool2d(res_inp, [2, 2])
        utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark.quantized_max_pool2d_out
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANT_MAX_POOL2D_CONFIGS,
)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", [0.1, 1.5])
@pytest.mark.parametrize("zero_point", [0, -128])
def test_quantized_max_pool2d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype, scale, zero_point
):
    if dtype == torch.quint8:
        zero_point = abs(zero_point) + 3
    res_inp = _make_quant_tensor(shape, dtype, scale, zero_point)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantized_max_pool2d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # Pre-allocate a quantized ``out`` tensor via the public API with the input's
    # scale/zero_point; the FlagGems out kernel overwrites its integer storage.
    out_tensor = torch.quantize_per_tensor(
        torch.zeros(ref_out.shape), ref_out.q_scale(), ref_out.q_zero_point(), dtype
    )
    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        out=out_tensor,
    )

    assert res_r is out_tensor
    assert res_r.dtype == dtype
    assert res_r.q_scale() == ref_out.q_scale()
    assert res_r.q_zero_point() == ref_out.q_zero_point()
    assert res_r.shape == ref_out.shape
    utils.gems_assert_equal(res_r.int_repr(), ref_out.int_repr())
    utils.gems_assert_close(res_r.dequantize(), ref_out.dequantize(), torch.float32)
