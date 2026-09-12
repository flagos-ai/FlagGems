# Copyright 2026 FlagOS Contributors.
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

# Per-tensor quantized max-pool operates on the uint8 integer representation
# of a quint8 tensor. The result is exact (no floating-point arithmetic), so
# the tests compare integer representations with exact equality.
#
# PyTorch ships no native QuantizedCUDA kernel for ``quantized_max_pool3d``, so
# the reference is always evaluated on the CPU (QuantizedCPU backend), while the
# FlagGems kernel runs on the GPU.
QDTYPE = torch.quint8
SCALE = 0.05
ZERO_POINT = 100


def _make_qinput(shape, device, scale=SCALE, zero_point=ZERO_POINT):
    x = torch.randn(shape, device=device)
    return torch.quantize_per_tensor(
        x, scale=scale, zero_point=zero_point, dtype=QDTYPE
    )


# (shape, kernel_size, stride, padding, dilation, ceil_mode)
QUANTIZED_MAXPOOL3D_CONFIGS = [
    # Classic cubic kernel, stride 2, padding 1
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel and stride
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    # Typical 3D CNN shape
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    # Small input
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    # Large batch, stride 1
    ((4, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


def _assert_qpool_equal(res_out, ref_out):
    """Compare two quint8 max-pool results for exact equality."""
    assert (
        res_out.shape == ref_out.shape
    ), f"shape mismatch: res={tuple(res_out.shape)} ref={tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype == QDTYPE
    # scale / zero_point must be preserved from the input
    assert (
        abs(res_out.q_scale() - ref_out.q_scale()) < 1e-9
    ), f"scale mismatch: res={res_out.q_scale()} ref={ref_out.q_scale()}"
    assert (
        res_out.q_zero_point() == ref_out.q_zero_point()
    ), f"zero_point mismatch: res={res_out.q_zero_point()} ref={ref_out.q_zero_point()}"
    # Max-pool over the quantized integers is exact, so compare integer reprs.
    res_int = res_out.to("cpu").int_repr()
    ref_int = ref_out.int_repr()
    utils.gems_assert_equal(res_int, ref_int)


def _pool_args(kernel_size, stride, padding, dilation, ceil_mode):
    return dict(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANTIZED_MAXPOOL3D_CONFIGS,
)
def test_quantized_max_pool3d(shape, kernel_size, stride, padding, dilation, ceil_mode):
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    res_out = flag_gems.quantized_max_pool3d(
        qx, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d
@pytest.mark.parametrize("scale_zp", [(0.1, 128), (0.02, 50), (0.5, 30)])
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    [
        ((1, 3, 16, 16, 16), 3, 2, 1, 1, False),
        ((2, 8, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
        ((1, 4, 15, 15, 15), 3, 2, 1, 1, True),
    ],
)
def test_quantized_max_pool3d_quant_params(
    shape, kernel_size, stride, padding, dilation, ceil_mode, scale_zp
):
    """Verify the output preserves arbitrary quantization parameters."""
    scale, zero_point = scale_zp
    qx = _make_qinput(
        shape, device=flag_gems.device, scale=scale, zero_point=zero_point
    )
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    res_out = flag_gems.quantized_max_pool3d(
        qx, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d_out
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANTIZED_MAXPOOL3D_CONFIGS,
)
def test_quantized_max_pool3d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode
):
    """``.out`` variant writes into a pre-allocated quantized tensor."""
    qx = _make_qinput(shape, device=flag_gems.device)
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, **_pool_args(kernel_size, stride, padding, dilation, ceil_mode)
    )

    # Pre-allocate an out tensor with the expected (ref) shape and matching
    # quantization parameters.
    out_shape = ref_out.shape
    out_q = torch.quantize_per_tensor(
        torch.zeros(out_shape, dtype=torch.float32, device=flag_gems.device),
        ref_out.q_scale(),
        ref_out.q_zero_point(),
        QDTYPE,
    )

    res_out = flag_gems.quantized_max_pool3d_out(
        qx,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        out=out_q,
    )

    # The out variant must alias and return the provided tensor.
    assert res_out is out_q
    _assert_qpool_equal(res_out, ref_out)


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d_zero_input():
    """All-zero input must yield the zero_point for every pooled output."""
    # Small 4x4x4 volume so every element maps to a single 2x2x2 window.
    shape = (1, 1, 4, 4, 4)
    qx = torch.quantize_per_tensor(
        torch.zeros(shape, device=flag_gems.device),
        scale=SCALE,
        zero_point=ZERO_POINT,
        dtype=QDTYPE,
    )
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False
    )

    res_out = flag_gems.quantized_max_pool3d(
        qx, kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False
    )

    _assert_qpool_equal(res_out, ref_out)
    # Every output entry should equal the zero_point (all inputs map to it).
    assert bool((res_out.to("cpu").int_repr() == ZERO_POINT).all())


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d_saturated_input():
    """Saturated inputs (clamped to the quint8 range) pool to the clamped max."""
    # 6x6x6 volume gives 3x3x3 windows under a 2x2x2 kernel with stride 2.
    shape = (1, 2, 6, 6, 6)
    # Fill with values that clamp to both ends of the quint8 range.
    raw = torch.full(shape, 3.0, device=flag_gems.device)
    raw[0, 0, 0, 0, 0] = -3.0
    qx = torch.quantize_per_tensor(
        raw, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE
    )
    ref_inp = qx.to("cpu")

    ref_out = torch.quantized_max_pool3d(
        ref_inp, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
    )

    res_out = flag_gems.quantized_max_pool3d(
        qx, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
    )

    _assert_qpool_equal(res_out, ref_out)
