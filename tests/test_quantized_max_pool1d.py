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

# quantized_max_pool1d operates on quantized tensors (torch.quint8 by default).
# The reference implementation only exists on the CPU backend, so we always
# compare the CUDA FlagGems result against the CPU PyTorch result.
QUANT_DTYPES = [torch.quint8, torch.qint8]

# Pooling is applied along the last dimension. Shapes cover both 2D (N, L)
# and 3D (N, C, L) inputs as well as a few larger reduce dimensions.
POOL_SHAPES = [
    (4, 16),  # 2D input (N, L)
    (2, 3, 16),  # 3D input (N, C, L)
    (1, 8),  # minimal 2D
    (1, 1, 8),  # minimal 3D
    (32, 50257),  # large reduce dim, sequence-like
    (8, 3, 8192),  # large reduce dim, channel-like
]

# (kernel_size, stride, padding, dilation, ceil_mode)
POOL_CONFIGS = [
    (2, 2, 0, 1, False),
    (3, 2, 1, 1, False),
    (3, 2, 1, 1, True),  # ceil_mode
    (2, 1, 0, 1, False),  # stride=1, no padding
    (2, 1, 0, 2, False),  # dilation
    (5, 3, 2, 1, False),  # larger kernel
    (3, 2, 1, 1, True),  # ceil_mode + padding
]


def _make_quantized(shape, scale, zero_point, dtype, device):
    fp = torch.randn(shape, device="cpu").clamp_(-2, 2)
    return torch.quantize_per_tensor(
        fp, scale=scale, zero_point=zero_point, dtype=dtype
    ).to(device)


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("shape", POOL_SHAPES)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode", POOL_CONFIGS
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d(
    shape, kernel_size, stride, padding, dilation, ceil_mode, in_dtype
):
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_inp = res_inp.to("cpu")

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    res_out = flag_gems.quantized_max_pool1d(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    assert res_out.dtype == in_dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # Compare dequantized values; pool over the last dim so reduce_dim=1.
    # The native op only exists on CPU, so the reference runs on CPU and the
    # FlagGems result is moved to CPU before comparison.
    utils.gems_assert_close(
        res_out.to("cpu").dequantize(),
        ref_out.dequantize(),
        dtype=torch.float32,
        reduce_dim=1,
    )


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("shape", POOL_SHAPES)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode", POOL_CONFIGS
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode, in_dtype
):
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_inp = res_inp.to("cpu")

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    scale = float(res_inp.q_scale())
    zero_point = int(res_inp.q_zero_point())
    out_shape = ref_out.shape
    res_out = torch.quantize_per_tensor(
        torch.zeros(out_shape), scale=scale, zero_point=zero_point, dtype=in_dtype
    ).to(flag_gems.device)

    got = flag_gems.quantized_max_pool1d_out(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        out=res_out,
    )

    # The .out variant must return the same tensor that was passed in.
    assert got.data_ptr() == res_out.data_ptr()
    assert got.dtype == in_dtype
    utils.gems_assert_close(
        res_out.to("cpu").dequantize(),
        ref_out.dequantize(),
        dtype=torch.float32,
        reduce_dim=1,
    )
