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

from .accuracy_utils import gems_assert_close, to_reference

QUANT_SHAPES = [(4, 4), (16, 32), (2, 3, 4), (8, 16, 32)]


@pytest.mark.fake_quantize_per_channel_affine
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("quant_min, quant_max", [(0, 255), (-128, 127)])
def test_accuracy_fake_quantize_per_channel_affine(
    shape, axis, dtype, quant_min, quant_max
):
    if axis >= len(shape):
        pytest.skip(f"axis {axis} >= ndim {len(shape)}")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n_channels = shape[axis]
    scale = (
        torch.rand(n_channels, dtype=torch.float32, device=flag_gems.device) * 0.1
        + 0.01
    )
    zero_point = torch.randint(
        quant_min,
        quant_max + 1,
        (n_channels,),
        dtype=torch.int32,
        device=flag_gems.device,
    )

    ref_inp = to_reference(inp)
    ref_scale = to_reference(scale)
    ref_zero_point = to_reference(zero_point)

    ref_out = torch.fake_quantize_per_channel_affine(
        ref_inp, ref_scale, ref_zero_point, axis, quant_min, quant_max
    )

    with flag_gems.use_gems():
        res_out = torch.fake_quantize_per_channel_affine(
            inp, scale, zero_point, axis, quant_min, quant_max
        )

    gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.fake_quantize_per_channel_affine
@pytest.mark.parametrize("shape", [(2, 3, 4, 5)])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_accuracy_fake_quantize_per_channel_affine_multi_dim(shape, axis):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    n_channels = shape[axis]
    scale = (
        torch.rand(n_channels, dtype=torch.float32, device=flag_gems.device) * 0.1
        + 0.01
    )
    zero_point = torch.randint(
        0, 255, (n_channels,), dtype=torch.int32, device=flag_gems.device
    )

    ref_inp = to_reference(inp)
    ref_scale = to_reference(scale)
    ref_zero_point = to_reference(zero_point)

    ref_out = torch.fake_quantize_per_channel_affine(
        ref_inp, ref_scale, ref_zero_point, axis, 0, 255
    )

    with flag_gems.use_gems():
        res_out = torch.fake_quantize_per_channel_affine(
            inp, scale, zero_point, axis, 0, 255
        )

    gems_assert_close(res_out, ref_out, dtype=torch.float32)


@pytest.mark.fake_quantize_per_channel_affine
def test_accuracy_fake_quantize_per_channel_affine_half_to_even():
    inp = torch.tensor(
        [[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    scale = torch.ones(8, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.zeros(8, dtype=torch.int32, device=flag_gems.device)
    ref_out = torch.fake_quantize_per_channel_affine(
        to_reference(inp), to_reference(scale), to_reference(zero_point), 1, -128, 127
    )

    with flag_gems.use_gems():
        res_out = torch.fake_quantize_per_channel_affine(
            inp, scale, zero_point, 1, -128, 127
        )

    gems_assert_close(res_out, ref_out, dtype=torch.float32)


@pytest.mark.fake_quantize_per_channel_affine
def test_accuracy_fake_quantize_per_channel_affine_empty():
    inp = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_gems.device)
    scale = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.fake_quantize_per_channel_affine(
            inp, scale, zero_point, 1, 0, 255
        )

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
