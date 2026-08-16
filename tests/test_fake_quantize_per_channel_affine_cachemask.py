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

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

ATEN_OP = torch.ops.aten.fake_quantize_per_channel_affine_cachemask


@pytest.mark.fake_quantize_per_channel_affine_cachemask
@pytest.mark.parametrize("shape, axis", [((4, 7), 0), ((4, 7), 1), ((2, 3, 5), 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("quant_min, quant_max", [(0, 255), (-128, 127)])
def test_accuracy_fake_quantize_per_channel_affine_cachemask(
    shape, axis, dtype, quant_min, quant_max
):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 20
    channels = shape[axis]
    scale = torch.rand(channels, dtype=torch.float32, device=flag_gems.device) + 0.1
    zero_point = torch.randint(
        quant_min,
        quant_max + 1,
        (channels,),
        dtype=torch.int32,
        device=flag_gems.device,
    )

    ref_output, ref_mask = ATEN_OP(
        to_reference(input),
        to_reference(scale),
        to_reference(zero_point),
        axis,
        quant_min,
        quant_max,
    )
    with flag_gems.use_gems():
        output, mask = ATEN_OP(input, scale, zero_point, axis, quant_min, quant_max)

    gems_assert_close(output, ref_output, dtype=dtype)
    gems_assert_equal(mask, ref_mask)
    assert output.dtype == dtype
    assert mask.dtype == torch.bool


@pytest.mark.fake_quantize_per_channel_affine_cachemask
def test_accuracy_fake_quantize_per_channel_affine_cachemask_half_to_even():
    input = torch.tensor(
        [[-129.5, -128.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5, 127.5]],
        device=flag_gems.device,
    )
    scale = torch.ones(10, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.zeros(10, dtype=torch.int32, device=flag_gems.device)

    ref_output, ref_mask = ATEN_OP(
        to_reference(input), to_reference(scale), to_reference(zero_point), 1, -128, 127
    )
    with flag_gems.use_gems():
        output, mask = ATEN_OP(input, scale, zero_point, 1, -128, 127)

    gems_assert_equal(output, ref_output)
    gems_assert_equal(mask, ref_mask)


@pytest.mark.fake_quantize_per_channel_affine_cachemask
def test_accuracy_fake_quantize_per_channel_affine_cachemask_noncontiguous():
    input = torch.randn((2, 3, 5), device=flag_gems.device).transpose(0, 2)
    scale = torch.tensor([0.1, 0.2, 0.3], device=flag_gems.device)
    zero_point = torch.tensor([0, 5, -5], dtype=torch.int32, device=flag_gems.device)

    ref_output, ref_mask = ATEN_OP(
        to_reference(input), to_reference(scale), to_reference(zero_point), 1, -128, 127
    )
    with flag_gems.use_gems():
        output, mask = ATEN_OP(input, scale, zero_point, 1, -128, 127)

    gems_assert_close(output, ref_output, dtype=torch.float32)
    gems_assert_equal(mask, ref_mask)


@pytest.mark.fake_quantize_per_channel_affine_cachemask_out
def test_accuracy_fake_quantize_per_channel_affine_cachemask_out():
    input = torch.randn((2, 3, 5), device=flag_gems.device) * 20
    scale = torch.tensor([0.1, 0.2, 0.3], device=flag_gems.device)
    zero_point = torch.tensor([0, 5, -5], dtype=torch.int32, device=flag_gems.device)
    ref_output, ref_mask = ATEN_OP(
        to_reference(input), to_reference(scale), to_reference(zero_point), 1, -128, 127
    )
    out0 = torch.empty_like(input)
    out1 = torch.empty_like(input, dtype=torch.bool)

    with flag_gems.use_gems():
        output, mask = ATEN_OP.out(
            input,
            scale,
            zero_point,
            1,
            -128,
            127,
            out0=out0,
            out1=out1,
        )

    assert output is out0
    assert mask is out1
    gems_assert_close(output, ref_output, dtype=torch.float32)
    gems_assert_equal(mask, ref_mask)


@pytest.mark.fake_quantize_per_channel_affine_cachemask
def test_accuracy_fake_quantize_per_channel_affine_cachemask_empty():
    input = torch.empty((2, 0, 3), device=flag_gems.device)
    scale = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    zero_point = torch.empty(0, dtype=torch.int32, device=flag_gems.device)

    with flag_gems.use_gems():
        output, mask = ATEN_OP(input, scale, zero_point, 1, 0, 255)

    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert mask.shape == input.shape
    assert mask.dtype == torch.bool
