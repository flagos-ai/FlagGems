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

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

HAS_NATIVE_LANCZOS = hasattr(torch.ops.aten, "_upsample_lanczos2d_aa")

if QUICK_MODE:
    SHAPES = [(1, 2, 7, 9)]
    SCALES = [(1.5, 0.75)]
    DTYPES = [torch.float32]
else:
    SHAPES = [(1, 1, 3, 5), (2, 3, 17, 19), (4, 8, 32, 48)]
    SCALES = [(2.0, 2.0), (0.5, 0.75)]
    DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _scale(input_size, output_size, align_corners, explicit_scale):
    if align_corners:
        return (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    if explicit_scale is not None and explicit_scale > 0:
        return 1.0 / explicit_scale
    return input_size / output_size


def _weight_matrix(input_size, output_size, align_corners, explicit_scale, dtype):
    scale = _scale(input_size, output_size, align_corners, explicit_scale)
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    output_index = torch.arange(output_size, dtype=dtype)
    input_index = torch.arange(input_size, dtype=dtype)
    center = scale * (output_index + 0.5)
    index_min = (center - support + 0.5).to(torch.int64).clamp_min(0)
    index_max = (center + support + 0.5).to(torch.int64).clamp_max(input_size)
    distance = (input_index[None, :] - center[:, None] + 0.5) * invscale
    abs_distance = distance.abs()
    pix = math.pi * distance
    sinc = torch.where(abs_distance == 0, 1.0, torch.sin(pix) / pix)
    sinc_three = torch.where(abs_distance == 0, 1.0, torch.sin(pix / 3.0) / (pix / 3.0))
    valid = (
        (input_index[None, :] >= index_min[:, None])
        & (input_index[None, :] < index_max[:, None])
        & (abs_distance < 3.0)
    )
    weights = torch.where(valid, sinc * sinc_three, 0.0)
    return weights / weights.sum(dim=1, keepdim=True)


def _reference(
    input,
    output_size,
    align_corners=False,
    scales_h=None,
    scales_w=None,
):
    output_h, output_w = output_size
    opmath_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    weights_w = _weight_matrix(
        input.shape[-1], output_w, align_corners, scales_w, opmath_dtype
    ).to(input.device)
    weights_h = _weight_matrix(
        input.shape[-2], output_h, align_corners, scales_h, opmath_dtype
    ).to(input.device)
    value = (
        input.to(opmath_dtype).unsqueeze(-2) * weights_w.view(1, 1, 1, output_w, -1)
    ).sum(dim=-1)
    if input.dtype == torch.uint8:
        value = value.round().clamp(0, 255).to(torch.uint8)
    else:
        value = value.to(input.dtype)
    value = (
        value.to(opmath_dtype).unsqueeze(-3) * weights_h.view(1, 1, output_h, -1, 1)
    ).sum(dim=-2)
    if input.dtype == torch.uint8:
        return value.round().clamp(0, 255).to(torch.uint8)
    return value.to(input.dtype)


def _assert_close(result, reference, dtype):
    if dtype == torch.uint8:
        torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=0, atol=2)
        return
    atol = {
        torch.float16: 2e-2,
        torch.bfloat16: 8e-2,
        torch.float32: 2e-4,
        torch.float64: 1e-10,
    }[dtype]
    rtol = {
        torch.float16: 1e-3,
        torch.bfloat16: 2e-2,
        torch.float32: 1e-5,
        torch.float64: 1e-12,
    }[dtype]
    torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=rtol, atol=atol)


@pytest.mark.upsample_lanczos2d_aa
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_lanczos2d_aa(dtype, shape, scale, align_corners):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    reference_input = utils.to_reference(input)
    output_size = tuple(int(shape[i + 2] * scale[i]) for i in range(2))
    reference = _reference(reference_input, output_size, align_corners)
    result = flag_gems._upsample_lanczos2d_aa(input, output_size, align_corners)
    _assert_close(result, reference, dtype)


@pytest.mark.upsample_lanczos2d_aa
def test_upsample_lanczos2d_aa_uint8_and_layout():
    input = torch.randint(
        0, 256, (1, 3, 11, 13), dtype=torch.uint8, device=flag_gems.device
    ).contiguous(memory_format=torch.channels_last)
    reference = _reference(utils.to_reference(input), (7, 19))
    result = flag_gems._upsample_lanczos2d_aa(input, (7, 19))
    _assert_close(result, reference, torch.uint8)
    assert result.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.upsample_lanczos2d_aa
def test_upsample_lanczos2d_aa_transposed_input():
    input = torch.randn((1, 2, 9, 7), device=flag_gems.device).transpose(-1, -2)
    reference = _reference(utils.to_reference(input), (13, 11))
    result = flag_gems._upsample_lanczos2d_aa(input, (13, 11))
    _assert_close(result, reference, torch.float32)
    assert result.is_contiguous()


@pytest.mark.upsample_lanczos2d_aa_out
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_upsample_lanczos2d_aa_out(noncontiguous):
    input = torch.randn((1, 2, 7, 9), device=flag_gems.device)
    reference = _reference(utils.to_reference(input), (11, 13))
    if noncontiguous:
        out = torch.empty((1, 2, 13, 11), device=flag_gems.device).transpose(-1, -2)
    else:
        out = torch.empty(0, device=flag_gems.device)
    result = flag_gems._upsample_lanczos2d_aa_out(input, (11, 13), out=out)
    assert result is out
    _assert_close(result, reference, torch.float32)


@pytest.mark.upsample_lanczos2d_aa_vec
@pytest.mark.parametrize("use_scale_factors", [False, True])
def test_upsample_lanczos2d_aa_vec(use_scale_factors):
    input = torch.randn((1, 2, 8, 10), device=flag_gems.device)
    if use_scale_factors:
        args = (None, False, (1.5, 0.7))
        output_size = (12, 7)
        scales = (1.5, 0.7)
    else:
        args = ((12, 7), False, None)
        output_size = (12, 7)
        scales = (None, None)
    reference = _reference(
        utils.to_reference(input), output_size, False, scales[0], scales[1]
    )
    result = flag_gems._upsample_lanczos2d_aa_vec(input, *args)
    _assert_close(result, reference, torch.float32)


@pytest.mark.upsample_lanczos2d_aa_vec
@pytest.mark.parametrize(
    "output_size,scale_factors", [(None, None), ((12, 7), (1.5, 0.7))]
)
def test_upsample_lanczos2d_aa_vec_requires_one_size(output_size, scale_factors):
    input = torch.randn((1, 2, 8, 10), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._upsample_lanczos2d_aa_vec(input, output_size, False, scale_factors)


@pytest.mark.upsample_lanczos2d_aa
@pytest.mark.skipif(
    not HAS_NATIVE_LANCZOS,
    reason="ATen Lanczos schema was added after the local PyTorch 2.9 build",
)
def test_upsample_lanczos2d_aa_dispatch():
    input = torch.randn((1, 2, 7, 9), device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_lanczos2d_aa(
            input, (11, 13), False, None, None
        )
    reference = _reference(utils.to_reference(input), (11, 13))
    _assert_close(result, reference, torch.float32)
