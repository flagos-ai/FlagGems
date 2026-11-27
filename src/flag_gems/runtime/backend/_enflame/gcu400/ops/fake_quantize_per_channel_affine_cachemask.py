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

import logging

import torch

from .fake_quantize_per_channel_affine import launch_fake_quantize_per_channel_affine

logger = logging.getLogger(__name__)


def _fake_quantize_per_channel_affine_cachemask_impl(
    input,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    output=None,
    cachemask=None,
):
    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    if output is None:
        output = torch.empty_like(input)
    if cachemask is None:
        cachemask = torch.empty_like(input, dtype=torch.bool)

    if input.numel() == 0:
        return output, cachemask

    launch_fake_quantize_per_channel_affine(
        input, scale, zero_point, output, cachemask, axis, quant_min, quant_max
    )
    return output, cachemask


def fake_quantize_per_channel_affine_cachemask(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input, scale, zero_point, axis, quant_min, quant_max
    )


def fake_quantize_per_channel_affine_cachemask_out(
    input, scale, zero_point, axis, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_OUT")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        output=out0,
        cachemask=out1,
    )
