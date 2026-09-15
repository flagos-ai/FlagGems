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
from collections import OrderedDict

import pytest
import torch

from flag_gems.ops._upsample_lanczos2d_aa_backward import (
    _upsample_lanczos2d_aa_backward,
    _upsample_lanczos2d_aa_backward_out,
)

from . import base

_WEIGHT_CACHE = OrderedDict()


def _weight_matrix(input_size, output_size, device):
    key = (input_size, output_size, str(device))
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _WEIGHT_CACHE.move_to_end(key)
        return cached

    scale = input_size / output_size
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    output_index = torch.arange(output_size, dtype=torch.float32)
    input_index = torch.arange(input_size, dtype=torch.float32)
    center = scale * (output_index + 0.5)
    index_min = (center - support + 0.5).to(torch.int64).clamp_min(0)
    index_max = (center + support + 0.5).to(torch.int64).clamp_max(input_size)
    distance = (input_index[None, :] - center[:, None] + 0.5) * invscale
    pix = math.pi * distance
    sinc = torch.where(distance == 0.0, 1.0, torch.sin(pix) / pix)
    sinc_three = torch.where(distance == 0.0, 1.0, torch.sin(pix / 3.0) / (pix / 3.0))
    valid = (
        (input_index[None, :] >= index_min[:, None])
        & (input_index[None, :] < index_max[:, None])
        & (distance.abs() < 3.0)
    )
    weight = torch.where(valid, sinc * sinc_three, 0.0)
    weight /= weight.sum(dim=1, keepdim=True)
    weight = weight.to(device)
    if len(_WEIGHT_CACHE) >= 32:
        _WEIGHT_CACHE.popitem(last=False)
    _WEIGHT_CACHE[key] = weight
    return weight


def _composite_reference(
    grad_output,
    output_size,
    input_size,
    align_corners=False,
    scales_h=None,
    scales_w=None,
):
    # PyTorch 2.9 predates the native Lanczos schema. This cached, independent
    # matrix formulation is the GPU benchmark baseline until that schema ships.
    n, c, input_h, input_w = input_size
    output_h, output_w = output_size
    weight_w = _weight_matrix(input_w, output_w, grad_output.device)
    weight_h = _weight_matrix(input_h, output_h, grad_output.device)
    temp = torch.einsum("ncho,oi->nchi", grad_output.float(), weight_w)
    result = torch.einsum("oh,ncow->nchw", weight_h, temp)
    return result.to(grad_output.dtype)


def _composite_reference_out(
    grad_output,
    output_size,
    input_size,
    align_corners=False,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    grad_input.copy_(
        _composite_reference(
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_h,
            scales_w,
        )
    )
    return grad_input


class UpsampleLanczos2dAaBackwardBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Includes practical small, channel-heavy, and larger image workloads.
        self._configs = [
            (1, 1, 8, 8, 16, 16),
            (1, 3, 32, 48, 16, 24),
            (4, 64, 64, 64, 32, 32),
            (8, 32, 96, 128, 48, 64),
        ]

    def get_input_iter(self, dtype):
        for n, c, input_h, input_w, output_h, output_w in self._configs:
            grad = torch.randn(
                (n, c, output_h, output_w), device=self.device, dtype=dtype
            )
            yield (
                grad,
                (output_h, output_w),
                (n, c, input_h, input_w),
                False,
                None,
                None,
            )

    def get_tflops(self, op, *args, **kwargs):
        grad_output, output_size, input_size = args[:3]
        filter_area = 36
        return (
            grad_output.shape[0]
            * grad_output.shape[1]
            * math.prod(output_size)
            * filter_area
            * 2
        )


class UpsampleLanczos2dAaBackwardOutBenchmark(UpsampleLanczos2dAaBackwardBenchmark):
    def get_input_iter(self, dtype):
        for args in super().get_input_iter(dtype):
            grad_output, output_size, input_size, *rest = args
            grad_input = torch.empty(input_size, device=self.device, dtype=dtype)
            yield grad_output, output_size, input_size, *rest, {
                "grad_input": grad_input
            }


@pytest.mark.upsample_lanczos2d_aa_backward
def test_upsample_lanczos2d_aa_backward():
    bench = UpsampleLanczos2dAaBackwardBenchmark(
        op_name="upsample_lanczos2d_aa_backward",
        torch_op=_composite_reference,
        gems_op=_upsample_lanczos2d_aa_backward,
        # The upstream CPU schema supports float/double; benchmark the two
        # primary accelerator dtypes while accuracy tests retain BF16/FP64.
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.upsample_lanczos2d_aa_backward_grad_input
def test_upsample_lanczos2d_aa_backward_grad_input():
    bench = UpsampleLanczos2dAaBackwardOutBenchmark(
        op_name="upsample_lanczos2d_aa_backward_grad_input",
        torch_op=_composite_reference_out,
        gems_op=_upsample_lanczos2d_aa_backward_out,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
