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

from flag_gems.ops.upsample_lanczos2d_aa import (
    _upsample_lanczos2d_aa,
    _upsample_lanczos2d_aa_out,
    _upsample_lanczos2d_aa_vec,
)

from . import base, consts

_WEIGHT_CACHE = {}


class UpsampleLanczosBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1, 8, 8),
            (1, 3, 32, 48),
            (4, 16, 64, 64),
            (8, 32, 96, 128),
            (4, 64, 128, 128),
        ]
        self.shape_desc = ["N", "C", "H", "W"]


def _weight_matrix(input_size, output_size, device):
    key = (input_size, output_size, str(device))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]
    scale = input_size / output_size
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    output_index = torch.arange(output_size, device=device, dtype=torch.float32)
    input_index = torch.arange(input_size, device=device, dtype=torch.float32)
    center = scale * (output_index + 0.5)
    index_min = (center - support + 0.5).to(torch.int64).clamp_min(0)
    index_max = (center + support + 0.5).to(torch.int64).clamp_max(input_size)
    distance = (input_index[None, :] - center[:, None] + 0.5) * invscale
    valid = (
        (input_index[None, :] >= index_min[:, None])
        & (input_index[None, :] < index_max[:, None])
        & (distance.abs() < 3.0)
    )
    weights = torch.where(valid, torch.sinc(distance) * torch.sinc(distance / 3), 0)
    weights /= weights.sum(dim=1, keepdim=True)
    _WEIGHT_CACHE[key] = weights
    return weights


def _composite_reference(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    # PyTorch 2.9 predates the native Lanczos schema. This cached composition is
    # used only as the benchmark baseline until a native CUDA kernel is available.
    output_h, output_w = output_size
    weights_w = _weight_matrix(input.shape[-1], output_w, input.device)
    weights_h = _weight_matrix(input.shape[-2], output_h, input.device)
    value = torch.einsum("nchi,oi->ncho", input.float(), weights_w).to(input.dtype)
    return torch.einsum("nciw,oi->ncow", value.float(), weights_h).to(input.dtype)


def _composite_reference_out(
    input,
    output_size,
    align_corners=False,
    scales_h=None,
    scales_w=None,
    *,
    out,
):
    out.copy_(
        _composite_reference(input, output_size, align_corners, scales_h, scales_w)
    )
    return out


def _composite_reference_vec(input, output_size, align_corners, scale_factors):
    if output_size is None:
        output_size = (
            int(input.shape[-2] * scale_factors[0]),
            int(input.shape[-1] * scale_factors[1]),
        )
    return _composite_reference(input, output_size, align_corners)


def _scale(shape):
    return 2.0 if shape[-2] < 64 else 0.5


def _input_fn(shape, dtype, device):
    scale = _scale(shape)
    output_size = (int(shape[-2] * scale), int(shape[-1] * scale))
    yield torch.randn(shape, device=device, dtype=dtype), output_size, False, None, None


@pytest.mark.upsample_lanczos2d_aa
def test_upsample_lanczos2d_aa():
    bench = UpsampleLanczosBenchmark(
        input_fn=_input_fn,
        op_name="upsample_lanczos2d_aa",
        torch_op=_composite_reference,
        gems_op=_upsample_lanczos2d_aa,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_out(shape, dtype, device):
    scale = _scale(shape)
    output_size = (int(shape[-2] * scale), int(shape[-1] * scale))
    input = torch.randn(shape, device=device, dtype=dtype)
    out = torch.empty((*shape[:2], *output_size), device=device, dtype=dtype)
    yield input, output_size, False, None, None, {"out": out}


@pytest.mark.upsample_lanczos2d_aa_out
def test_upsample_lanczos2d_aa_out():
    bench = UpsampleLanczosBenchmark(
        input_fn=_input_fn_out,
        op_name="upsample_lanczos2d_aa_out",
        torch_op=_composite_reference_out,
        gems_op=_upsample_lanczos2d_aa_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_vec(shape, dtype, device):
    scale = _scale(shape)
    yield torch.randn(shape, device=device, dtype=dtype), None, False, (scale, scale)


@pytest.mark.upsample_lanczos2d_aa_vec
def test_upsample_lanczos2d_aa_vec():
    bench = UpsampleLanczosBenchmark(
        input_fn=_input_fn_vec,
        op_name="upsample_lanczos2d_aa_vec",
        torch_op=_composite_reference_vec,
        gems_op=_upsample_lanczos2d_aa_vec,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
